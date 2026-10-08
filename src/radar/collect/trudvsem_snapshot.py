"""Backfill from the daily XML snapshots of the trudvsem open-data set.

The dataset passport lists about 100 daily versions and the window slides: the oldest one goes
away as a new one arrives, so versions are taken oldest first. Each version (3-4 GB of XML) is
downloaded to a temporary file with resumable Range requests, parsed with `iterparse`, cleaned of
personal data and written whole (all professions; the IT filter lives in staging, so it can be
changed later) to `data/raw/trudvsem_snapshot/dt=<snapshot date>/`. The XML is deleted right after
its partition is complete. A version whose partition has `_SUCCESS` is skipped, so a rerun
continues where the last one stopped.

    RADAR_USER_AGENT='...' python -m radar.collect.trudvsem_snapshot [--limit N]
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import httpx
import psutil
from lxml import etree
from pydantic import ValidationError

from radar.collect.contracts import TrudvsemSnapshotVacancy
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.writer import PartitionWriter, is_complete, partition_dir, raw_root, read_marker
from radar.config import RESULTS, data_dir, user_agent
from radar.pii import clean_trudvsem_snapshot
from radar.results import write_result

log = logging.getLogger("radar.collect.trudvsem_snapshot")

SOURCE = "trudvsem_snapshot"
PASSPORT_URL = "https://opendata.trudvsem.ru/7710538364-vacancy/"
RESULT_PATH = "collect/trudvsem_snapshots.json"
PAUSE_S = 2.0
BACKOFF_S = (10, 30, 90, 270, 810)
CHUNK = 1 << 20

# Elements that are lists even when they hold one child.
LIST_CONTAINERS = frozenset(
    {
        "contactList",
        "skills",
        "hardSkills",
        "softSkills",
        "languageKnowledge",
        "shift",
        "requiredDriveLicense",
        "workPlaceSocialProtected",
    }
)
# RDF links that repeat codes stored elsewhere in the record; `organization` carries the OGRNIP of a
# sole proprietor, so they are not kept.
REFERENCES = frozenset({"region", "organization", "profession", "federal_district"})
# Free-text fields go to `text` as labelled sections and leave the payload. In the snapshots
# `requirements`, `qualifications` and `positionRequirements` are usually identical; a section equal
# to an earlier one is not repeated.
TEXT_SECTIONS = (
    ("responsibilities", "Обязанности"),
    ("requirements", "Требования"),
    ("qualifications", "Квалификация"),
    ("positionRequirements", "Требования к должности"),
    ("additionalRequirements", "Дополнительные требования"),
    ("benefitDetails", "Условия"),
    ("otherVacancyBenefit", "Другие льготы"),
)
# Stamps that move without the vacancy changing; they do not enter the content hash.
VOLATILE = ("dateModify", "changeTime", "datePublished", "snapshot_version")

# XML 1.0 forbids C0 controls other than tab, LF and CR, and the snapshots contain some. These bytes
# never occur inside multi-byte UTF-8 sequences, so dropping them from the byte stream is safe.
_BAD_BYTES = bytes(b for b in range(32) if b not in (9, 10, 13))


class GoneError(RuntimeError):
    """The version is no longer served (the sliding window moved past it)."""


@dataclass(frozen=True)
class Version:
    created: str
    url: str

    @property
    def dt(self) -> date:
        return datetime.strptime(self.created[:8], "%Y%m%d").date()


class Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class _ControlCharFilter:
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.dropped = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        cleaned = chunk.translate(None, _BAD_BYTES)
        self.dropped += len(chunk) - len(cleaned)
        return cleaned


def parse_passport(xml: bytes) -> list[Version]:
    root = etree.fromstring(xml)
    versions = [
        Version(created=(dv.findtext("created") or "").strip(), url=(dv.findtext("source") or "").strip())
        for dv in root.iter("dataversion")
    ]
    return sorted((v for v in versions if v.created and v.url.endswith(".xml")), key=lambda v: v.created)


def element_to_value(el: Any) -> Any:
    children = list(el)
    if not children:
        text = (el.text or "").strip()
        return text or None
    if etree.QName(el).localname in LIST_CONTAINERS:
        return [v for v in (element_to_value(c) for c in children) if v is not None]
    out: dict[str, Any] = {}
    for child in children:
        key = etree.QName(child).localname
        value = element_to_value(child)
        if value is None:
            continue
        if key in out:
            out[key] = out[key] if isinstance(out[key], list) else [out[key]]
            out[key].append(value)
        else:
            out[key] = value
    return out or None


def vacancy_to_dict(el: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child in el:
        key = etree.QName(child).localname
        if key in REFERENCES:
            continue
        value = element_to_value(child)
        if value is not None:
            out[key] = value
    return out


def iter_vacancies(stream: Readable) -> Iterator[dict[str, Any]]:
    for _, el in etree.iterparse(stream, events=("end",), tag="vacancy", huge_tree=True):
        yield vacancy_to_dict(el)
        el.clear(keep_tail=False)
        parent = el.getparent()
        while el.getprevious() is not None and parent is not None:
            del parent[0]


def to_record(raw: dict[str, Any], version: Version, fetched_at: datetime) -> RawRecord:
    contract = TrudvsemSnapshotVacancy.model_validate(raw)
    cleaned, removed = clean_trudvsem_snapshot(raw)
    sections: list[str] = []
    seen: set[str] = set()
    for key, label in TEXT_SECTIONS:
        value = cleaned.pop(key, None)
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            sections.append(f"{label}: {value}")
    title = cleaned.get("vacancyName")
    text = "\n\n".join(sections) or None
    cleaned["snapshot_version"] = version.created
    return RawRecord(
        source=SOURCE,
        source_id=contract.id,
        fetched_at=fetched_at,
        published_at=contract.creationDate,
        modified_at=contract.dateModify,
        url=cleaned.get("vacancyUrl"),
        title=title,
        text=text,
        payload=cleaned,
        content_hash=content_hash(title, text, cleaned, ignore=VOLATILE),
        pii_removed=dict(removed),
    )


def _get(client: httpx.Client, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    for attempt, pause in enumerate((*BACKOFF_S, None)):
        try:
            response = client.get(url, headers=headers)
            if response.status_code == 404:
                raise GoneError(url)
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if pause is None:
                raise
            log.warning("GET %s failed (%s), attempt %d, retry in %ds", url, exc, attempt + 1, pause)
            time.sleep(pause)
    raise AssertionError("unreachable")


def fetch_versions(client: httpx.Client) -> list[Version]:
    return parse_passport(_get(client, PASSPORT_URL).content)


@dataclass
class Download:
    path: Path
    fetched_at: datetime
    bytes_this_run: int
    seconds: float
    reused: bool


def _fetched_at(stamp: Path) -> datetime:
    return datetime.fromisoformat(stamp.read_text(encoding="utf-8").strip())


def download(client: httpx.Client, version: Version, tmp_dir: Path) -> Download:
    """Download one version to `tmp_dir`, resuming a partial file; one stream, never parallel.

    `fetched_at` is when the first request for this file was sent, kept in a sidecar file so a
    resumed download keeps it.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = tmp_dir / f"data-{version.created}.xml"
    part = final.with_suffix(".xml.part")
    stamp = final.with_suffix(".fetched_at")
    head = client.head(version.url)
    if head.status_code == 404:
        raise GoneError(version.url)
    head.raise_for_status()
    if "content-length" not in head.headers:
        raise RuntimeError(f"{version.url}: no Content-Length, cannot resume or verify the download")
    total = int(head.headers["content-length"])
    if final.exists() and final.stat().st_size == total and stamp.exists():
        return Download(final, _fetched_at(stamp), 0, 0.0, reused=True)
    free = shutil.disk_usage(tmp_dir).free
    if free < 2 * total:
        raise RuntimeError(f"not enough disk for {final.name}: {free} free, {total} needed twice")
    if not stamp.exists():
        stamp.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    started = time.monotonic()
    fetched = 0
    for attempt, pause in enumerate((*BACKOFF_S, None)):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", version.url, headers=headers) as response:
                if response.status_code == 404:
                    raise GoneError(version.url)
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise RuntimeError(f"{version.url}: encoded response, byte offsets would not match the file")
                mode = "ab" if have and response.status_code == 206 else "wb"
                with part.open(mode) as f:
                    for chunk in response.iter_bytes(CHUNK):
                        f.write(chunk)
                        fetched += len(chunk)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if pause is None:
                raise
            log.warning(
                "download %s broke at %d bytes (%s), retry %d in %ds", final.name, have, exc, attempt + 1, pause
            )
            time.sleep(pause)
    size = part.stat().st_size
    if size != total:
        raise RuntimeError(f"{final.name}: got {size} bytes, expected {total}")
    part.replace(final)
    return Download(final, _fetched_at(stamp), fetched, time.monotonic() - started, reused=False)


def process_file(path: Path, version: Version, fetched_at: datetime, root: Path | None = None) -> dict[str, Any]:
    """Parse one snapshot file into its partition; returns the per-snapshot summary."""
    proc = psutil.Process()
    pii: Counter[str] = Counter()
    started = time.monotonic()
    writer = PartitionWriter(SOURCE, version.dt, root=root)
    with path.open("rb") as raw:
        stream = _ControlCharFilter(raw)
        try:
            vacancies, rss_max = _fill(writer, iter_vacancies(stream), version, fetched_at, pii, proc, started)
        except BaseException:
            # Keep what was parsed, without the completion marker: a rerun skips these keys.
            writer.close(complete=False)
            raise
    writer.flush()
    rss_max = max(rss_max, proc.memory_info().rss)
    summary = {
        "version": version.created,
        "dt": version.dt.isoformat(),
        "vacancies_in_file": vacancies,
        "written": writer.written,
        "duplicates_skipped": writer.duplicates,
        "rejected": writer.rejected,
        "control_bytes_dropped": stream.dropped,
        "pii_removed": dict(sorted(pii.items())),
        "seconds_parse": round(time.monotonic() - started, 1),
        "rss_sampled_max_mb": round(rss_max / 2**20),
    }
    writer.close(complete=True, **summary)
    summary["parquet_bytes"] = sum(
        p.stat().st_size for p in partition_dir(SOURCE, version.dt, root).glob("part-*.parquet")
    )
    return summary


def _fill(
    writer: PartitionWriter,
    items: Iterator[dict[str, Any]],
    version: Version,
    fetched_at: datetime,
    pii: Counter[str],
    proc: psutil.Process,
    started: float,
) -> tuple[int, int]:
    vacancies = 0
    rss_max = proc.memory_info().rss
    for item in items:
        vacancies += 1
        try:
            record = to_record(item, version, fetched_at)
        except ValidationError as exc:
            first = exc.errors()[0]
            source_id = item.get("id") if isinstance(item.get("id"), str) else None
            # Error type only: the message quotes the input, which may be personal data.
            writer.reject(source_id, ".".join(str(p) for p in first["loc"]) + ": " + first["type"])
            continue
        # Counted for every valid record of the file, written now or by an interrupted earlier run.
        pii.update(record.pii_removed)
        writer.add(record)
        if vacancies % 50_000 == 0:
            rss_max = max(rss_max, proc.memory_info().rss)
            log.info("%s: %d vacancies parsed, %.0f s", version.created, vacancies, time.monotonic() - started)
    return vacancies, rss_max


def _load_summaries() -> dict[str, dict[str, Any]]:
    path = RESULTS / RESULT_PATH
    if not path.exists():
        return {}
    return {s["version"]: s for s in json.loads(path.read_text(encoding="utf-8")).get("snapshots", [])}


def _save_summaries(summaries: dict[str, dict[str, Any]]) -> None:
    done = [s for s in summaries.values() if s.get("status") == "done"]
    pii_total: Counter[str] = Counter()
    for s in done:
        pii_total.update(s.get("pii_removed", {}))
    metrics: dict[str, dict[str, Any]] = {
        "snapshots_done": {"value": len(done)},
        "vacancies_total": {"value": sum(s["vacancies_in_file"] for s in done)},
        "bytes_downloaded_total": {"value": sum(s.get("bytes_downloaded", 0) for s in done)},
        "parquet_bytes_total": {"value": sum(s.get("parquet_bytes", 0) for s in done)},
    }
    metrics.update({f"pii_removed_{k}": {"value": v} for k, v in sorted(pii_total.items())})
    write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/collect/trudvsem_snapshot.py",
        snapshots=sorted(summaries.values(), key=lambda s: s["version"]),
    )


def run(limit: int | None, keep_xml: bool, tmp_dir: Path) -> int:
    # Identity encoding: with gzip the server streams without a length and byte ranges stop lining up.
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "identity"}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=True) as client:
        return _run(client, limit, keep_xml, tmp_dir)


def _run(client: httpx.Client, limit: int | None, keep_xml: bool, tmp_dir: Path) -> int:
    versions = fetch_versions(client)
    log.info("passport lists %d versions, %s .. %s", len(versions), versions[0].created, versions[-1].created)
    summaries = _load_summaries()
    processed = 0
    for version in versions:
        if is_complete(SOURCE, version.dt):
            marker = read_marker(SOURCE, version.dt) or {}
            if marker.get("version") != version.created:
                log.warning(
                    "dt=%s is complete from version %s, passport now has %s; skipped",
                    version.dt,
                    marker.get("version"),
                    version.created,
                )
            continue
        if limit is not None and processed >= limit:
            break
        time.sleep(PAUSE_S)
        log.info("version %s: download", version.created)
        try:
            dl = download(client, version, tmp_dir)
        except GoneError:
            log.warning("version %s is gone from the server", version.created)
            summaries[version.created] = {"version": version.created, "dt": version.dt.isoformat(), "status": "gone"}
            _save_summaries(summaries)
            continue
        log.info(
            "version %s: %d bytes in %.0f s%s, parse",
            version.created,
            dl.path.stat().st_size,
            dl.seconds,
            " (file from an earlier run)" if dl.reused else "",
        )
        summary = process_file(dl.path, version, dl.fetched_at)
        summary.update(
            status="done",
            file_bytes=dl.path.stat().st_size,
            bytes_downloaded=dl.bytes_this_run,
            seconds_download=round(dl.seconds, 1),
            reused_local_file=dl.reused,
            fetched_at=dl.fetched_at.isoformat(),
        )
        summaries[version.created] = summary
        _save_summaries(summaries)
        log.info(
            "version %s: done, %d vacancies, %d written, %d rejected, parquet %d bytes, %.0f s parse",
            version.created,
            summary["vacancies_in_file"],
            summary["written"],
            summary["rejected"],
            summary["parquet_bytes"],
            summary["seconds_parse"],
        )
        if not keep_xml:
            dl.path.unlink()
            dl.path.with_suffix(".fetched_at").unlink()
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--limit", type=int, default=None, help="process at most N versions in this run")
    parser.add_argument("--keep-xml", action="store_true", help="keep the downloaded XML after parsing")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="where XML is downloaded (default data/tmp)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    tmp_dir = args.tmp_dir or data_dir() / "tmp" / SOURCE
    log.info("raw root %s, tmp %s", raw_root(), tmp_dir)
    processed = run(args.limit, args.keep_xml, tmp_dir)
    log.info("finished, %d versions processed in this run", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

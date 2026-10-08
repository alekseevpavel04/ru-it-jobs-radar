"""Backfill from the daily XML snapshots of the trudvsem open-data set.

The dataset passport lists about 100 daily versions and the window slides: the oldest one goes
away as a new one arrives, so versions are taken oldest first. Each version (3-4 GB of XML) is
downloaded to a temporary file with resumable Range requests, parsed with `iterparse`, cleaned of
personal data and written whole (all professions; the IT filter lives in staging, so it can be
changed later) to `data/raw/trudvsem_snapshot/dt=<snapshot date>/`. The XML is deleted right after
its partition is complete. A version whose partition has `_SUCCESS` is skipped, so a rerun
continues where the last one stopped.

The results file is rebuilt from disk after every snapshot (`--summarize` does only that), and the
partition is published to the topic `raw.trudvsem_snapshot`; a stream that is down costs a warning.
`--repair DT` appends records missing from a complete partition, from a local XML of the same
version, without touching what is already written. `--since DATE` leaves versions dated earlier to
another machine: the daily server takes new versions while the backfill of old ones runs elsewhere.

    RADAR_USER_AGENT='...' python -m radar.collect.trudvsem_snapshot [--limit N] [--since 2026-10-09]
    python -m radar.collect.trudvsem_snapshot --summarize
    python -m radar.collect.trudvsem_snapshot --repair 2026-06-30
"""

import argparse
import json
import logging
import re
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
import pyarrow.parquet as pq
from lxml import etree
from pydantic import ValidationError

from radar.collect import setup_logging
from radar.collect.contracts import TrudvsemSnapshotVacancy
from radar.collect.envelope import RawRecord, content_hash
from radar.collect.lock import AlreadyRunning, single_instance
from radar.collect.summary import append_meta, meta_dir, partition_dates, partition_stats, read_meta
from radar.collect.writer import PartitionWriter, is_complete, partition_dir, raw_root, read_marker
from radar.config import ROOT, data_dir, user_agent
from radar.pii import clean_trudvsem_snapshot
from radar.results import write_result
from radar.stream.producer import publish_after_collect

log = logging.getLogger("radar.collect.trudvsem_snapshot")

SOURCE = "trudvsem_snapshot"
PASSPORT_URL = "https://opendata.trudvsem.ru/7710538364-vacancy/"
RESULT_PATH = "collect/trudvsem_snapshots.json"
LOG_PATH = ROOT / "logs" / "trudvsem_snapshot.log"
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
    reused: bool
    seconds: float | None
    bytes_transferred: int | None
    runs: int | None


def _fetched_at(stamp: Path) -> datetime:
    return datetime.fromisoformat(stamp.read_text(encoding="utf-8").strip())


def _timing(path: Path) -> dict[str, Any] | None:
    loaded: dict[str, Any] | None = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return loaded


def _head(client: httpx.Client, url: str) -> httpx.Response:
    """HEAD with the download's backoff: the server once answered 405 to a HEAD that worked minutes later."""
    for attempt, pause in enumerate((*BACKOFF_S, None)):
        try:
            head = client.head(url)
        except httpx.TransportError as exc:
            problem = f"{type(exc).__name__}: {exc}"
        else:
            if head.status_code == 404:
                raise GoneError(url)
            if head.is_success:
                return head
            problem = f"HTTP {head.status_code}"
        if pause is None:
            raise RuntimeError(f"HEAD {url} failed after {attempt + 1} attempts: {problem}")
        log.warning("HEAD %s: %s, retry %d in %ds", url, problem, attempt + 1, pause)
        time.sleep(pause)
    raise AssertionError("unreachable")


def download(client: httpx.Client, version: Version, tmp_dir: Path) -> Download:
    """Download one version to `tmp_dir`, resuming a partial file; one stream, never parallel.

    `fetched_at` is when the first request for this file was sent, kept in a sidecar file so a
    resumed download keeps it. Transfer time and bytes add up over runs in a second sidecar; a file
    that came without it (downloaded by other means) has no timing.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final = tmp_dir / f"data-{version.created}.xml"
    part = final.with_suffix(".xml.part")
    stamp = final.with_suffix(".fetched_at")
    timing_path = final.with_suffix(".download.json")
    head = _head(client, version.url)
    if "content-length" not in head.headers:
        raise RuntimeError(f"{version.url}: no Content-Length, cannot resume or verify the download")
    total = int(head.headers["content-length"])
    if final.exists() and final.stat().st_size == total and stamp.exists():
        timing = _timing(timing_path)
        if timing is None:
            return Download(final, _fetched_at(stamp), True, None, None, None)
        return Download(final, _fetched_at(stamp), True, timing["seconds"], timing["bytes"], timing["runs"])
    free = shutil.disk_usage(tmp_dir).free
    if free < 2 * total:
        raise RuntimeError(f"not enough disk for {final.name}: {free} free, {total} needed twice")
    if not stamp.exists():
        stamp.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    timing = _timing(timing_path) or {"seconds": 0.0, "bytes": 0, "runs": 0}
    if part.exists() and timing["bytes"] < part.stat().st_size:
        # Part of the file came from a run that kept no timing: the total time is unknown.
        timing["seconds"] = None
    timing["runs"] += 1
    started = time.monotonic()
    fetched = 0
    try:
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
    finally:
        if timing["seconds"] is not None:
            timing["seconds"] = round(timing["seconds"] + time.monotonic() - started, 1)
        timing["bytes"] += fetched
        timing_path.write_text(json.dumps(timing), encoding="utf-8")
    size = part.stat().st_size
    if size != total:
        raise RuntimeError(f"{final.name}: got {size} bytes, expected {total}")
    part.replace(final)
    return Download(final, _fetched_at(stamp), False, timing["seconds"], timing["bytes"], timing["runs"])


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


_LOG_DOWNLOAD = re.compile(r"INFO version (\d{8}T\d{6}): download$")
_LOG_FETCHED = re.compile(r"INFO version (\d{8}T\d{6}): (\d+) bytes in (\d+) s(.*), parse$")


def downloads_from_log(log_path: Path) -> dict[str, dict[str, Any]]:
    """Download timings logged by runs that kept no download records.

    A timing counts only when the transfer was not interrupted: exactly one `download` line for the
    version before its `bytes in` line, and the file was not reused from an earlier run.
    """
    if not log_path.exists():
        return {}
    starts: Counter[str] = Counter()
    found: dict[str, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := _LOG_DOWNLOAD.search(line):
            starts[m.group(1)] += 1
        elif (m := _LOG_FETCHED.search(line)) and "earlier run" not in m.group(4):
            version = m.group(1)
            if starts[version] == 1 and version not in found:
                found[version] = {"file_bytes": int(m.group(2)), "seconds": int(m.group(3)), "source": "log"}
            starts[version] = 0
    return found


def _download_entry(version: str, log_timings: dict[str, dict[str, Any]], root: Path | None) -> dict[str, Any]:
    recorded = [d for d in read_meta(SOURCE, "downloads", root) if d["version"] == version]
    if recorded and recorded[-1].get("seconds") is not None:
        d = recorded[-1]
        return {"download": {"file_bytes": d["file_bytes"], "seconds": d["seconds"], "source": "collector"}}
    if version in log_timings:
        return {"download": log_timings[version], "download_note": "seconds rounded, from the collector log line"}
    return {
        "download": None,
        "download_note": "not measured: the file was resumed across runs, reused, or fetched outside the collector",
    }


def _repair_record(dt: date, root: Path | None) -> dict[str, Any] | None:
    path = meta_dir(SOURCE, root) / f"repair-dt={dt.isoformat()}.json"
    loaded: dict[str, Any] | None = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return loaded


def summarize(root: Path | None = None, log_path: Path | None = None) -> Path:
    """Rebuild `results/collect/trudvsem_snapshots.json` from the partitions on disk."""
    log_timings = downloads_from_log(log_path or LOG_PATH)
    snapshots: list[dict[str, Any]] = []
    for dt in partition_dates(SOURCE, root):
        stats = partition_stats(SOURCE, dt, root)
        marker = stats["marker"] or {}
        entry: dict[str, Any] = {
            "dt": stats["dt"],
            "version": marker.get("version"),
            "complete": stats["complete"],
            "rows": stats["rows"],
            "vacancies_in_file": marker.get("vacancies_in_file"),
            "rejected_not_written": stats["rejected_not_written"],
            "rejected_ids": stats["rejected_ids"],
            "parquet_bytes": stats["parquet_bytes"],
            "pii_removed": stats["pii_removed"],
            "pii_method": "sum of the pii_removed column",
            "seconds_parse": marker.get("seconds_parse"),
            "rss_sampled_max_mb": marker.get("rss_sampled_max_mb"),
            "control_bytes_dropped": marker.get("control_bytes_dropped"),
        }
        repair_record = _repair_record(dt, root)
        if repair_record is not None:
            entry["repair"] = repair_record
            entry["pii_removed"] = repair_record["pii_removed"]
            entry["pii_method"] = repair_record["pii_method"]
        if entry["version"]:
            entry.update(_download_entry(entry["version"], log_timings, root))
        snapshots.append(entry)
    done = [e for e in snapshots if e["complete"]]
    in_files = sum(e["vacancies_in_file"] or 0 for e in done)
    lost = sum(e["rejected_not_written"] for e in done)
    timed = [e for e in done if e.get("download")]
    pii_total: Counter[str] = Counter()
    for e in done:
        pii_total.update(e["pii_removed"])
    metrics: dict[str, dict[str, Any]] = {
        "snapshots_complete": {"value": len(done)},
        "rows_total": {"value": sum(e["rows"] for e in done)},
        "vacancies_in_files_total": {"value": in_files},
        "rows_lost": {"value": round(lost / in_files, 9) if in_files else 0, "n": lost, "denominator": in_files},
        "parquet_bytes_total": {"value": sum(e["parquet_bytes"] for e in done)},
        "downloads_timed": {"value": len(timed), "n": len(timed), "denominator": len(done)},
    }
    metrics.update({f"pii_removed_{k}": {"value": v} for k, v in sorted(pii_total.items())})
    return write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/collect/trudvsem_snapshot.py",
        method="recomputed from data/raw/trudvsem_snapshot partitions, their markers, rejects logs and _meta",
        snapshots=snapshots,
        gone=read_meta(SOURCE, "gone", root),
    )


def _local_version(dt: date, tmp_dir: Path, root: Path | None) -> tuple[Version, Path, datetime]:
    marker = read_marker(SOURCE, dt, root) or {}
    created = marker.get("version")
    if not created:
        raise RuntimeError(f"dt={dt} has no completed partition to repair")
    path = tmp_dir / f"data-{created}.xml"
    stamp = path.with_suffix(".fetched_at")
    if not path.exists() or not stamp.exists():
        raise RuntimeError(f"no local XML for version {created} in {tmp_dir}")
    return Version(created, ""), path, _fetched_at(stamp)


def repair(dt: date, tmp_dir: Path, root: Path | None = None) -> dict[str, Any]:
    """Append records missing from a complete partition, using the local XML of the same version.

    Existing parts and the marker are not touched: missing ids go to a new part file. Ids that are
    present with another content hash are counted, not written. `pii_removed` is recomputed over
    the whole file with the current cleaner and stored next to the partition.
    """
    version, path, fetched_at = _local_version(dt, tmp_dir, root)
    directory = partition_dir(SOURCE, dt, root)
    present: dict[str, set[str]] = {}
    for part in sorted(directory.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["source_id", "content_hash"])
        ids = table.column("source_id").to_pylist()
        for sid, digest in zip(ids, table.column("content_hash").to_pylist(), strict=True):
            present.setdefault(str(sid), set()).add(str(digest))
    pii: Counter[str] = Counter()
    rejected = hash_mismatch = vacancies = 0
    appended: list[str] = []
    writer = PartitionWriter(SOURCE, dt, root=root)
    with path.open("rb") as raw:
        for item in iter_vacancies(_ControlCharFilter(raw)):
            vacancies += 1
            try:
                record = to_record(item, version, fetched_at)
            except ValidationError:
                rejected += 1
                continue
            pii.update(record.pii_removed)
            if record.source_id not in present:
                writer.add(record)
                appended.append(record.source_id)
            elif record.content_hash not in present[record.source_id]:
                hash_mismatch += 1
    writer.close(complete=False)
    result = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": version.created,
        "method": "local XML of the same version reparsed; ids absent from the partition appended as a new part",
        "vacancies_in_file": vacancies,
        "rows_appended": len(appended),
        "appended_ids": appended,
        "present_with_other_hash": hash_mismatch,
        "still_rejected": rejected,
        "pii_removed": dict(sorted(pii.items())),
        "pii_method": "recomputed over the local XML with the current cleaner; the partition column was written "
        "by a cleaner that also counted contact entries without a value",
    }
    target = meta_dir(SOURCE, root) / f"repair-dt={dt.isoformat()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def run(limit: int | None, keep_xml: bool, tmp_dir: Path, since: date | None = None) -> int:
    # Identity encoding: with gzip the server streams without a length and byte ranges stop lining up.
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "identity"}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=True) as client:
        return _run(client, limit, keep_xml, tmp_dir, since)


def _run(client: httpx.Client, limit: int | None, keep_xml: bool, tmp_dir: Path, since: date | None = None) -> int:
    versions = fetch_versions(client)
    log.info("passport lists %d versions, %s .. %s", len(versions), versions[0].created, versions[-1].created)
    if since is not None:
        versions = [v for v in versions if v.dt >= since]
        log.info("%d versions dated %s or later", len(versions), since)
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
            append_meta(SOURCE, "gone", {"version": version.created, "at": datetime.now(UTC).isoformat()})
            continue
        log.info(
            "version %s: %d bytes in %s s%s, parse",
            version.created,
            dl.path.stat().st_size,
            "?" if dl.seconds is None else f"{dl.seconds:.0f}",
            " (file from an earlier run)" if dl.reused else "",
        )
        summary = process_file(dl.path, version, dl.fetched_at)
        append_meta(
            SOURCE,
            "downloads",
            {
                "version": version.created,
                "file_bytes": dl.path.stat().st_size,
                "seconds": dl.seconds,
                "bytes_transferred": dl.bytes_transferred,
                "runs": dl.runs,
                "fetched_at": dl.fetched_at.isoformat(),
            },
        )
        summarize()
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
            for suffix in (".xml", ".fetched_at", ".download.json"):
                dl.path.with_suffix(suffix).unlink(missing_ok=True)
        publish_after_collect(SOURCE, version.dt)
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--limit", type=int, default=None, help="process at most N versions in this run")
    parser.add_argument("--keep-xml", action="store_true", help="keep the downloaded XML after parsing")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="where XML is downloaded (default data/tmp)")
    parser.add_argument("--summarize", action="store_true", help="only rebuild the results file from disk")
    parser.add_argument("--repair", type=date.fromisoformat, metavar="YYYY-MM-DD", help="append missing records")
    parser.add_argument(
        "--since", type=date.fromisoformat, metavar="YYYY-MM-DD", help="skip versions dated earlier than this"
    )
    args = parser.parse_args(argv)
    setup_logging()
    tmp_dir = args.tmp_dir or data_dir() / "tmp" / SOURCE
    if args.repair:
        result = repair(args.repair, tmp_dir)
        log.info(
            "repair dt=%s: %d appended, %d present with another hash, %d still rejected",
            args.repair,
            result["rows_appended"],
            result["present_with_other_hash"],
            result["still_rejected"],
        )
    if args.repair or args.summarize:
        log.info("results: %s", summarize())
        return 0
    log.info("raw root %s, tmp %s", raw_root(), tmp_dir)
    try:
        with single_instance(tmp_dir / "collector.lock"):
            processed = run(args.limit, args.keep_xml, tmp_dir, args.since)
    except AlreadyRunning as exc:
        log.info("another run is in progress, nothing to do: %s", exc)
        return 0
    log.info("finished, %d versions processed in this run", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

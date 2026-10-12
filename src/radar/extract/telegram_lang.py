"""What the channel language rule keeps (remoteit counts only its Russian posts).

A channel with `lang` in configs/telegram_channels.toml counts only posts whose own text is in that
language (`radar.extract.telegram_corpus.post_lang`; dbt macro post_lang does the same in
stg_telegram). The raw layer keeps every post. This counts, per such channel, the posts of the
complete raw partitions (latest version of each post) that the rule keeps and drops, all posts and
vacancy posts, and checks that no frozen Telegram test or dev record is one it drops: those samples
do not change.

    python -m radar.extract.telegram_lang
"""

import argparse
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from radar.extract.corpus import SPLITS, share
from radar.extract.telegram_corpus import (
    FROZEN_DIR,
    SOURCE,
    channel_langs,
    load_config,
    own_text,
    partitions,
    post_lang,
)

log = logging.getLogger("radar.extract.telegram_lang")

RESULT = "corpus/telegram_lang.json"
COLUMNS = ["source_id", "fetched_at", "content_hash", "is_vacancy", "text"]
LANGS = ("ru", "en", "other")
_CYRILLIC_WORD = re.compile(r"[а-яё]{3,}")


def channel_rows(root: Path, days: Sequence[str], channels: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Raw rows of these channels, read in batches with only the columns the rule needs."""
    prefixes = [f"{c}/" for c in channels]
    for day in days:
        for path in sorted((root / SOURCE / f"dt={day}").glob("part-*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=8192):
                ids = batch.column("source_id")
                mask = pc.starts_with(ids, prefixes[0])
                for p in prefixes[1:]:
                    mask = pc.or_(mask, pc.starts_with(ids, p))
                for row in batch.filter(mask).to_pylist():
                    row["dt"] = day
                    yield row


def latest(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The latest stored version of every post, as radar.extract.telegram_corpus.read_posts picks it."""
    best: dict[str, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["fetched_at"]), str(row["dt"]), str(row["content_hash"]))
        cur = best.get(row["source_id"])
        if cur is None or key > cur[0]:
            best[row["source_id"]] = (key, row)
    return [best[k][1] for k in sorted(best)]


def measure(
    posts: Iterable[Mapping[str, Any]], lang_of: Mapping[str, str], min_content_chars: int
) -> dict[str, dict[str, dict[str, Any]]]:
    """Channel -> shares of its posts and vacancy posts the rule keeps and drops, posts per language.

    Vacancy posts with at least `min_content_chars` of own text are those the corpus can take
    (shorter ones are link stubs there). A dropped short post with a Cyrillic word is the rule's known
    miss: a title like "Senior/Middle frontend-разработчик (Vue)" has more Latin than Cyrillic."""
    counts: dict[str, Counter[str]] = {c: Counter() for c in lang_of}
    for post in posts:
        channel = str(post["source_id"]).rsplit("/", 1)[0]
        own = own_text(post["text"])
        lang = post_lang(post["text"])
        kept = lang == lang_of[channel]
        c = counts[channel]
        c["posts"] += 1
        c[f"lang.{lang}"] += 1
        c["kept"] += kept
        if post["is_vacancy"]:
            c["vacancy_posts"] += 1
            c["vacancy_kept"] += kept
            if len(own) >= min_content_chars:
                c["vacancy_text"] += 1
                c["vacancy_text_kept"] += kept
            elif not kept and _CYRILLIC_WORD.search(own.lower()):
                c["vacancy_dropped_short_cyrillic"] += 1
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for channel, c in sorted(counts.items()):
        m = out[channel] = {}
        m["posts_kept"] = share(c["kept"], c["posts"])
        m["posts_dropped"] = share(c["posts"] - c["kept"], c["posts"])
        for lang in LANGS:
            m[f"posts_{lang}"] = share(c[f"lang.{lang}"], c["posts"])
        m["vacancy_posts_kept"] = share(c["vacancy_kept"], c["vacancy_posts"])
        m["vacancy_posts_dropped"] = share(c["vacancy_posts"] - c["vacancy_kept"], c["vacancy_posts"])
        m["vacancy_posts_with_text_kept"] = share(c["vacancy_text_kept"], c["vacancy_text"])
        m["vacancy_posts_dropped_short_with_cyrillic_word"] = share(
            c["vacancy_dropped_short_cyrillic"], c["vacancy_posts"] - c["vacancy_kept"]
        )
    return out


def frozen_dropped(directory: Path, lang_of: Mapping[str, str]) -> tuple[int, dict[str, list[str]]]:
    """Frozen test and dev records of the ruled channels, and per split the ids the rule would drop."""
    split_of = {
        r["gold_id"]: r["split"]
        for r in pq.read_table(directory / "split.parquet", columns=["gold_id", "split"]).to_pylist()
        if r["split"] in SPLITS
    }
    records = pq.read_table(directory / "records.parquet", columns=["id", "channel", "input"]).to_pylist()
    ruled = [r for r in records if r["id"] in split_of and r["channel"] in lang_of]
    dropped: dict[str, list[str]] = {s: [] for s in SPLITS}
    for r in ruled:
        if post_lang(r["input"]) != lang_of[r["channel"]]:
            dropped[split_of[r["id"]]].append(r["id"])
    return len(ruled), {s: sorted(ids) for s, ids in dropped.items()}


def main(argv: list[str] | None = None) -> int:
    from radar.collect.writer import raw_root
    from radar.config import data_dir
    from radar.results import write_result

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m radar.extract.telegram_lang", description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    lang_of = channel_langs()
    if not lang_of:
        raise SystemExit("no channel has `lang` in configs/telegram_channels.toml")
    root = raw_root()
    complete, partial = partitions(root)
    posts = latest(channel_rows(root, complete, lang_of))
    min_content_chars = load_config().min_content_chars
    channels = measure(posts, lang_of, min_content_chars)
    n_gold, dropped = frozen_dropped(data_dir() / "corpus" / "telegram" / FROZEN_DIR, lang_of)
    metrics: dict[str, dict[str, Any]] = {"frozen_gold_records_in_ruled_channels": {"value": n_gold}}
    for s in SPLITS:
        metrics[f"frozen_{s}_records_dropped"] = {"value": len(dropped[s])}
    path = write_result(
        RESULT,
        metrics,
        source_script="src/radar/extract/telegram_lang.py",
        config={"channel_lang": lang_of, "min_content_chars": min_content_chars},
        partitions_used=complete,
        partitions_skipped_incomplete=partial,
        channels=channels,
        frozen_dropped_ids=dropped,
    )
    log.info("%s", path)
    if any(dropped.values()):
        log.error("frozen test or dev records the rule drops: %s", dropped)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

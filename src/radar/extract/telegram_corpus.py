"""Telegram extraction records and the frozen Telegram gold samples.

On trudvsem and Habr salary, city and experience come from structured source fields, and the regex
baseline matches the teachers (EXPERIMENTS E4, E5). A Telegram post is free text only, so this is
where an extractor is tested on text. Telegram is part of the training corpus (DECISIONS 10).

Records: one per post, its latest stored version, read from complete raw partitions only (`_SUCCESS`;
the backfill keeps writing others). A record is a post the collector's rules call a vacancy
(`is_vacancy`) with one vacancy in it: not `multi_vacancy` and not a digest the rules count
as one vacancy (`is_digest`: a "Вакансии для ..." title, two job-board keys, three link-only lines),
with at least `min_content_chars` of its own text once links, hashtags and @names are cut: a title
with a link has its vacancy behind the link. Every drop is counted by reason. The input is the post text as
`radar.extract.inputs.input_for_record` renders it for Telegram, without a header and without the
channel name: the Space receives a pasted text and nothing else, and a channel name would let a model
guess the role from where the post was seen. English posts stay: the schema has `lang`. The exception
is a channel the config keeps to one language (`lang` in configs/telegram_channels.toml, remoteit
since 2026-10-12): there a post in another language by `post_lang` is dropped (`channel_lang`).

Groups: cross-posts and reposts are one group, joined by the same normalized text, by a near-duplicate
text (MinHash proposes, the exact Jaccard of word shingles decides) or by the same job-board link
(Habr vacancy id or short link). One record per group goes to gold.

Samples: drawn like S1 (radar.extract.corpus) and separately from it. Strata are channel network x
role proxy; inside a stratum groups are taken in the order of a permanent random number from the seed
of configs/eval.toml, test first, then dev. A group whose record is a near duplicate of a record
already drawn is skipped, so test and dev never hold near duplicates of each other; the inclusion
probability n_h / N_h stored per group is therefore approximate where skips happened (counted).

Kept out of test and dev: groups with posts from repository fixtures (read while writing the parser)
and groups linked to any record of the trudvsem and Habr corpus, by Habr vacancy id or by a near
duplicate input: those vacancies are in the S1 corpus already, so a model trained there has seen them.

Leak check: every test and dev record against every training-pool record; a pool group with a near
duplicate is withheld, and so is a pool group linked to an S1 test or dev record. The check is run
again on the result.

Frozen: `sample` writes `data/corpus/telegram/t1/` and the hashes in results/corpus/telegram.json
(`frozen`), and refuses to redraw once drawn (`--resample`) or once labels exist (never). `refresh`
rebuilds records and groups from the current partitions around the frozen test and dev rows, as
radar.extract.refresh does for S1: a gold record keeps the input it was drawn with unless its current
version differs only by PII placeholders, a new record that joins a gold group's component is
withheld (`<group>_linked`).

    python -m radar.extract.telegram_corpus sample [--resample]
    python -m radar.extract.telegram_corpus refresh
"""

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from radar.collect.telegram_rules import ROLE_RX, title_line
from radar.config import ROOT, data_dir
from radar.dedup.normalize import jaccard, normalize_text, shingles
from radar.extract.baseline import Input, output_for, predict_lang, split_input
from radar.extract.corpus import (
    EVAL_CONFIG,
    SPLITS,
    TRAIN_POOL,
    WITHHELD,
    GroupRow,
    Plan,
    SampleConfig,
    UnionFind,
    id_list_digest,
    permanent_random,
    plan,
    records_in_two_splits,
    share,
    split_digest,
    stratum,
)
from radar.extract.inputs import input_for_record
from radar.extract.refresh import change_kind, frozen_rows_unchanged
from radar.extract.refresh import write_split as write_split_rows
from radar.extract.role_proxy import BUCKETS, ML_DS_DATA, NON_IT_NOISE, role_proxy
from radar.pii.text import PLACEHOLDER

log = logging.getLogger("radar.extract.telegram_corpus")

SOURCE = "telegram"
SUCCESS = "_SUCCESS"
CONFIG = ROOT / "configs" / "telegram_sample.toml"
CHANNELS = ROOT / "configs" / "telegram_channels.toml"
FIXTURES = ROOT / "tests" / "fixtures" / "telegram"
RESULT = "corpus/telegram.json"
FROZEN_DIR = "t1"
GOLD_INPUTS = {s: f"tg_{s}_inputs.jsonl" for s in SPLITS}
GOLD_LABELS = {s: f"tg_{s}.jsonl" for s in SPLITS}
LINKED = "_linked"

RAW_COLUMNS = [
    "source_id",
    "fetched_at",
    "published_at",
    "text",
    "payload",
    "is_vacancy",
    "multi_vacancy",
    "content_hash",
]

_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+[a-z]{2,}/\S*", re.I)
_HASHTAG = re.compile(r"#[^\s#]+")
_HANDLE = re.compile(r"@\w+")
_PLACEHOLDERS = re.compile("|".join(re.escape(p) for p in sorted(set(PLACEHOLDER.values()))))
_HABR_LINK = re.compile(r"(?:career\.habr\.com|moikrug\.ru)/vacancies/(\d+)", re.I)
_HABR_SHORT = re.compile(r"u\.habr\.com/(\w+)", re.I)
_FIXTURE_POST = re.compile(r'data-post="([\w]+/\d+)"')
_DIGEST_TITLE = re.compile(
    r"^\W*(?:(?:свежие|новые|горячие|топ)\W+)?(?:ваканси[йи]|подборк|дайджест)"
    r"|\bваканси[ий] (?:для|в|на|недели|дня)\b|подборк|топ из|собрали",
    re.I,
)
# Openings of lists the rules miss: "Собрали для вас топ из N вакансий", "приглашает на стажировки по
# направлениям:". Only the head: "Подборки в бигтех" is a footer of single-vacancy posts.
_DIGEST_HEAD = re.compile(r"собрали для вас|топ из\W+\w*\W*ваканси|по направлениям\s*:|сразу несколько ваканси", re.I)
DIGEST_HEAD_CHARS = 400
_LINK_LINE = re.compile(r"^\W*(?:https?://|www\.)\S+\s*$", re.M)
# geekjob cards credit the recruiter as "Ищет Имя Фамилия, её пост на LinkedIn"; the PII cleaner
# keeps such names, so they are counted here for the data layer to fix.
_RECRUITER_NAME = re.compile(r"\bИщет\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+")

# Reasons a post is not a record, in the order they are checked.
DROPS = ("not_vacancy", "multi_vacancy", "digest", "link_stub", "channel_lang")


@dataclass(frozen=True)
class Config:
    min_content_chars: int
    group_jaccard: float
    sample: SampleConfig
    baseline_pool_sample: int


def load_config(path: Path = CONFIG, eval_path: Path = EVAL_CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config(
        min_content_chars=int(raw["records"]["min_content_chars"]),
        group_jaccard=float(raw["groups"]["jaccard"]),
        sample=SampleConfig(
            targets={split: {b: int(raw[split][b]) for b in BUCKETS} for split in SPLITS},
            power=float(raw["allocation"]["power"]),
            max_share=float(raw["allocation"]["max_share"]),
            leak_jaccard=float(raw["leak"]["jaccard"]),
            shingle_k=int(raw["leak"]["shingle_k"]),
            num_perm=int(raw["leak"]["num_perm"]),
            seed=int(tomllib.loads(eval_path.read_text(encoding="utf-8"))["bootstrap"]["seed"]),
        ),
        baseline_pool_sample=int(raw["check"]["baseline_pool_sample"]),
    )


def networks(path: Path = CHANNELS) -> dict[str, str]:
    """Channel -> publisher network; a channel without one is a network of its own."""
    channels = tomllib.loads(path.read_text(encoding="utf-8"))["channel"]
    return {str(c["name"]): str(c.get("network") or c["name"]) for c in channels}


def channel_langs(path: Path = CHANNELS) -> dict[str, str]:
    """Channel -> the one language its posts count in (`lang`); a channel without it keeps every post."""
    channels = tomllib.loads(path.read_text(encoding="utf-8"))["channel"]
    return {str(c["name"]): str(c["lang"]) for c in channels if c.get("lang")}


# --- records --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TgRecord:
    id: str
    source_id: str
    channel: str
    network: str
    dt: str
    published_at: str
    content_hash: str
    title: str
    hashtags: tuple[str, ...]
    role_proxy: str
    lang: str
    content_chars: int
    links: tuple[str, ...]
    input: str
    group_id: str = ""


def record_id(source_id: str) -> str:
    return f"{SOURCE}:{source_id}"


def own_text(text: str | None) -> str:
    """The post without links, hashtags, @names and PII placeholders: what the post itself says."""
    out = _URL.sub(" ", text or "")
    out = _HASHTAG.sub(" ", out)
    out = _HANDLE.sub(" ", out)
    out = _PLACEHOLDERS.sub(" ", out)
    return " ".join(out.split())


def post_lang(text: str | None) -> str:
    """Language of what the post itself says, by the letter shares of the regex baseline's `lang`
    (Latin above 60% of letters is English). Links, hashtags and @names are Latin in any post: with
    them a Russian title whose vacancy sits behind a telegra.ph link reads as English. Macro
    post_lang in dbt/macros/telegram_lang.sql repeats this for stg_telegram."""
    return predict_lang(Input({}, own_text(text)))


def hashtags(text: str) -> tuple[str, ...]:
    """Hashtag words: `#data_science` is "data science", Habr's `#backend@habr_career` is "backend"."""
    out = []
    for tag in _HASHTAG.findall(text):
        word = tag[1:].split("@", 1)[0].replace("_", " ").strip(".,;:!?)")
        if word and word not in out:
            out.append(word)
    return tuple(out)


def link_ids(text: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Job-board keys of a post: Habr vacancy ids (also the S1 record id `habr:<id>`) and Habr short
    links, from the collector's `job_links` and from links written out in the text."""
    found = {f"habr:{m}" for m in _HABR_LINK.findall(text)} | {f"u.habr.com/{m}" for m in _HABR_SHORT.findall(text)}
    for link in payload.get("job_links") or ():
        if link.get("habr_vacancy_id"):
            found.add(f"habr:{link['habr_vacancy_id']}")
        if link.get("habr_short"):
            found.add(f"u.habr.com/{link['habr_short']}")
    return tuple(sorted(found))


def telegram_role_proxy(text: str, tags: Sequence[str]) -> str:
    """The S1 proxy over the post's title line with the hashtags as specializations; a post whose
    first line names no role ("Всем привет!", a company name) falls back to its first line with a
    role word among the first ten, where "Позиция: ..." usually sits."""
    bucket = role_proxy(title_line(text), tags)
    if bucket != NON_IT_NOISE:
        return bucket
    line = next((line for line in text.splitlines()[:10] if ROLE_RX.search(line)), "")
    return role_proxy(line, tags) if line else NON_IT_NOISE


def is_digest(text: str, links: Sequence[str]) -> bool:
    """A list of vacancies the rules count as one: a plural "Вакансии ..." or "Подборка" title, a list
    opening in the head, two job-board keys, or three lines that hold only a link (Habr's digests
    through amp.gs and bit.ly)."""
    return (
        bool(_DIGEST_TITLE.search(title_line(text)))
        or bool(_DIGEST_HEAD.search(text[:DIGEST_HEAD_CHARS]))
        or len(links) >= 2
        or len(_LINK_LINE.findall(text)) >= 3
    )


def payload_of(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row["payload"]
    return dict(json.loads(raw) if isinstance(raw, str) else raw or {})


def drop_reason(row: Mapping[str, Any], min_content_chars: int, lang: str | None = None) -> str | None:
    """`lang`: the only language the post's channel counts in (remoteit keeps its Russian posts)."""
    text = str(row["text"] or "")
    if not row["is_vacancy"]:
        return "not_vacancy"
    if row["multi_vacancy"]:
        return "multi_vacancy"
    if is_digest(text, link_ids(text, payload_of(row))):
        return "digest"
    if len(own_text(text)) < min_content_chars:
        return "link_stub"
    if lang and post_lang(text) != lang:
        return "channel_lang"
    return None


def make_record(row: Mapping[str, Any], network_of: Mapping[str, str]) -> TgRecord:
    payload = payload_of(row)
    source_id = str(row["source_id"])
    channel = source_id.rsplit("/", 1)[0]
    text = str(row["text"] or "")
    tags = hashtags(text)
    model_input = input_for_record({"source": SOURCE, "title": None, "text": text, "payload": payload})
    return TgRecord(
        id=record_id(source_id),
        source_id=source_id,
        channel=channel,
        network=network_of.get(channel, channel),
        dt=str(row["dt"]),
        published_at=row["published_at"].isoformat(),
        content_hash=str(row["content_hash"]),
        title=title_line(text),
        hashtags=tags,
        role_proxy=telegram_role_proxy(text, tags),
        lang=predict_lang(split_input(model_input)),
        content_chars=len(own_text(text)),
        links=link_ids(text, payload),
        input=model_input,
    )


def partitions(root: Path) -> tuple[list[str], list[str]]:
    """Days of complete partitions and of partitions still being written."""
    complete: list[str] = []
    partial: list[str] = []
    for d in sorted((root / SOURCE).glob("dt=*")):
        if d.is_dir():
            (complete if (d / SUCCESS).exists() else partial).append(d.name.removeprefix("dt="))
    return complete, partial


def _version(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["fetched_at"]), str(row["dt"]), str(row["content_hash"]))


def read_posts(root: Path, days: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    """The latest stored version of every post in these partitions, and the number of rows read."""
    latest: dict[str, dict[str, Any]] = {}
    rows = 0
    for day in days:
        for path in sorted((root / SOURCE / f"dt={day}").glob("part-*.parquet")):
            for row in pq.read_table(path, columns=RAW_COLUMNS).to_pylist():
                rows += 1
                row["dt"] = day
                cur = latest.get(row["source_id"])
                if cur is None or _version(row) > _version(cur):
                    latest[row["source_id"]] = row
    return [latest[k] for k in sorted(latest)], rows


@dataclass(frozen=True)
class Posts:
    records: list[TgRecord]
    posts: int
    rows: int
    dropped: dict[str, int]
    not_vacancy_kinds: dict[str, int]
    per_channel: dict[str, dict[str, int]]
    published: tuple[str, str]


def build_records(
    rows: Sequence[Mapping[str, Any]],
    n_rows: int,
    network_of: Mapping[str, str],
    cfg: Config,
    lang_of: Mapping[str, str] | None = None,
) -> Posts:
    records = []
    dropped: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    per_channel: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        channel = str(row["source_id"]).rsplit("/", 1)[0]
        per_channel[channel]["posts"] += 1
        per_channel[channel]["vacancy_posts"] += bool(row["is_vacancy"])
        why = drop_reason(row, cfg.min_content_chars, (lang_of or {}).get(channel))
        if why is None:
            records.append(make_record(row, network_of))
            per_channel[channel]["records"] += 1
            continue
        dropped[why] += 1
        per_channel[channel][why] += 1
        if why == "not_vacancy":
            kinds[str(payload_of(row).get("rule_kind"))] += 1
    days = [r.published_at[:10] for r in records]
    return Posts(
        records=records,
        posts=len(rows),
        rows=n_rows,
        dropped={k: dropped[k] for k in DROPS},
        not_vacancy_kinds=dict(sorted(kinds.items())),
        per_channel={
            c: {k: v[k] for k in ("posts", "vacancy_posts", *DROPS, "records")} for c, v in sorted(per_channel.items())
        },
        published=(min(days), max(days)) if days else ("", ""),
    )


# --- groups ---------------------------------------------------------------------------------


def group_docs(records: Sequence[TgRecord], k: int) -> list[set[str]]:
    return [shingles(own_text(r.input), k) for r in records]


def input_docs(records: Iterable[TgRecord], k: int) -> list[set[str]]:
    """Shingles of the extractor input, the unit of every leak check (as in S1)."""
    return [shingles(r.input, k) for r in records]


@dataclass(frozen=True)
class Grouping:
    group_of: dict[str, str]
    edges: dict[str, int]
    # gold group -> new records linked to it (refresh only)
    linked: dict[str, list[str]]
    gold_groups_joined: list[list[str]]


def group_records(records: Sequence[TgRecord], cfg: Config, frozen: Sequence[GroupRow] = ()) -> Grouping:
    """Components over records joined by the same normalized own text, a near duplicate own text
    (Jaccard >= the group threshold) or a shared job-board key. Members of frozen test and dev groups
    stay in them; other records of their component form `<group>_linked`. The rest is named after
    its smallest record id."""
    # MinHash is imported where it is used: the corpus rules (drop_reason) also serve the extraction
    # runner in the WSL training venv, which has no datasketch.
    from radar.dedup.minhash import lsh_pairs

    uf = UnionFind()
    edges: Counter[str] = Counter()
    for r in records:
        uf.find(r.id)
    docs = group_docs(records, cfg.sample.shingle_k)
    by_text: dict[str, str] = {}
    by_link: dict[str, str] = {}
    for r in records:
        key = normalize_text(own_text(r.input))
        if key in by_text:
            uf.union(by_text[key], r.id)
            edges["same_text"] += 1
        else:
            by_text[key] = r.id
        for link in r.links:
            if link in by_link:
                uf.union(by_link[link], r.id)
                edges["same_link"] += 1
            else:
                by_link[link] = r.id
    for i, j in lsh_pairs(docs, cfg.group_jaccard, cfg.sample.num_perm, cfg.sample.seed):
        if jaccard(docs[i], docs[j]) >= cfg.group_jaccard:
            uf.union(records[i].id, records[j].id)
            edges["near_text"] += 1
    frozen_of = {m: row.group_id for row in frozen for m in row.member_ids}
    for r in records:
        if r.id in frozen_of:
            uf.union(r.id, f"g:{frozen_of[r.id]}")

    components: dict[str, list[str]] = defaultdict(list)
    for r in records:
        components[uf.find(r.id)].append(r.id)
    group_of: dict[str, str] = {}
    linked: dict[str, list[str]] = {}
    joined = []
    for members in components.values():
        gold = sorted({frozen_of[m] for m in members if m in frozen_of})
        if len(gold) > 1:
            joined.append(gold)
        rest = sorted(m for m in members if m not in frozen_of)
        for m in members:
            if m in frozen_of:
                group_of[m] = frozen_of[m]
        if not rest:
            continue
        name = gold[0] + LINKED if gold else "tg_" + rest[0].removeprefix(f"{SOURCE}:")
        if gold:
            linked[gold[0]] = rest
        for m in rest:
            group_of[m] = name
    return Grouping(group_of, {k: edges[k] for k in ("same_text", "same_link", "near_text")}, linked, sorted(joined))


def with_groups(records: Sequence[TgRecord], group_of: Mapping[str, str]) -> list[TgRecord]:
    return [replace(r, group_id=group_of[r.id]) for r in records]


def representatives(records: Iterable[TgRecord]) -> dict[str, TgRecord]:
    """The group's latest post: the one a test or dev group contributes to gold."""
    best: dict[str, TgRecord] = {}
    for r in records:
        cur = best.get(r.group_id)
        if cur is None or (r.published_at, r.id) > (cur.published_at, cur.id):
            best[r.group_id] = r
    return best


# --- links to the trudvsem and Habr corpus (S1) ----------------------------------------------


@dataclass(frozen=True)
class S1Corpus:
    split_of: dict[str, str]  # every S1 member record id -> split
    inputs: dict[str, str]  # S1 corpus record id -> extractor input


def load_s1(corpus: Path) -> S1Corpus:
    split_of = {
        str(m): str(row["split"])
        for row in pq.read_table(corpus / "split.parquet", columns=["split", "member_ids"]).to_pylist()
        for m in row["member_ids"]
    }
    inputs = {
        str(r["id"]): str(r["input"])
        for r in pq.read_table(corpus / "corpus.parquet", columns=["id", "input"]).to_pylist()
    }
    return S1Corpus(split_of, inputs)


def neighbours(
    queries: Sequence[tuple[str, set[str]]],
    index: Sequence[tuple[str, set[str]]],
    threshold: float,
    num_perm: int,
    seed: int,
) -> dict[str, list[str]]:
    """Query id -> index ids with an exact Jaccard >= threshold; MinHash LSH only proposes pairs."""
    from datasketch import MinHashLSH  # type: ignore[import-untyped]

    from radar.dedup.minhash import signatures

    sigs = signatures([d for _, d in index] + [d for _, d in queries], num_perm, seed)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as session:
        for i, sig in enumerate(sigs[: len(index)]):
            if sig is not None:
                session.insert(i, sig, check_duplication=False)
    out: dict[str, list[str]] = {}
    for k, (qid, doc) in enumerate(queries):
        sig = sigs[len(index) + k]
        if sig is None:
            continue
        hits = sorted(index[j][0] for j in lsh.query(sig) if jaccard(doc, index[j][1]) >= threshold)
        if hits:
            out[qid] = hits
    return out


def s1_links(records: Sequence[TgRecord], s1: S1Corpus, cfg: SampleConfig) -> dict[str, dict[str, list[str]]]:
    """Telegram record id -> S1 record ids it names by Habr vacancy id (`link`) or nearly repeats (`text`)."""
    out: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for r in records:
        hits = sorted(link for link in r.links if link.startswith("habr:") and link in s1.split_of)
        if hits:
            out[r.id]["link"] = hits
    s1_ids = sorted(s1.inputs)
    index = list(zip(s1_ids, (shingles(s1.inputs[i], cfg.shingle_k) for i in s1_ids), strict=True))
    queries = list(zip((r.id for r in records), input_docs(records, cfg.shingle_k), strict=True))
    for rid, hits in neighbours(queries, index, cfg.leak_jaccard, cfg.num_perm, cfg.seed).items():
        out[rid]["text"] = hits
    return dict(out)


# --- samples --------------------------------------------------------------------------------


def fixture_ids(root: Path = FIXTURES) -> set[str]:
    out: set[str] = set()
    for path in sorted(root.glob("*.html")):
        out.update(record_id(p) for p in _FIXTURE_POST.findall(path.read_text(encoding="utf-8", errors="replace")))
    return out


@dataclass(frozen=True)
class Draw:
    split_of: dict[str, str]
    plan: Plan
    skipped_near_gold: int
    shortfall: dict[str, dict[str, int]]


def draw(reps: Mapping[str, TgRecord], blocked: set[str], cfg: SampleConfig) -> Draw:
    strata: dict[str, list[str]] = defaultdict(list)
    for g, rep in reps.items():
        if g not in blocked:
            strata[stratum(rep.network, rep.role_proxy)].append(g)
    the_plan = plan({h: len(gs) for h, gs in strata.items()}, cfg)
    split_of: dict[str, str] = {}
    drawn: list[set[str]] = []
    skipped = 0
    got: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    for h in sorted(strata):
        want = {s: the_plan.draw[s][h] for s in SPLITS}
        taken = Counter[str]()
        for g in sorted(strata[h], key=lambda g: (permanent_random(cfg.seed, g), g)):
            split = next((s for s in SPLITS if taken[s] < want[s]), None)
            if split is None:
                break
            doc = shingles(reps[g].input, cfg.shingle_k)
            if any(jaccard(doc, d) >= cfg.leak_jaccard for d in drawn):
                skipped += 1
                continue
            split_of[g] = split
            taken[split] += 1
            drawn.append(doc)
        bucket = h.rsplit("|", 1)[1]
        for s in SPLITS:
            got[s][bucket] += taken[s]
    shortfall = {
        s: {b: cfg.targets[s][b] - got[s][b] for b in BUCKETS if got[s][b] < cfg.targets[s][b]} for s in SPLITS
    }
    return Draw(split_of, the_plan, skipped, shortfall)


@dataclass(frozen=True)
class Assigned:
    rows: list[GroupRow]
    reason: dict[str, str]  # withheld group -> why
    leak_before: dict[str, dict[str, list[str]]]
    leak_after: dict[str, dict[str, list[str]]]
    test_dev_pairs: list[tuple[str, str, float]]


def gold_leaks(
    rows: Sequence[GroupRow], records: Sequence[TgRecord], cfg: SampleConfig
) -> dict[str, dict[str, list[str]]]:
    """Per split: gold record id -> training-pool record ids with an input Jaccard >= the leak threshold."""
    split_of = {r.group_id: r.split for r in rows}
    by_id = {r.id: r for r in records}
    pool = [r for r in records if split_of[r.group_id] == TRAIN_POOL]
    index = list(zip((r.id for r in pool), input_docs(pool, cfg.shingle_k), strict=True))
    out: dict[str, dict[str, list[str]]] = {}
    for s in SPLITS:
        gold = [by_id[r.gold_id] for r in rows if r.split == s and r.gold_id]
        queries = list(zip((r.id for r in gold), input_docs(gold, cfg.shingle_k), strict=True))
        out[s] = neighbours(queries, index, cfg.leak_jaccard, cfg.num_perm, cfg.seed) if index else {}
    return out


def cross_split_pairs(
    rows: Sequence[GroupRow], records: Sequence[TgRecord], cfg: SampleConfig
) -> list[tuple[str, str, float]]:
    by_id = {r.id: r for r in records}
    docs = {
        s: {r.gold_id: shingles(by_id[r.gold_id].input, cfg.shingle_k) for r in rows if r.split == s and r.gold_id}
        for s in SPLITS
    }
    pairs = []
    for t, dt in docs["test"].items():
        for d, dd in docs["dev"].items():
            score = jaccard(dt, dd)
            if score >= cfg.leak_jaccard:
                pairs.append((t, d, round(score, 4)))
    return sorted(pairs)


def assign(
    rows: Sequence[GroupRow],
    records: Sequence[TgRecord],
    s1_link: Mapping[str, Mapping[str, list[str]]],
    s1: S1Corpus,
    cfg: SampleConfig,
    reason: Mapping[str, str],
) -> Assigned:
    """Withhold training-pool groups linked to S1 test or dev, then those near a Telegram gold record."""
    s1_gold = {
        r.group_id
        for r in records
        if any(s1.split_of.get(i) in SPLITS for hits in s1_link.get(r.id, {}).values() for i in hits)
    }
    why = dict(reason)
    out = []
    for row in rows:
        if row.split == TRAIN_POOL and row.group_id in s1_gold:
            why[row.group_id] = "s1_gold"
            row = replace(row, split=WITHHELD)
        out.append(row)
    before = gold_leaks(out, records, cfg)
    group_of = {r.id: r.group_id for r in records}
    near = {group_of[i] for per in before.values() for hits in per.values() for i in hits}
    final = []
    for row in out:
        if row.split == TRAIN_POOL and row.group_id in near:
            why[row.group_id] = "near_duplicate"
            row = replace(row, split=WITHHELD)
        final.append(row)
    return Assigned(final, why, before, gold_leaks(final, records, cfg), cross_split_pairs(final, records, cfg))


def group_rows(
    records: Sequence[TgRecord], split_of: Mapping[str, str], the_plan: Plan | None, frozen: Sequence[GroupRow] = ()
) -> list[GroupRow]:
    reps = representatives(records)
    members: dict[str, list[str]] = defaultdict(list)
    for r in records:
        members[r.group_id].append(r.id)
    frozen_ids = {row.group_id for row in frozen}
    rows = list(frozen)
    for g in sorted(members):
        if g in frozen_ids:
            continue
        cell = stratum(reps[g].network, reps[g].role_proxy)
        split = split_of.get(g, TRAIN_POOL)
        in_frame = the_plan is not None and cell in the_plan.frame
        rows.append(
            GroupRow(
                group_id=g,
                split=split,
                stratum=cell,
                gold_id=reps[g].id if split in SPLITS else None,
                pi_test=the_plan.draw["test"][cell] / the_plan.frame[cell] if the_plan and in_frame else None,
                pi_dev=the_plan.draw["dev"][cell] / the_plan.frame[cell] if the_plan and in_frame else None,
                n_corpus=len(members[g]),
                member_ids=tuple(sorted(members[g])),
            )
        )
    return sorted(rows, key=lambda r: r.group_id)


# --- files ----------------------------------------------------------------------------------

RECORD_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("channel", pa.string()),
        pa.field("network", pa.string()),
        pa.field("dt", pa.string()),
        pa.field("published_at", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("title", pa.string()),
        pa.field("hashtags", pa.list_(pa.string())),
        pa.field("role_proxy", pa.string()),
        pa.field("lang", pa.string()),
        pa.field("content_chars", pa.int32()),
        pa.field("links", pa.list_(pa.string())),
        pa.field("input", pa.string()),
        pa.field("group_id", pa.string()),
    ]
)


def write_records(records: Sequence[TgRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(r), "hashtags": list(r.hashtags), "links": list(r.links)} for r in records]
    pq.write_table(pa.Table.from_pylist(rows, schema=RECORD_SCHEMA), path)


def read_records(path: Path) -> list[TgRecord]:
    return [
        TgRecord(**{**row, "hashtags": tuple(row["hashtags"]), "links": tuple(row["links"])})
        for row in pq.read_table(path).to_pylist()
    ]


def write_inputs(path: Path, gold: Sequence[TgRecord], seed: int) -> None:
    """The line format of data/gold/test_inputs.jsonl: id, source and input, nothing the annotators
    should not see; in permanent random order, so neither stratum nor channel runs in blocks."""
    write_lines(path, sorted(gold, key=lambda r: (permanent_random(seed, r.id), r.id)))


def write_lines(path: Path, records: Sequence[TgRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": r.id, "source": SOURCE, "input": r.input}, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_split(path: Path) -> list[GroupRow]:
    return [GroupRow(**{**r, "member_ids": tuple(r["member_ids"])}) for r in pq.read_table(path).to_pylist()]


def write_split(rows: Iterable[GroupRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_split_rows(rows, path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- baseline check -------------------------------------------------------------------------


def baseline_check(inputs: Sequence[tuple[str, str]]) -> tuple[int, list[str]]:
    """Schema-valid regex answers and the ids whose answer fails the schema; any other error raises."""
    invalid = []
    for rid, text in inputs:
        try:
            output_for(text)
        except ValidationError:
            invalid.append(rid)
    return len(inputs) - len(invalid), invalid


# --- stats ----------------------------------------------------------------------------------


def stats(
    posts: Posts,
    records: Sequence[TgRecord],
    grouping: Grouping,
    assigned: Assigned,
    blocked: Mapping[str, int],
    s1_link: Mapping[str, Mapping[str, list[str]]],
    baseline: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    rows = assigned.rows
    vacancy_posts = posts.posts - posts.dropped["not_vacancy"]
    single = vacancy_posts - posts.dropped["multi_vacancy"]
    m["raw.rows_read"] = {"value": posts.rows}
    m["posts.read"] = {"value": posts.posts}
    m["posts.vacancy"] = share(vacancy_posts, posts.posts)
    m["posts.multi_vacancy"] = share(posts.dropped["multi_vacancy"], vacancy_posts)
    m["posts.single_vacancy"] = share(single, vacancy_posts)
    m["posts.link_stub"] = share(posts.dropped["link_stub"], single)
    m["records"] = share(len(records), vacancy_posts)
    for kind, n in posts.not_vacancy_kinds.items():
        m[f"dropped.not_vacancy.{kind}"] = share(n, posts.posts)
    for why in DROPS:
        m[f"dropped.{why}"] = share(posts.dropped[why], posts.posts)

    sizes = Counter(r.group_id for r in records)
    channels_of: dict[str, set[str]] = defaultdict(set)
    for r in records:
        channels_of[r.group_id].add(r.channel)
    m["groups"] = {"value": len(sizes)}
    m["groups.multi_record"] = share(sum(n > 1 for n in sizes.values()), len(sizes))
    m["groups.cross_channel"] = share(sum(len(c) > 1 for c in channels_of.values()), len(sizes))
    m["groups.largest"] = {"value": max(sizes.values(), default=0)}
    for kind, n in grouping.edges.items():
        m[f"groups.edges.{kind}"] = {"value": n}
    m["lang.records.ru"] = share(sum(r.lang == "ru" for r in records), len(records))
    m["lang.records.en"] = share(sum(r.lang == "en" for r in records), len(records))
    m["lang.records.other"] = share(sum(r.lang == "other" for r in records), len(records))
    m["pii_suspect.recruiter_name.records"] = share(
        sum(bool(_RECRUITER_NAME.search(r.input)) for r in records), len(records)
    )

    linked = {r.id for r in records if r.id in s1_link}
    m["s1.records_linked"] = share(len(linked), len(records))
    m["s1.records_linked.by_habr_id"] = share(sum("link" in v for v in s1_link.values()), len(records))
    m["s1.records_linked.by_text"] = share(sum("text" in v for v in s1_link.values()), len(records))
    for why, n in blocked.items():
        m[f"frame.blocked_groups.{why}"] = {"value": n}

    split_of = {r.group_id: r.split for r in rows}
    by_id = {r.id: r for r in records}
    for split in (*SPLITS, TRAIN_POOL, WITHHELD):
        m[f"split.{split}.groups"] = {"value": sum(r.split == split for r in rows)}
        m[f"split.{split}.records"] = {"value": sum(split_of[r.group_id] == split for r in records)}
    for why in ("linked", "s1_gold", "near_duplicate"):
        m[f"split.withheld.groups.{why}"] = {"value": sum(v == why for v in assigned.reason.values())}
    for split in SPLITS:
        gold = [by_id[r.gold_id] for r in rows if r.split == split and r.gold_id]
        per_bucket = Counter(r.role_proxy for r in gold)
        for b in BUCKETS:
            m[f"{split}.role_proxy.{b}"] = share(per_bucket[b], len(gold))
        for lang in ("ru", "en", "other"):
            m[f"{split}.lang.{lang}"] = share(sum(r.lang == lang for r in gold), len(gold))
        m[f"{split}.networks"] = {"value": len({r.network for r in gold})}
        m[f"{split}.channels"] = {"value": len({r.channel for r in gold})}
        m[f"{split}.pii_suspect.recruiter_name"] = share(
            sum(bool(_RECRUITER_NAME.search(r.input)) for r in gold), len(gold)
        )
        m[f"leak.{split}_with_train_neighbour_drawn"] = share(len(assigned.leak_before[split]), len(gold))
        m[f"leak.{split}_with_train_neighbour"] = share(len(assigned.leak_after[split]), len(gold))
        m[f"leak.{split}_linked_to_s1"] = share(sum(r.id in linked for r in gold), len(gold))
    m["leak.test_dev_pairs"] = {"value": len(assigned.test_dev_pairs)}
    m["leak.records_in_two_splits"] = {"value": records_in_two_splits(rows)}
    m["leak.gold_groups_joined"] = {"value": len(grouping.gold_groups_joined)}

    pool = [r for r in records if split_of[r.group_id] == TRAIN_POOL]
    m["decisions10.train_pool.ml_ds_data"] = share(sum(r.role_proxy == ML_DS_DATA for r in pool), len(pool))
    pool_reps = representatives(pool)
    m["decisions10.train_pool_groups.ml_ds_data"] = share(
        sum(r.role_proxy == ML_DS_DATA for r in pool_reps.values()), len(pool_reps)
    )
    for b in BUCKETS:
        m[f"role_proxy.train_pool.{b}"] = share(sum(r.role_proxy == b for r in pool), len(pool))
    m["train_pool.linked_to_s1_train_pool"] = share(sum(r.id in linked for r in pool), len(pool))
    for name, (valid, total) in baseline.items():
        m[f"baseline_check.{name}.schema_valid"] = share(valid, total)
    return m


def hashes(rows: Sequence[GroupRow], records: Sequence[TgRecord]) -> dict[str, str]:
    split_of = {r.group_id: r.split for r in rows}
    gold = {s: [r.gold_id for r in rows if r.split == s and r.gold_id] for s in SPLITS}
    return {
        "test_ids_sha256": id_list_digest(i for i in gold["test"] if i),
        "dev_ids_sha256": id_list_digest(i for i in gold["dev"] if i),
        "train_pool_ids_sha256": id_list_digest(r.id for r in records if split_of[r.group_id] == TRAIN_POOL),
        "split_content_sha256": split_digest(rows),
    }


# --- commands -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Built:
    posts: Posts
    complete: list[str]
    # incomplete partition day -> rows written so far, read from the Parquet footers only
    partial: dict[str, int]


def partition_rows(root: Path, day: str) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in (root / SOURCE / f"dt={day}").glob("part-*.parquet"))


def build(root: Path, cfg: Config) -> Built:
    complete, partial = partitions(root)
    if not complete:
        raise SystemExit(f"no complete Telegram partition under {root / SOURCE}")
    rows, n_rows = read_posts(root, complete)
    posts = build_records(rows, n_rows, networks(), cfg, channel_langs())
    log.info(
        "%d posts in %d complete partitions (%s skipped as incomplete): %d records",
        posts.posts,
        len(complete),
        partial or "none",
        len(posts.records),
    )
    return Built(posts, complete, {day: partition_rows(root, day) for day in partial})


def pool_inputs(rows: Sequence[GroupRow], records: Sequence[TgRecord], n: int, seed: int) -> list[tuple[str, str]]:
    split_of = {r.group_id: r.split for r in rows}
    pool = sorted((r.id, r.input) for r in records if split_of[r.group_id] == TRAIN_POOL)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def finish(
    built: Built,
    records: list[TgRecord],
    grouping: Grouping,
    assigned: Assigned,
    blocked: Mapping[str, int],
    s1_link: Mapping[str, Mapping[str, list[str]]],
    cfg: Config,
    frozen: Mapping[str, Any],
    **extra: Any,
) -> Path:
    from radar.results import write_result

    out = data_dir() / "corpus" / "telegram"
    write_records(records, out / "records.parquet")
    write_split(assigned.rows, out / "split.parquet")
    by_id = {r.id: r for r in records}
    dev_inputs = [(r.gold_id, by_id[r.gold_id].input) for r in assigned.rows if r.split == "dev" and r.gold_id]
    checks: dict[str, tuple[int, int]] = {}
    valid, invalid_dev = baseline_check(dev_inputs)
    checks["dev"] = (valid, len(dev_inputs))
    sample = pool_inputs(assigned.rows, records, cfg.baseline_pool_sample, cfg.sample.seed)
    valid, invalid_pool = baseline_check(sample)
    checks["train_pool_sample"] = (valid, len(sample))

    metrics = stats(built.posts, records, grouping, assigned, blocked, s1_link, checks)
    for k, v in extra.pop("checks", {}).items():
        metrics[f"check.{k}"] = {"value": str(v).lower()}
    per_split_channels = {
        s: dict(Counter(by_id[r.gold_id].channel for r in assigned.rows if r.split == s and r.gold_id).most_common())
        for s in SPLITS
    }
    pool_groups = {r.group_id for r in assigned.rows if r.split == TRAIN_POOL}
    return write_result(
        RESULT,
        metrics,
        source_script="src/radar/extract/telegram_corpus.py",
        config={
            "min_content_chars": cfg.min_content_chars,
            "group_jaccard": cfg.group_jaccard,
            "targets": cfg.sample.targets,
            "power": cfg.sample.power,
            "max_share": cfg.sample.max_share,
            "leak_jaccard": cfg.sample.leak_jaccard,
            "seed": cfg.sample.seed,
            "baseline_pool_sample": cfg.baseline_pool_sample,
        },
        partitions_used=built.complete,
        partitions_skipped_incomplete_rows=built.partial,
        published_range=list(built.posts.published),
        channels=built.posts.per_channel,
        train_pool_channels=dict(Counter(r.channel for r in records if r.group_id in pool_groups).most_common()),
        gold_channels=per_split_channels,
        hashes=hashes(assigned.rows, records),
        frozen=dict(frozen),
        baseline_invalid={"dev": invalid_dev, "train_pool_sample": invalid_pool},
        leak_neighbours_drawn=assigned.leak_before,
        leak_test_dev_pairs=assigned.test_dev_pairs,
        gold_groups_joined=grouping.gold_groups_joined,
        **extra,
    )


def cmd_sample(args: argparse.Namespace) -> int:
    from radar.collect.writer import raw_root

    cfg = load_config()
    gold_dir = data_dir() / "gold"
    out = data_dir() / "corpus" / "telegram"
    if any((gold_dir / name).exists() for name in GOLD_LABELS.values()):
        raise SystemExit("Telegram gold labels exist: test and dev are fixed, redrawing would detach them")
    if (out / FROZEN_DIR).exists() and not args.resample:
        raise SystemExit("Telegram test and dev are drawn already; --resample redraws them (record it in EXPERIMENTS)")

    built = build(raw_root(), cfg)
    grouping = group_records(built.posts.records, cfg)
    records = with_groups(built.posts.records, grouping.group_of)
    s1 = load_s1(data_dir() / "corpus")
    s1_link = s1_links(records, s1, cfg.sample)
    reps = representatives(records)
    group_of = {r.id: r.group_id for r in records}
    fixtures = {group_of[i] for i in fixture_ids() if i in group_of}
    s1_linked = {group_of[i] for i in s1_link}
    the_draw = draw(reps, fixtures | s1_linked, cfg.sample)
    rows = group_rows(records, the_draw.split_of, the_draw.plan)
    assigned = assign(rows, records, s1_link, s1, cfg.sample, {})
    if any(assigned.leak_after.values()) or assigned.test_dev_pairs:
        raise SystemExit("leak check failed after withholding, nothing written")

    by_id = {r.id: r for r in records}
    frozen_dir = out / FROZEN_DIR
    if frozen_dir.exists():
        shutil.rmtree(frozen_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        gold = [by_id[r.gold_id] for r in assigned.rows if r.split == split and r.gold_id]
        write_inputs(frozen_dir / GOLD_INPUTS[split], gold, cfg.sample.seed)
        shutil.copyfile(frozen_dir / GOLD_INPUTS[split], gold_dir / GOLD_INPUTS[split])
    write_records(records, frozen_dir / "records.parquet")
    write_split(assigned.rows, frozen_dir / "split.parquet")
    frozen = {
        **{k: v for k, v in hashes(assigned.rows, records).items() if k != "train_pool_ids_sha256"},
        "split_file_sha256": file_sha256(frozen_dir / "split.parquet"),
        **{f"{s}_inputs_sha256": file_sha256(frozen_dir / GOLD_INPUTS[s]) for s in SPLITS},
        "drawn_from_partitions": built.complete,
    }
    strata = [
        {
            "stratum": h,
            "frame_groups": n,
            "test": the_draw.plan.draw["test"][h],
            "dev": the_draw.plan.draw["dev"][h],
        }
        for h, n in sorted(the_draw.plan.frame.items())
    ]
    path = finish(
        built,
        records,
        grouping,
        assigned,
        {"fixtures": len(fixtures), "s1_linked": len(s1_linked)},
        s1_link,
        cfg,
        frozen,
        stage="sample",
        draw={
            "frame_groups": sum(the_draw.plan.frame.values()),
            "skipped_near_gold": the_draw.skipped_near_gold,
            "shortfall": the_draw.shortfall,
        },
        strata=strata,
    )
    log.info("test %s, dev %s; %s", *(sum(r.split == s for r in assigned.rows) for s in SPLITS), path)
    return 0


@dataclass(frozen=True)
class Frozen:
    records: dict[str, TgRecord]
    rows: list[GroupRow]


def load_frozen(directory: Path, recorded: Mapping[str, Any]) -> Frozen:
    split_path = directory / "split.parquet"
    actual = file_sha256(split_path)
    if actual != recorded["split_file_sha256"]:
        raise SystemExit(f"{split_path} hashes to {actual}, the draw recorded {recorded['split_file_sha256']}")
    for s in SPLITS:
        got = file_sha256(directory / GOLD_INPUTS[s])
        if got != recorded[f"{s}_inputs_sha256"]:
            raise SystemExit(f"{directory / GOLD_INPUTS[s]} hashes to {got}, the draw recorded another file")
    rows = [r for r in read_split(split_path) if r.split in SPLITS]
    gold_ids = {r.gold_id for r in rows}
    return Frozen(
        records={r.id: r for r in read_records(directory / "records.parquet") if r.id in gold_ids},
        rows=rows,
    )


def refresh_records(current: Sequence[TgRecord], frozen: Frozen) -> tuple[list[TgRecord], dict[str, str]]:
    """Current records with every gold record at its drawn version, unless the current version only
    replaces spans with PII placeholders (a cleaner rewrite): then the cleaned one, the labels hold.
    Also gold id -> `same`, `pii`, `other` (an edit, kept out) or `missing` (no longer a record)."""
    by_id = {r.id: r for r in current}
    kinds = {}
    for rid, drawn in frozen.records.items():
        now = by_id.get(rid)
        kinds[rid] = "missing" if now is None else change_kind(drawn.input, now.input)
        by_id[rid] = replace(now, group_id=drawn.group_id) if now is not None and kinds[rid] == "pii" else drawn
    return [by_id[k] for k in sorted(by_id)], kinds


def write_gold_inputs(path: Path, order: Path, records: Mapping[str, TgRecord]) -> None:
    """The ids of the drawn file in its order, with the inputs the records hold now."""
    ids = [json.loads(line)["id"] for line in order.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_lines(path, [records[i] for i in ids])


def cmd_refresh(_: argparse.Namespace) -> int:
    from radar.collect.writer import raw_root
    from radar.config import RESULTS

    cfg = load_config()
    out = data_dir() / "corpus" / "telegram"
    previous = json.loads((RESULTS / RESULT).read_text(encoding="utf-8"))
    recorded = previous["frozen"]
    frozen = load_frozen(out / FROZEN_DIR, recorded)
    built = build(raw_root(), cfg)
    records, kinds = refresh_records(built.posts.records, frozen)
    grouping = group_records(records, cfg, frozen.rows)
    records = with_groups(records, grouping.group_of)
    s1 = load_s1(data_dir() / "corpus")
    s1_link = s1_links(records, s1, cfg.sample)
    rows = group_rows(records, {}, None, frozen.rows)
    reason = {g + LINKED: "linked" for g in grouping.linked}
    rows = [replace(r, split=WITHHELD) if r.group_id in reason else r for r in rows]
    assigned = assign(rows, records, s1_link, s1, cfg.sample, reason)
    now = hashes(assigned.rows, records)
    checks = {
        "test_ids_equal_frozen": now["test_ids_sha256"] == recorded["test_ids_sha256"],
        "dev_ids_equal_frozen": now["dev_ids_sha256"] == recorded["dev_ids_sha256"],
        "test_dev_rows_equal_frozen": frozen_rows_unchanged(frozen.rows, assigned.rows),
    }
    if not all(checks.values()) or any(assigned.leak_after.values()):
        raise SystemExit(f"refresh checks failed, nothing written: {checks}")
    by_id = {r.id: r for r in records}
    for s in SPLITS:
        write_gold_inputs(data_dir() / "gold" / GOLD_INPUTS[s], out / FROZEN_DIR / GOLD_INPUTS[s], by_id)
    gold_split = {r.gold_id: r.split for r in frozen.rows}
    changes = {
        f"{s}.{kind}": sum(1 for rid, k in kinds.items() if k == kind and gold_split[rid] == s)
        for s in SPLITS
        for kind in ("same", "pii", "other", "missing")
    }
    path = finish(
        built,
        records,
        grouping,
        assigned,
        {},
        s1_link,
        cfg,
        recorded,
        stage="refresh",
        draw=previous["draw"],
        strata=previous["strata"],
        checks=checks,
        gold_current_version=changes,
    )
    log.info("refresh: %d records; %s", len(records), path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m radar.extract.telegram_corpus", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sample = sub.add_parser("sample", help="records, groups, frozen test and dev, results/corpus/telegram.json")
    sample.add_argument("--resample", action="store_true", help="redraw unlabeled samples")
    sample.set_defaults(func=cmd_sample)
    sub.add_parser("refresh", help="rebuild records and the training pool around the frozen test and dev").set_defaults(
        func=cmd_refresh
    )
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

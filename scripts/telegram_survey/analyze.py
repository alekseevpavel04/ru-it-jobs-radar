"""Survey of public Telegram job channels: writes results/sources/telegram_channels.json.

Per channel it reports subscribers, the creation date, the first and last post, an estimate of the
number of posts, and on the last 50 posts the share of vacancy posts (rules in classify.py), multi-vacancy
posts, Russian posts, posts per week and near duplicates in other channels. It also checks the channel
selection in configs/telegram_channels.toml against the criteria below and fails if an enabled channel
does not meet them without a recorded exception. Method and tables: docs/sources.md, section Telegram.

Rerun (bash scripts/telegram_survey/run.sh does all three steps):
    python scripts/telegram_survey/shallow.py --offline   # one page per candidate -> probes.json
    python scripts/telegram_survey/deep.py --offline      # depth survey -> survey.json
    python scripts/telegram_survey/analyze.py             # offline, -> results/
With `--offline` the first two steps read only the page cache in data/tmp/telegram_survey/cache
(not in git) and reproduce the published numbers. Without it they fetch missing pages from t.me
(pause 3 s); on a fresh machine that is the whole survey again, about 1 000 requests, and the numbers
drift because the channels keep posting.
"""

import csv
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import tg
from classify import Verdict, classify

from radar.results import write_result

HERE = Path(__file__).parent
SURVEY = tg.CACHE.parent / "survey.json"
PROBES = tg.CACHE.parent / "probes.json"
SAMPLE = 50
SHINGLE = 5
DUP_JACCARD = 0.5
MIN_DUP_N = 10  # fewer shared vacancies than this say nothing about one channel copying another
RU_LETTERS = 0.4  # IT vacancies in Russian carry a lot of Latin tech names
MIN_RU_POSTS = 0.3
ALIVE_DAYS = 30
MIN_VACANCY_SHARE = 0.3
PAUSE_PLUS_ANSWER = 3 + 0.6
NOT_IT = {"dnative_job", "jun_hi_vacancies"}  # SMM and design, judged by their pages

URL = re.compile(r"https?://\S+|t\.me/\S+|@\w+")
WORD = re.compile(r"\w+")


def shingles(text: str) -> set[str]:
    words = WORD.findall(URL.sub(" ", text.lower()))
    return {" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def subscribers_number(shown: str | None) -> int | None:
    if not shown:
        return None
    shown = shown.replace(" ", "")
    return round(float(shown.rstrip("KM")) * {"K": 1_000, "M": 1_000_000}.get(shown[-1], 1))


def letters(text: str) -> tuple[int, int]:
    cyr = sum(("а" <= c.lower() <= "я") or c in "ёЁ" for c in text)
    lat = sum("a" <= c.lower() <= "z" for c in text)
    return cyr, lat


def is_russian(text: str) -> bool:
    cyr, lat = letters(text)
    return cyr / max(cyr + lat, 1) >= RU_LETTERS


def share(n: int, d: int) -> dict:
    return {"value": round(n / d, 3) if d else 0.0, "n": n, "denominator": d}


def aligned_duplicates(a: list, b: list) -> tuple[int, int]:
    """Vacancy posts of `a` with a near duplicate in `b`, both cut to the dates the two samples share."""
    if not a or not b:
        return 0, 0
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    wa = [s for d, s in a if lo <= d <= hi]
    wb = [s for d, s in b if lo <= d <= hi]
    return sum(any(len(x & y) / len(x | y) >= DUP_JACCARD for y in wb) for x in wa), len(wa)


def probe_summary(surveyed: set[str]) -> dict:
    out = {}
    for name, v in json.loads(PROBES.read_text(encoding="utf-8")).items():
        if name in surveyed:
            continue
        if not v["exists"]:
            out[name] = {"status": "no_preview"}
            continue
        age = datetime.fromisoformat(v["fetched_at"]) - datetime.fromisoformat(v["last_date"])
        status = "dead" if age.days > ALIVE_DAYS else "not_it" if name in NOT_IT else "reserve"
        out[name] = {
            "status": status,
            "title": v["title"],
            "subscribers": v["subscribers"],
            "last_post_date": v["last_date"][:10],
        }
    return out


def load_labels() -> dict[str, dict[str, dict[int, tuple[bool, bool | None]]]]:
    labels: dict = defaultdict(lambda: defaultdict(dict))
    with (HERE / "check_labels.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            multi = None if row["multi"] == "" else row["multi"] == "1"
            labels[row["label_set"]][row["channel"]][int(row["post_id"])] = (row["vacancy"] == "1", multi)
    return labels


def agreement(labelled: dict, verdicts: dict) -> dict:
    tp = fp = fn = tn = m_ok = m_n = 0
    for ch, posts in labelled.items():
        for pid, (truth, multi) in posts.items():
            v = verdicts[ch][pid]
            tp += truth and v.vacancy
            fp += (not truth) and v.vacancy
            fn += truth and not v.vacancy
            tn += (not truth) and not v.vacancy
            if truth and multi is not None:
                m_n += 1
                m_ok += multi == v.multi
    return {
        "agreement": share(tp + tn, tp + fp + fn + tn),
        "precision": share(tp, tp + fp),
        "recall": share(tp, tp + fn),
        "multi_agreement": share(m_ok, m_n),
    }


def channel_metrics(d: dict, posts: list[dict], verdicts: list[Verdict], dup_with: dict, surveyed: set) -> dict:
    n = len(posts)
    vac = [v for v in verdicts if v.vacancy]
    kinds = Counter(v.kind for v in verdicts)
    texts = [p["text"] for p in posts if p["text"].strip()]
    counts = [letters(t) for t in texts]
    cyr, lat = sum(c for c, _ in counts), sum(la for _, la in counts)
    fetched = datetime.fromisoformat(d["fetched_at"])
    days = max((fetched - datetime.fromisoformat(posts[0]["date"])).total_seconds() / 86400, 1 / 24)
    per_week = n / days * 7
    # Forward sources outside the survey are often personal channels: they are counted, not named.
    fwd = Counter(
        p["forwarded_from"] if p["forwarded_from"] in surveyed else "other" for p in posts if p["forwarded_from"]
    )
    dup_keys = [dup_with.get(p["id"], set()) for p, v in zip(posts, verdicts, strict=True) if v.vacancy]
    dups = Counter(o for keys in dup_keys for o in keys)
    units = sum(v.n_units for v in vac)
    return {
        "title": d["title"],
        "subscribers": {"value": subscribers_number(d["subscribers"]), "shown": d["subscribers"]},
        "created": {"value": (d["created_date"] or "")[:10]},
        "first_post": {"value": d["first_id"], "date": d["first_date"][:10]},
        "second_post": {"value": d["second_id"], "date": (d["second_date"] or "")[:10]},
        "last_post": {"value": d["last_id"], "date": d["last_date"][:10]},
        "id_density": {"value": d["density"]},
        "posts_estimate": {
            "value": d["posts_estimate"],
            "method": d["depth_method"],
            "by_window_layout": d["posts_estimate_by_layout"],
        },
        "requests": {"value": d["requests"]},
        "sample_posts": {"value": n, "from": posts[0]["date"][:10], "to": posts[-1]["date"][:10]},
        "vacancy_share": share(len(vac), n),
        "multi_share": share(sum(v.multi for v in vac), len(vac)),
        "vacancy_units_in_sample": {"value": units},
        "ad_share": share(kinds["ad"], n),
        "resume_share": share(kinds["resume"], n),
        "no_text_share": share(kinds["no_text"], n),
        "posts_per_week": {"value": round(per_week, 1)},
        "vacancy_posts_per_week": {"value": round(per_week * len(vac) / n, 1)},
        "vacancy_units_per_week": {"value": round(per_week * units / n, 1)},
        "cyrillic_letter_share": {"value": round(cyr / max(cyr + lat, 1), 2)},
        "russian_post_share": share(sum(is_russian(t) for t in texts), len(texts)),
        "forwarded_from_channel_share": share(sum(fwd.values()), n),
        "forwarded_from_top": {"value": ", ".join(f"{k} {v}" for k, v in fwd.most_common(3))},
        "habr_link_share": share(sum(v.habr_link for v in vac), len(vac)),
        "near_duplicate_share": share(sum(bool(k) for k in dup_keys), len(vac)),
        "near_duplicate_top": {"value": ", ".join(f"{k} {v}" for k, v in dups.most_common(3))},
        "days_since_last_post": {"value": (fetched - datetime.fromisoformat(d["last_date"])).days},
    }


def main() -> None:
    survey = json.loads(SURVEY.read_text(encoding="utf-8"))
    toml = tomllib.loads((tg.ROOT / "configs" / "telegram_channels.toml").read_text(encoding="utf-8"))
    config = {c["name"]: c for c in toml["channel"]}
    if set(config) != set(survey):
        sys.exit(f"survey and config differ: {sorted(set(config) ^ set(survey))}; rerun deep.py")
    samples = {ch: d["posts"][-SAMPLE:] for ch, d in survey.items()}
    verdicts = {ch: [classify(p["text"], p["links"]) for p in posts] for ch, posts in samples.items()}

    # Near duplicates across all samples, found through shared word shingles.
    sh: dict[tuple[str, int], set[str]] = {}
    index: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for ch, posts in samples.items():
        for p, v in zip(posts, verdicts[ch], strict=True):
            s = shingles(p["text"]) if v.vacancy else set()
            if len(s) >= 10:
                sh[(ch, p["id"])] = s
                for g in s:
                    index[g].add((ch, p["id"]))
    dup_with: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for key, s in sh.items():
        common = Counter(k for g in s for k in index[g] if k[0] != key[0])
        for other, c in common.items():
            if c / len(s | sh[other]) >= DUP_JACCARD:
                dup_with[key[0]][key[1]].add(other[0])

    surveyed = set(samples)
    channels = {
        ch: channel_metrics(survey[ch], posts, verdicts[ch], dup_with[ch], surveyed) for ch, posts in samples.items()
    }

    labels = load_labels()
    by_id = {ch: {p["id"]: v for p, v in zip(samples[ch], verdicts[ch], strict=True)} for ch in samples}
    for label_set in labels.values():
        for ch, posts in label_set.items():
            channels[ch]["vacancy_share_manual"] = share(sum(t for t, _ in posts.values()), len(posts))
    enabled = [ch for ch, c in config.items() if c["enabled"]]
    runs = {
        "v2_tuning": labels["tuning"],
        "v2_check": labels["check"],
        "v2_check_enabled": {ch: v for ch, v in labels["check"].items() if config[ch]["enabled"]},
    }
    rule_metrics = {
        f"rules_{run}_{name}": value
        for run, labelled in runs.items()
        for name, value in agreement(labelled, by_id).items()
    }

    for ch in enabled:
        m = channels[ch]
        manual = m.get("vacancy_share_manual", m["vacancy_share"])
        fails = [
            reason
            for reason, bad in (
                ("stale", m["days_since_last_post"]["value"] > ALIVE_DAYS),
                ("few vacancies", m["vacancy_share"]["value"] < MIN_VACANCY_SHARE),
                ("few vacancies by manual check", manual["value"] < MIN_VACANCY_SHARE),
                ("not russian", m["russian_post_share"]["value"] < MIN_RU_POSTS),
            )
            if bad
        ]
        if fails and not config[ch].get("exception"):
            sys.exit(f"{ch} is enabled but fails {fails}; disable it or record an exception")

    vac_shingles = {ch: [(p["date"], sh[(ch, p["id"])]) for p in samples[ch] if (ch, p["id"]) in sh] for ch in enabled}
    for ch in enabled:
        best = (0, 0, "")
        for other in enabled:
            hit, n = aligned_duplicates(vac_shingles[ch], vac_shingles[other]) if other != ch else (0, 0)
            if n and hit / n > (best[0] / best[1] if best[1] else 0):
                best = (hit, n, other)
        channels[ch]["aligned_duplicate_max"] = share(best[0], best[1]) | {"with": best[2]}
        if best[1] >= MIN_DUP_N and best[0] / best[1] >= 0.5 and not config[ch].get("exception"):
            sys.exit(f"{ch} repeats {best[2]} on shared dates; disable it or record an exception")

    for ch, c in config.items():
        channels[ch]["config"] = {
            "value": "enabled" if c["enabled"] else "disabled",
            "topic": c["topic"],
            "network": c.get("network", ""),
            "reason": c["reason"],
        }

    new = [ch for ch in enabled if not config[ch].get("known_before_1_5")]
    pages_new = sum(channels[ch]["posts_estimate"]["value"] for ch in new) / 20

    def total(key: str, names: list[str]) -> int:
        return round(sum(channels[ch][key]["value"] for ch in names))

    def pooled(key: str, names: list[str]) -> dict:
        return share(sum(channels[ch][key]["n"] for ch in names), sum(channels[ch][key]["denominator"] for ch in names))

    probed = probe_summary(surveyed)
    metrics = {
        **rule_metrics,
        "backfill_pages_new": {"value": round(pages_new)},
        "backfill_hours_new": {"value": round(pages_new * PAUSE_PLUS_ANSWER / 3600, 1)},
        "channels_surveyed": {"value": len(channels)},
        "channels_enabled": {"value": len(enabled)},
        "channels_enabled_new": {"value": len(new)},
        "posts_estimate_enabled": {"value": total("posts_estimate", enabled)},
        "posts_estimate_enabled_new": {"value": total("posts_estimate", new)},
        "vacancy_posts_per_week_enabled": {"value": total("vacancy_posts_per_week", enabled)},
        "vacancy_posts_per_week_enabled_new": {"value": total("vacancy_posts_per_week", new)},
        "posts_per_week_enabled": {"value": total("posts_per_week", enabled)},
        "vacancy_share_enabled_pooled": pooled("vacancy_share", enabled),
        "near_duplicate_share_enabled_pooled": pooled("near_duplicate_share", enabled),
        "requests_survey": {"value": total("requests", list(channels))},
        "requests_survey_max_per_channel": {"value": max(channels[ch]["requests"]["value"] for ch in channels)},
        # Each t.me page was fetched once and kept, so the cache size counts the requests of the survey.
        "requests_tme_total": {"value": sum(1 for _ in tg.CACHE.iterdir())},
        "candidates_probed_one_page": {"value": len(json.loads(PROBES.read_text(encoding="utf-8")))},
        **{
            f"probed_{status}": {"value": sum(v["status"] == status for v in probed.values())}
            for status in ("no_preview", "dead", "not_it", "reserve")
        },
    }
    path = write_result(
        "sources/telegram_channels.json",
        metrics,
        source_script="scripts/telegram_survey/analyze.py",
        method={
            "sample": f"last {SAMPLE} posts per channel, service messages excluded",
            "vacancy_rule": "scripts/telegram_survey/classify.py, manual labels in check_labels.csv",
            "depth": "first and last post, 8 windows of ?before= as strata (deep.py)",
            "near_duplicate": f"word {SHINGLE}-shingles, Jaccard >= {DUP_JACCARD}, vacancy posts across samples",
            "selection": f"alive <= {ALIVE_DAYS} days, vacancy share >= {MIN_VACANCY_SHARE}, russian posts >= "
            f"{MIN_RU_POSTS} (cyrillic letters >= {RU_LETTERS}), IT topic, no channel repeated on shared dates",
            "surveyed_on": "2026-10-08",
        },
        channels=channels,
        probed=probed,
    )
    print(path)


if __name__ == "__main__":
    main()

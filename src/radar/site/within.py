"""Change of a share between periods measured within channels, with a channel bootstrap for its noise.

A channel that starts or stops posting, or grows several times, moves a pooled share by itself: in 2026Q3
javascript_jobs_feed gave 253 of 569 frontend posts and almost none a year earlier. Two estimates here:

- within_change: two periods. Each channel with posts in both weighs n1 n2 / (n1 + n2) in both (the
  Mantel-Haenszel weight of a risk difference); the shares are the weighted shares of those channels.
- fixed_weight_series: a yearly series. Each channel of the panel (posts in every year) keeps one weight in
  every year, its share of the panel's posts over all the years, so a line moves only when shares move
  inside channels.

A change has to clear two kinds of noise, and its p is the larger of the two:
- posts: the binomial variance of the weighted difference when nothing changed, p (1 - p) sum w_c^2
  (1/n0_c + 1/n1_c) / (sum w_c)^2 with p the share over both periods in those channels;
- channels: a bootstrap over channels, because channels change their own mix and the market is the spread
  of channels, not one pool of posts. The bootstrap p is twice the smaller tail around zero.
`python -m radar.site.within` checks the rule on a placebo (results/market/within_placebo.json): within one
year each channel's weeks go at random to two halves, so nothing changed, and the share of skills flagged
at 5 % is counted for each test alone and for the rule.

    python -m radar.site.within [--year 2022] [--splits 40] [--skills 40]
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

import numpy as np

from radar.config import ROOT
from radar.results import write_result

REPS = 2000
SEED = 20261015

Counts = Mapping[str, Mapping[str, int]]  # channel -> period -> posts


def panel(size: Counts, periods: Sequence[str]) -> list[str]:
    """The channels with posts in every period, sorted."""
    return sorted(c for c, n in size.items() if all(n.get(p) for p in periods))


def _resample(channels: int, reps: int, seed: int) -> np.ndarray:
    """How many times each channel is drawn in each bootstrap replicate: (reps, channels)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(channels, np.full(channels, 1 / channels), size=reps)


def _p_two_sided(boot: np.ndarray) -> np.ndarray:
    low, high = (boot <= 0).mean(axis=-1), (boot >= 0).mean(axis=-1)
    return np.asarray(np.clip(2 * np.minimum(low, high), 1 / boot.shape[-1], 1.0))


def _p_posts(diff: np.ndarray, pooled: np.ndarray, w: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Two-sided p of a weighted difference under binomial noise of the posts, nothing changed."""
    var = pooled * (1 - pooled) * (w**2 * (1 / n0 + 1 / n1)).sum() / w.sum() ** 2
    se = np.sqrt(var)
    z = np.divide(np.abs(diff), se, out=np.zeros_like(diff), where=se > 0)
    p = 2 * (1 - np.vectorize(NormalDist().cdf)(z))
    return np.where(se > 0, p, 1.0)


def within_change(
    size: Counts,
    named: Mapping[str, Counts],
    before: str,
    now: str,
    reps: int = REPS,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    """For each key of named (key -> channel -> period -> posts naming it): the weighted shares before and
    now over the channels with posts in both periods, and the bootstrap p of the change."""
    channels = panel(size, (before, now))
    if not channels or not named:
        return {}
    n0 = np.array([size[c][before] for c in channels], dtype=float)
    n1 = np.array([size[c][now] for c in channels], dtype=float)
    w = n0 * n1 / (n0 + n1)
    keys = list(named)
    p0 = np.array([[named[k].get(c, {}).get(before, 0) for c in channels] for k in keys]) / n0
    p1 = np.array([[named[k].get(c, {}).get(now, 0) for c in channels] for k in keys]) / n1
    was, is_ = p0 @ w / w.sum(), p1 @ w / w.sum()
    draws = _resample(len(channels), reps, seed) * w  # (reps, channels)
    boot = ((p1 - p0) @ draws.T) / draws.sum(axis=1)  # (keys, reps)
    pooled = (p0 * n0 + p1 * n1).sum(axis=1) / (n0 + n1).sum()
    p = np.maximum(_p_two_sided(boot), _p_posts(is_ - was, pooled, w, n0, n1))
    return {k: (float(was[i]), float(is_[i]), float(p[i])) for i, k in enumerate(keys)}


def _tests(size: Counts, named: Mapping[str, Counts], reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Placebo helper: the post-noise p and the bootstrap p of within_change, apart."""
    channels = panel(size, ("A", "B"))
    n0 = np.array([size[c]["A"] for c in channels], dtype=float)
    n1 = np.array([size[c]["B"] for c in channels], dtype=float)
    w = n0 * n1 / (n0 + n1)
    p0 = np.array([[named[k].get(c, {}).get("A", 0) for c in channels] for k in named]) / n0
    p1 = np.array([[named[k].get(c, {}).get("B", 0) for c in channels] for k in named]) / n1
    draws = _resample(len(channels), reps, seed) * w
    boot = ((p1 - p0) @ draws.T) / draws.sum(axis=1)
    pooled = (p0 * n0 + p1 * n1).sum(axis=1) / (n0 + n1).sum()
    return _p_posts(p1 @ w / w.sum() - p0 @ w / w.sum(), pooled, w, n0, n1), _p_two_sided(boot)


def fixed_weight_series(
    size: Counts,
    named: Mapping[str, Counts],
    years: Sequence[str],
    reps: int = REPS,
    seed: int = SEED,
) -> tuple[list[str], dict[str, tuple[list[float], float]]]:
    """The panel channels (posts in every year) and, for each key of named, its share per year at fixed
    channel weights and the bootstrap p of the change from the first year to the last."""
    channels = panel(size, years)
    if not channels or not named:
        return channels, {}
    n = np.array([[size[c][y] for y in years] for c in channels], dtype=float)  # (channels, years)
    w = n.sum(axis=1) / n.sum()
    keys = list(named)
    shares = np.array([[[named[k].get(c, {}).get(y, 0) for y in years] for c in channels] for k in keys]) / n
    series = np.einsum("kcy,c->ky", shares, w)
    draws = _resample(len(channels), reps, seed) * w  # (reps, channels)
    delta = shares[:, :, -1] - shares[:, :, 0]  # (keys, channels)
    boot = (delta @ draws.T) / draws.sum(axis=1)
    first, last = n[:, 0], n[:, -1]
    pooled = (shares[:, :, 0] * first + shares[:, :, -1] * last).sum(axis=1) / (first + last).sum()
    p = np.maximum(_p_two_sided(boot), _p_posts(series[:, -1] - series[:, 0], pooled, w, first, last))
    return channels, {k: ([float(x) for x in series[i]], float(p[i])) for i, k in enumerate(keys)}


def benjamini_hochberg(p: Sequence[float]) -> list[float]:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def placebo(aggregates: Path, year: int, splits: int, top: int, reps: int = 1000) -> dict[str, dict[str, float]]:
    posts = list(csv.DictReader((aggregates / "posts" / f"{year}.csv").open(encoding="utf-8", newline="")))
    skills = list(csv.DictReader((aggregates / "skills" / f"{year}.csv").open(encoding="utf-8", newline="")))
    mentions: Counter[str] = Counter()
    for r in skills:
        mentions[r["skill"]] += int(r["posts"])
    listed = sorted(mentions, key=lambda k: (-mentions[k], k))[:top]
    channels = sorted({r["channel"] for r in posts})
    weeks = sorted({r["week"] for r in posts})
    flagged: Counter[str] = Counter()
    for split in range(splits):
        rng = random.Random(SEED + split)
        half = {(c, wk): "A" if rng.random() < 0.5 else "B" for c in channels for wk in weeks}
        size: dict[str, Counter[str]] = defaultdict(Counter)
        named: dict[str, dict[str, Counter[str]]] = {k: defaultdict(Counter) for k in listed}
        for r in posts:
            size[r["channel"]][half[(r["channel"], r["week"])]] += int(r["posts"])
        for r in skills:
            if r["skill"] in named:
                named[r["skill"]][r["channel"]][half[(r["channel"], r["week"])]] += int(r["posts"])
        p_posts, p_boot = _tests(size, named, reps, SEED + split)
        flagged["posts"] += int((p_posts < 0.05).sum())
        flagged["channels"] += int((p_boot < 0.05).sum())
        flagged["rule"] += int((np.maximum(p_posts, p_boot) < 0.05).sum())
    total = splits * len(listed)
    return {
        k: {"value": round(flagged[k] / total, 4), "n": flagged[k], "denominator": total}
        for k in ("posts", "channels", "rule")
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Placebo of the within-channel change test")
    ap.add_argument("--aggregates", type=Path, default=ROOT / "aggregates" / "telegram")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--splits", type=int, default=40)
    ap.add_argument("--skills", type=int, default=40)
    args = ap.parse_args(argv)
    flagged = placebo(args.aggregates, args.year, args.splits, args.skills)
    metrics = {f"flagged_at_5pct.{k}": v for k, v in flagged.items()}
    path = write_result(
        "market/within_placebo.json",
        metrics,
        source_script="src/radar/site/within.py",
        year=args.year,
        splits=args.splits,
        skills=args.skills,
        definitions={
            "flagged_at_5pct": "skills with p < 0.05 when each channel's weeks of one year are split at random "
            "into two halves; posts: binomial test of the weighted difference, channels: bootstrap over channels, "
            "rule: the larger p of the two"
        },
    )
    print({k: v["value"] for k, v in flagged.items()}, "->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

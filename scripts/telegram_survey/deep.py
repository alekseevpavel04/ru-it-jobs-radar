"""Depth survey of every channel in configs/telegram_channels.toml, about 13 pages per channel.

Usage: python scripts/telegram_survey/deep.py [--offline]
Writes data/tmp/telegram_survey/survey.json with the last 50 posts of each channel (texts included,
so the file stays out of git).
"""

import argparse
import json
import tomllib

import tg

SAMPLE = 50
WINDOWS = 8
OUT = tg.CACHE.parent / "survey.json"
# A first post followed by more than this many deleted ids means the channel was wiped and reused.
GAP = 100


def layout(base: int, last: int) -> list[int]:
    span = last - base + 1
    return [base + int(span * (k + 1) / (WINDOWS + 1)) for k in range(WINDOWS)]


def strata_estimate(client: tg.Client, ch: str, xs: list[int], start: int, sample: list[tg.Post]) -> tuple[int, list]:
    """Each window `?before=x` stands for the ids between the previous window and x; the newest stratum
    is counted from the sample of latest posts."""
    bounds = [start, *xs]
    total = 0.0
    windows = []
    for k, x in enumerate(xs):
        page = [p for p in client.page(ch, before=x).posts if start <= p.id < x]
        n = sum(not p.service for p in page)
        # A full page means the window did not reach the start; service messages fill pages too.
        length = x - min(p.id for p in page) if len(page) >= 20 else x - start
        width = x - bounds[k]
        total += width * n / max(length, 1)
        windows.append({"before": x, "n": n, "len": length, "stratum": width})
    last = sample[-1].id
    if sample[0].id <= xs[-1]:
        total += sum(p.id >= xs[-1] for p in sample)
    else:
        total += (last + 1 - xs[-1]) * len(sample) / (last - sample[0].id + 1)
    return round(total), windows


def survey(client: tg.Client, ch: str) -> dict:
    touched0 = len(client.touched)
    head = client.page(ch)
    sample = [p for p in head.posts if not p.service]
    before = head.prev_before
    while len(sample) < SAMPLE and before:
        page = client.page(ch, before=before)
        if not page.posts:
            break
        sample = [p for p in page.posts if not p.service] + sample
        before = page.prev_before
    sample.sort(key=lambda p: p.id)
    last = sample[-1]

    early = client.page(ch, after=1).posts
    if early and early[0].id > 1:
        early += client.page(ch, before=early[0].id).posts
    early.sort(key=lambda p: p.id)
    created = early[0] if early and early[0].service else None
    real = [p for p in early if not p.service]
    first = real[0] if real else sample[0]
    second = real[1] if len(real) > 1 else None
    wiped = second is not None and second.id - first.id > GAP
    start = second if wiped and second else first
    lead = int(wiped)
    span = last.id - start.id + 1

    by_layout: list[int] = []
    windows: list = []
    if span > (WINDOWS + 3) * 20:
        # Windows are laid out from the earliest message, the service one included (the original run
        # of 2026-10-08). For some channels a second run laid them out from the first real post; when
        # those pages are cached they are used too, and the two layouts show the spread of the method.
        base = early[0] if early else first
        if len(early) > 1 and early[1].id - early[0].id > GAP:
            base = early[1]
        layouts = [layout(base.id, last.id)]
        alt = layout(start.id, last.id)
        if alt != layouts[0] and all(client.is_cached(tg.page_url(ch, before=x)) for x in alt):
            layouts.append(alt)
        for xs in layouts:
            by_layout.append(strata_estimate(client, ch, sorted({x for x in xs if x > start.id}), start.id, sample)[0])
        merged = sorted({x for xs in layouts for x in xs if x > start.id})
        estimate, windows = strata_estimate(client, ch, merged, start.id, sample)
        estimate += lead
        by_layout = [e + lead for e in by_layout]
        method = "strata"
    else:
        ids = {p.id for p in sample} | {p.id for p in real}
        cursor = sample[0].id
        while cursor > first.id + 1:
            page = client.page(ch, before=cursor)
            if not page.posts:
                break
            ids |= {p.id for p in page.posts if not p.service}
            cursor = min(p.id for p in page.posts)
        estimate = len(ids)
        method = "full"
    return {
        "title": head.title,
        "subscribers": head.subscribers,
        "fetched_at": head.fetched_at,
        "created_date": created.date if created else None,
        "first_id": first.id,
        "first_date": first.date,
        "second_id": second.id if second else None,
        "second_date": second.date if second else None,
        "last_id": last.id,
        "last_date": last.date,
        "density": round((estimate - lead) / span, 3) if method == "strata" else round(estimate / span, 3),
        "posts_estimate": estimate,
        "posts_estimate_by_layout": by_layout,
        "depth_method": method,
        "windows": windows,
        "requests": len(set(client.touched[touched0:])),
        "posts": [p.to_dict() for p in sample],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached pages only, never call t.me")
    args = parser.parse_args()
    client = tg.Client(offline=args.offline)
    config = tomllib.loads((tg.ROOT / "configs" / "telegram_channels.toml").read_text(encoding="utf-8"))
    out = {}
    for channel in config["channel"]:
        out[channel["name"]] = survey(client, channel["name"])
        print(channel["name"], out[channel["name"]]["posts_estimate"], flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} channels, {client.requests} requests to t.me")


if __name__ == "__main__":
    main()

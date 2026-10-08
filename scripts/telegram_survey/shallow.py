"""One page per candidate: is there a public channel preview, how big and how fresh is the channel.

Usage: python scripts/telegram_survey/shallow.py [--offline]
Reads candidates.txt, writes data/tmp/telegram_survey/probes.json.
"""

import argparse
import json
from pathlib import Path

import tg

HERE = Path(__file__).parent
OUT = tg.CACHE.parent / "probes.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached pages only, never call t.me")
    args = parser.parse_args()
    client = tg.Client(offline=args.offline)
    names = [ln.strip() for ln in (HERE / "candidates.txt").read_text(encoding="utf-8").splitlines()]
    probes = {}
    for name in dict.fromkeys(n for n in names if n and not n.startswith("#")):
        page = client.page(name)
        posts = [p for p in page.posts if p.date]
        probes[name] = {"exists": page.exists, "status": page.status, "fetched_at": page.fetched_at}
        if page.exists and posts:
            probes[name] |= {
                "title": page.title,
                "subscribers": page.subscribers,
                "last_id": max(p.id for p in posts),
                "last_date": max(p.date for p in posts),
            }
        elif page.exists:
            probes[name]["exists"] = False
    OUT.write_text(json.dumps(probes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sum(p['exists'] for p in probes.values())} of {len(probes)} exist, {client.requests} requests to t.me")


if __name__ == "__main__":
    main()

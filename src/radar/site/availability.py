"""Are the public artifacts reachable from Russia and from abroad? results/public/availability.json.

    python -m radar.site.availability

Each URL is requested once from this machine (in Russia) and from check-host.net nodes in Russia,
Germany, the Netherlands and the USA. A node counts as reaching the URL when it got HTTP 200.
"""

import argparse
import time
from typing import Any

import httpx

from radar.results import write_result

URLS = {
    "site": "https://alekseevpavel04.github.io/ru-it-jobs-radar/",
    "space": "https://alekseevpavel04-ru-it-jobs-radar.static.hf.space/index.html",
    "space_page": "https://huggingface.co/spaces/alekseevpavel04/ru-it-jobs-radar",
    "model": "https://huggingface.co/alekseevpavel04/ru-it-vacancy-extractor-qwen3-1.7b",
    "dataset": "https://huggingface.co/datasets/alekseevpavel04/ru-it-vacancies",
}
NODES = ("ru1", "ru2", "ru3", "de1", "nl1", "us1")
API = "https://check-host.net"


def local_check(client: httpx.Client, url: str) -> bool:
    try:
        return client.get(url, follow_redirects=True).status_code == 200
    except httpx.HTTPError:
        return False


def node_checks(client: httpx.Client, url: str, wait_s: float) -> dict[str, bool]:
    """HTTP status per node; a node that did not answer within `wait_s` counts as not reaching the URL."""
    params = httpx.QueryParams([("host", url)] + [("node", f"{n}.node.check-host.net") for n in NODES])
    request_id = client.get(f"{API}/check-http", params=params, headers={"Accept": "application/json"}).json()[
        "request_id"
    ]
    deadline = time.monotonic() + wait_s
    result: dict[str, Any] = {}
    while time.monotonic() < deadline:
        time.sleep(5)
        result = client.get(f"{API}/check-result/{request_id}", headers={"Accept": "application/json"}).json()
        if all(result.get(f"{n}.node.check-host.net") is not None for n in NODES):
            break
    out = {}
    for n in NODES:
        answer = result.get(f"{n}.node.check-host.net")
        # an answer is [[ok, seconds, message, code, ip]]
        out[n] = bool(answer and answer[0] and answer[0][0] == 1 and str(answer[0][3]) == "200")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.site.availability", description=__doc__.splitlines()[0])
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the nodes of one URL")
    args = ap.parse_args(argv)
    metrics: dict[str, dict[str, Any]] = {}
    per_url: dict[str, dict[str, bool]] = {}
    with httpx.Client(timeout=30.0) as client:
        for name, url in URLS.items():
            checks = {"local_ru": local_check(client, url), **node_checks(client, url, args.wait)}
            per_url[name] = checks
            ru = [v for k, v in checks.items() if k.startswith(("ru", "local"))]
            metrics[f"{name}.reached_from_ru"] = {"value": sum(ru) / len(ru), "n": sum(ru), "denominator": len(ru)}
            metrics[f"{name}.reached_all"] = {
                "value": sum(checks.values()) / len(checks),
                "n": sum(checks.values()),
                "denominator": len(checks),
            }
    path = write_result(
        "public/availability.json", metrics, "src/radar/site/availability.py", urls=URLS, checks=per_url
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The same page from one mart built with Evidence.dev and with Observable Framework, measured.

    python -m radar.site.bi_trial [--workdir site/_trial]

The page: IT vacancies seen open per week by role bucket, from marts.mart_weekly_dynamics (full
portal snapshots of trudvsem), as a stacked bar chart and one formatted number. Measured per tool:
install (time, packages, node_modules size, npm audit), build (cold and warm), output size, what a
browser downloads and from which hosts when it opens the page, and how the number is formatted.

Needs node and npm on PATH, the npm registry, a running ClickHouse and Playwright in site/node_modules
(shoot.mjs). Writes results/site/bi_choice.json. The work dir is throwaway: nothing in it is committed.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from radar.config import ROOT, load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

SITE = ROOT / "site"
WEEKLY_SQL = """
SELECT week, role_bucket, vacancies_active, days_collected, complete_days
FROM marts.mart_weekly_dynamics
WHERE source = 'trudvsem_snapshot'
ORDER BY week, role_bucket
FORMAT CSVWithNames
"""

EVIDENCE_PACKAGES = {
    "@evidence-dev/evidence": "40.1.8",
    "@evidence-dev/core-components": "5.4.2",
    "@evidence-dev/csv": "1.0.16",
    # svelte2tsx, a peer of evidence, accepts typescript 4 and 5 only; npm resolves 7 without the pin.
    "typescript": "5.4.2",
}
OBSERVABLE_PACKAGES = {"@observablehq/framework": "1.13.4"}

EVIDENCE_CONFIG = """plugins:
  components:
    "@evidence-dev/core-components": {}
  datasources:
    "@evidence-dev/csv": {}
"""
EVIDENCE_PAGE = """---
title: Активные IT-вакансии по неделям
---

```sql weekly
select week, role_bucket, vacancies_active from radar.weekly
```

```sql last
select sum(vacancies_active) as n from radar.weekly where week = (select max(week) from radar.weekly)
```

Последняя неделя: <Value data={last} column=n fmt='#,##0'/> вакансий

<BarChart data={weekly} x=week y=vacancies_active series=role_bucket yAxisTitle="вакансий"/>
"""

OBSERVABLE_CONFIG = """export default {{
  title: "Проба",
  root: "src",
  lang: "ru"{extra}
}};
"""
# Without these two options the default theme links Google Fonts and the footer credits the tool.
OBSERVABLE_CONFIG_NO_FONTS = ',\n  globalStylesheets: [],\n  footer: ""'
OBSERVABLE_PAGE = """# Активные IT-вакансии по неделям

```js
const weekly = FileAttachment("data/weekly.csv").csv({typed: true});
const fmt = new Intl.NumberFormat("ru-RU");
```

```js
const totals = d3.rollups(weekly, (v) => d3.sum(v, (d) => d.vacancies_active), (d) => d.week);
display(html`<p>Последняя неделя: ${fmt.format(totals.at(-1)[1])} вакансий</p>`);
```

```js
Plot.plot({
  width,
  y: {grid: true, label: "вакансий", tickFormat: (d) => fmt.format(d)},
  color: {legend: true},
  marks: [Plot.barY(weekly, {x: "week", y: "vacancies_active", fill: "role_bucket", tip: true})]
})
```
"""
NUMBER_LINE = re.compile(r"Последняя неделя: (?P<number>[\d\s,.   ]+?) вакансий")


def run(cmd: list[str], cwd: Path, log: Path) -> float:
    """Run a command, append its output to the log, return wall seconds; a failure stops the trial."""
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"{cmd[0]} is not on PATH")
    started = time.perf_counter()
    done = subprocess.run([exe, *cmd[1:]], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    seconds = time.perf_counter() - started
    with log.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}  ({seconds:.1f} s, rc={done.returncode})\n{done.stdout}\n{done.stderr}\n")
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}, see {log}")
    return seconds


def output(cmd: list[str], cwd: Path) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"{cmd[0]} is not on PATH")
    return subprocess.run([exe, *cmd[1:]], cwd=cwd, capture_output=True, text=True, encoding="utf-8").stdout


def tree_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def install(project: Path, packages: dict[str, str], log: Path) -> dict[str, Any]:
    seconds = run(["npm", "install", "--save-exact", *(f"{k}@{v}" for k, v in packages.items())], project, log)
    installed = json.loads(output(["npm", "ls", "--all", "--json"], project) or "{}")
    audit = json.loads(output(["npm", "audit", "--json"], project) or "{}")
    vulns = audit.get("metadata", {}).get("vulnerabilities", {})
    size, _ = tree_size(project / "node_modules")
    main = next(iter(packages))
    released = json.loads(output(["npm", "view", main, "time", "--json"], project))
    return {
        "install_seconds": round(seconds, 1),
        "packages_installed": len(_flatten(installed.get("dependencies", {}))),
        "node_modules_bytes": size,
        "audit_high_or_critical": vulns.get("high", 0) + vulns.get("critical", 0),
        "audit_total": vulns.get("total", 0),
        "release_used": packages[main],
        "release_used_date": released[packages[main]][:10],
        "latest_release_date": max(v for k, v in released.items() if k not in ("created", "modified"))[:10],
    }


def _flatten(deps: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    for name, info in deps.items():
        seen.add(f"{name}@{info.get('version')}")
        seen |= _flatten(info.get("dependencies", {}))
    return seen


def probe(dist: Path, out: Path, log: Path) -> dict[str, Any]:
    """Open the built page in Chromium (site/scripts/shoot.mjs); external hosts make it exit 1, kept here."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not on PATH")
    done = subprocess.run(
        [node, "scripts/shoot.mjs", "--dir", str(dist), "--out", str(out), "--pages", "index", "--text"],
        cwd=SITE,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(f"$ shoot {dist}\n{done.stdout}\n{done.stderr}\n")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    desktop = next(r for r in report if r["width"] == 1280)
    found = NUMBER_LINE.search(desktop["text"])
    if found is None:
        raise RuntimeError(f"no number line on the page built in {dist}")
    number = found.group("number").strip()
    return {
        "page_status": desktop["status"],
        "page_bytes_transferred": desktop["bytes"],
        "page_load_ms": desktop["load_ms"],
        "external_hosts_at_runtime": ",".join(desktop["external_hosts"]) or "none",
        "page_errors": len(desktop["errors"]),
        "number_as_rendered": number,
        "number_grouped_ru": int("," not in number and any(c in number for c in "   ")),
    }


def trial_evidence(work: Path, csv: str, log: Path) -> dict[str, Any]:
    project = work / "evidence"
    write(project / "package.json", '{"name": "trial-evidence", "private": true, "type": "module"}\n')
    write(project / "evidence.config.yaml", EVIDENCE_CONFIG)
    write(project / "sources" / "radar" / "connection.yaml", "name: radar\ntype: csv\noptions: {}\n")
    write(project / "sources" / "radar" / "weekly.csv", csv)
    write(project / "pages" / "index.md", EVIDENCE_PAGE)
    metrics = install(project, EVIDENCE_PACKAGES, log)
    sources = run(["npx", "evidence", "sources"], project, log)
    cold = run(["npx", "evidence", "build"], project, log)
    warm = run(["npx", "evidence", "build"], project, log)
    size, files = tree_size(project / "build")
    biggest = max((p for p in (project / "build").rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    metrics |= {
        "build_seconds_cold": round(sources + cold, 1),
        "build_seconds_warm": round(sources + warm, 1),
        "output_bytes": size,
        "output_files": files,
        "largest_output_file": biggest.name,
        "largest_output_file_bytes": biggest.stat().st_size,
    }
    return metrics | probe(project / "build", work / "shots" / "evidence", log)


def trial_observable(work: Path, csv: str, log: Path) -> dict[str, Any]:
    project = work / "observable"
    write(project / "package.json", '{"name": "trial-observable", "private": true, "type": "module"}\n')
    write(project / "observablehq.config.js", OBSERVABLE_CONFIG.format(extra=""))
    write(project / "src" / "data" / "weekly.csv", csv)
    write(project / "src" / "index.md", OBSERVABLE_PAGE)
    metrics = install(project, OBSERVABLE_PACKAGES, log)
    cold = run(["npx", "observable", "build"], project, log)
    default_theme = probe(project / "dist", work / "shots" / "observable_default", log)
    write(project / "observablehq.config.js", OBSERVABLE_CONFIG.format(extra=OBSERVABLE_CONFIG_NO_FONTS))
    warm = run(["npx", "observable", "build"], project, log)
    size, files = tree_size(project / "dist")
    biggest = max((p for p in (project / "dist").rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    metrics |= {
        "build_seconds_cold": round(cold, 1),
        "build_seconds_warm": round(warm, 1),
        "output_bytes": size,
        "output_files": files,
        "largest_output_file": biggest.name,
        "largest_output_file_bytes": biggest.stat().st_size,
        "external_hosts_with_default_theme": default_theme["external_hosts_at_runtime"],
    }
    return metrics | probe(project / "dist", work / "shots" / "observable", log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, default=SITE / "_trial")
    args = parser.parse_args()
    work: Path = args.workdir.resolve()
    if work.exists():
        raise SystemExit(f"{work} exists; the trial starts from an empty directory")
    work.mkdir(parents=True)
    log = work / "trial.log"

    load_dotenv()
    with ClickHouse() as ch:
        csv = ch.query(WEEKLY_SQL)

    tools = {"evidence": trial_evidence(work, csv, log), "observable": trial_observable(work, csv, log)}
    metrics = {
        f"{tool}_{name}": {"value": value} for tool, measured in tools.items() for name, value in measured.items()
    }
    path = write_result(
        "site/bi_choice.json",
        metrics,
        source_script="src/radar/site/bi_trial.py",
        mart="marts.mart_weekly_dynamics",
        csv_rows=csv.count("\n") - 1,
        choice="observable_framework",
        status="proposal",
    )
    print(f"wrote {path}")
    for name, metric in metrics.items():
        print(f"{name}\t{metric['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

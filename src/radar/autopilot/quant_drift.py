"""Does the daily run read posts as the history was read? The history of the site was extracted by the AWQ
student (vLLM, GPU); the daily run extracts with the Q4_K_M GGUF of the same student on the runner's CPU.
A different reading would put a step into every series on the day the autopilot starts.

    LLAMA_PORT=8094 STUDENT_GGUF=models/gguf/qwen3_1p7b_r16_clean/qwen3_1p7b_r16_clean-Q4_K_M.gguf \
        bash scripts/llm_server.sh student 4
    python -m radar.autopilot.quant_drift --base-url http://127.0.0.1:8094 [--sample 600]

A random sample of the Telegram IT units of RECENT_YEARS (int_extracted_weekly_units, AWQ fields) is read
again by Q4_K_M with the daily run's input and skill mapping; results/autopilot/quant_drift.json holds the
agreement on role, grade and skills, and each frequent skill's share under both readings. The weights are
the same on GPU and CPU, so the GPU server stands in for the runner here.
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from radar.autopilot.skillmap import SkillMap
from radar.config import load_dotenv
from radar.extract.corpus_run import checked_output, load_backend
from radar.extract.inputs import input_for_record
from radar.results import write_result
from radar.site.market_facts import RECENT_YEARS, fetch
from radar.stream.clickhouse import ClickHouse

SAMPLE_SQL = """
SELECT u.record_source_id AS source_id, u.role AS role, u.grade AS grade, u.skills AS skills, r.text AS text,
       r.payload AS payload
FROM staging.int_extracted_weekly_units AS u
INNER JOIN (
    SELECT source_id, argMax(text, fetched_at) AS text, argMax(payload, fetched_at) AS payload
    FROM raw.vacancies WHERE source = 'telegram' GROUP BY source_id
) AS r ON r.source_id = u.record_source_id
WHERE u.source = 'telegram' AND u.is_it AND toYear(u.week) IN {years}
ORDER BY cityHash64(u.record_source_id)
LIMIT {limit}
"""
TOP = 30


def read_again(rows: list[dict[str, Any]], base_url: str, workers: int) -> list[dict[str, Any] | None]:
    # the daily run's client (airflow/include on PYTHONPATH, as in the workflow)
    served = load_backend("student_cpu:backend", {"base_url": base_url})

    def one(row: dict[str, Any]) -> dict[str, Any] | None:
        payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"] or "{}")
        record = {"source": "telegram", "title": None, "text": row["text"], "payload": payload}
        (answer,) = served.extract([input_for_record(record)])
        output = checked_output(answer)
        return None if output is None else json.loads(output)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def compare(rows: list[dict[str, Any]], answers: list[dict[str, Any] | None]) -> dict[str, Any]:
    mapper = SkillMap()
    pairs = []
    for row, answer in zip(rows, answers, strict=True):
        if answer is None:
            continue
        found = set(mapper.skills(answer.get("skills") or [], "\n" + str(row["text"] or "")))
        pairs.append((row, answer, set(row["skills"]), found))
    n = len(pairs)
    it = sum(1 for _, a, _, _ in pairs if a.get("role") != "non_it")
    same_role = sum(1 for r, a, _, _ in pairs if a.get("role") == r["role"])
    same_grade = sum(1 for r, a, _, _ in pairs if (a.get("grade") or None) == (r["grade"] or None))
    jaccard = [len(x & y) / len(x | y) if x | y else 1.0 for _, _, x, y in pairs]
    awq: Counter[str] = Counter(s for _, _, x, _ in pairs for s in x)
    q4: Counter[str] = Counter(s for _, _, _, y in pairs for s in y)
    top = [s for s, _ in sorted(awq.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]
    metrics: dict[str, Any] = {
        "valid": {"value": round(n / len(rows), 4), "n": n, "denominator": len(rows)},
        "still_it": {"value": round(it / n, 4), "n": it, "denominator": n},
        "same_role": {"value": round(same_role / n, 4), "n": same_role, "denominator": n},
        "same_grade": {"value": round(same_grade / n, 4), "n": same_grade, "denominator": n},
        "skills_jaccard_mean": {"value": round(sum(jaccard) / n, 4)},
        "skills_per_post_awq": {"value": round(sum(awq.values()) / n, 3)},
        "skills_per_post_q4": {"value": round(sum(q4.values()) / n, 3)},
    }
    for s in top:
        metrics[f"share.{s}"] = {"value": round((q4[s] - awq[s]) / n, 4), "awq": awq[s], "q4": q4[s], "posts": n}
    diffs = [abs(metrics[f"share.{s}"]["value"]) for s in top]
    metrics["share_diff_max_top"] = {"value": max(diffs)}
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    load_dotenv()
    with ClickHouse() as ch:
        rows = fetch(ch, SAMPLE_SQL.format(years=tuple(RECENT_YEARS), limit=int(args.sample)))
    metrics = compare(rows, read_again(rows, args.base_url, args.workers))
    path = write_result(
        "autopilot/quant_drift.json",
        metrics,
        source_script="src/radar/autopilot/quant_drift.py",
        history_model="qwen3_1p7b_r16_clean-awq",
        daily_model="qwen3_1p7b_r16_clean-q4_k_m",
        sample=args.sample,
        years=list(RECENT_YEARS),
        definitions={
            "share.<skill>": "share of the sampled posts naming the skill under Q4_K_M minus under AWQ; awq and "
            "q4 are the posts naming it, posts the posts both readings answered",
        },
    )
    print({k: v["value"] for k, v in metrics.items() if not k.startswith("share.")}, "->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

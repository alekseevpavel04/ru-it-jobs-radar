"""When radar_retrain refits the salary model: the rule over the weekly drift summary of radar_drift.

Standard library only: the DAG's decision task imports it under Airflow's interpreter.

The salary model (radar.models.salary) trains on trudvsem (API and snapshots), Habr Career and
SuperJob, never on Telegram, and reads the inputs of radar.models.salary_features.BLOCKS. The drift
summary (results/monitor/drift.json, radar.monitor.drift) scores per source the features below; each
maps to the input block it describes. Not mapped: `region` (the model reads region_group only),
`text_len`, `has_salary_rub` and `salary_mid_rub` (not inputs; the last is the target). The source
block is the source itself, and employment type has no drift feature.

Rule: refit when, for a source the model trains on whose summary status is `ok`, a mapped feature is
both `drifted` (score at or above the summary's threshold, 0.1) and `beyond_spread` (score above the
spread of that feature between the reference weeks themselves). The second condition keeps out a
feature that moves this much from week to week anyway: on 2026-10-05 the snapshots' `region` scored
0.175 against a reference spread of 0.215. A summary older than MAX_AGE is not acted on: radar_drift
did not run that week.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

SALARY_SOURCES = ("trudvsem", "trudvsem_snapshot", "habr", "superjob")
# drift feature of radar.monitor.drift -> input block of radar.models.salary_features.BLOCKS
INPUT_BLOCK = {
    "role_bucket": "role",
    "grade": "grade",
    "region_group": "region",
    "work_format": "work_format",
    "experience_min_years": "experience",
    "no_experience": "experience",
    "n_skills": "skills",
}
MAX_AGE = timedelta(days=7)


@dataclass(frozen=True)
class Decision:
    retrain: bool
    reason: str
    week: str | None = None
    triggers: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"retrain": self.retrain, "reason": self.reason, "week": self.week, "triggers": self.triggers}


def decide(summary: Mapping[str, Any], now: datetime) -> Decision:
    created = datetime.fromisoformat(summary["created_at"])
    if now - created > MAX_AGE:
        return Decision(False, f"the drift summary is from {created:%Y-%m-%d}, older than {MAX_AGE.days} days")

    sources = summary["sources"]
    checked, unavailable, weeks = [], [], []
    triggers: list[dict[str, Any]] = []
    closest: tuple[float, str] | None = None
    for source in SALARY_SOURCES:
        s = sources.get(source)
        status = s["status"] if s else "absent"
        if status != "ok":
            unavailable.append(f"{source} {status}")
            continue
        checked.append(source)
        weeks.append(s["current_week"])
        for f in s["drift"]["features"]:
            block = INPUT_BLOCK.get(f["feature"])
            if block is None:
                continue
            if f["drifted"] and f["beyond_spread"]:
                triggers.append(
                    {
                        "source": source,
                        "feature": f["feature"],
                        "block": block,
                        "score": f["score"],
                        "threshold": f["threshold"],
                        "reference_spread": f["reference_spread"],
                    }
                )
            elif closest is None or f["score"] / f["threshold"] > closest[0]:
                closest = (
                    f["score"] / f["threshold"],
                    f"{source}/{f['feature']} score {f['score']}, threshold {f['threshold']}, "
                    f"reference spread {f['reference_spread']}",
                )

    week = max(weeks) if weeks else None
    skipped = f"; not checked: {', '.join(unavailable)}" if unavailable else ""
    if not checked:
        return Decision(False, f"no source of the salary model has a drift report{skipped}")
    if triggers:
        named = ", ".join(f"{t['source']}/{t['feature']} {t['score']}" for t in triggers)
        return Decision(True, f"input drift beyond the reference spread: {named}{skipped}", week, triggers)
    nearest = f"; nearest {closest[1]}" if closest else ""
    return Decision(
        False,
        f"no input of the salary model drifted beyond its reference spread in {', '.join(checked)} "
        f"(week {week}){nearest}{skipped}",
        week,
    )

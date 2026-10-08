"""When a radar_load run hands the raw layer over to dbt.

radar_dbt is scheduled by the asset raw.vacancies, and the last task of radar_load emits it. One
event per loaded date would mean one dbt build per date during a catch-up or a backfill, each
rebuilding history and marts. So a run that has more load runs queued behind it only remembers
the partitions it replayed (an Airflow Variable), and the last run of the batch emits one event
for all of them. Standard library only, like raw_layer.
"""

PENDING_VARIABLE = "radar_load_pending_dbt"


def hand_over(pending: list[str], replayed: list[str], runs_waiting: int) -> tuple[list[str], list[str]]:
    """(partitions to hand over now, partitions left pending) for `source/dt` items."""
    changed = sorted(set(pending) | set(replayed))
    if runs_waiting > 0:
        return [], changed
    return changed, []

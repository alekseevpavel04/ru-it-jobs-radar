"""Load infra/superset/assets into Superset, overwriting every object with the same uuid.

Run inside the superset container by scripts/superset_import.sh. The database file holds the ClickHouse
address only: the read-only account comes from CLICKHOUSE_RO_USER and CLICKHOUSE_RO_PASSWORD at import
time and Superset stores the password encrypted with SECRET_KEY. After the import every dataset is
queried through that account, so a wrong password, a missing grant or an empty mart stops the command.

    python import_assets.py /app/radar/assets
"""

import os
import sys
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy.engine.url import make_url
from superset.app import create_app

ADMIN = "admin"


def read_assets(root: Path) -> dict[str, str]:
    contents = {}
    for path in sorted(root.rglob("*.yaml")):
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if name.startswith("databases/"):
            config = yaml.safe_load(text)
            url = make_url(config["sqlalchemy_uri"]).set(
                username=os.environ["CLICKHOUSE_RO_USER"], password=os.environ["CLICKHOUSE_RO_PASSWORD"]
            )
            config["sqlalchemy_uri"] = url.render_as_string(hide_password=False)
            text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        contents[name] = text
    return contents


def uuids(contents: dict[str, str], prefix: str) -> set[UUID]:
    return {UUID(str(yaml.safe_load(text)["uuid"])) for name, text in contents.items() if name.startswith(prefix)}


def main(root: Path) -> None:
    contents = read_assets(root)
    app = create_app()
    with app.app_context():
        from superset import db, security_manager
        from superset.commands.importers.v1.assets import ImportAssetsCommand
        from superset.connectors.sqla.models import SqlaTable
        from superset.models.dashboard import Dashboard
        from superset.utils.core import override_user

        with override_user(security_manager.find_user(username=ADMIN)):
            ImportAssetsCommand(contents).run()

        empty = []
        datasets = db.session.query(SqlaTable).filter(SqlaTable.uuid.in_(uuids(contents, "datasets/")))
        for dataset in sorted(datasets, key=lambda d: d.table_name):
            source = f"({dataset.sql})" if dataset.sql else f"{dataset.schema}.{dataset.table_name}"
            rows = int(dataset.database.get_df(f"SELECT count() AS n FROM {source} AS t").iloc[0, 0])
            print(f"dataset {dataset.table_name}: {rows} rows")
            if rows == 0:
                empty.append(dataset.table_name)
        if empty:
            raise SystemExit(f"empty datasets: {', '.join(empty)}")

        dashboards = db.session.query(Dashboard).filter(Dashboard.uuid.in_(uuids(contents, "dashboards/")))
        for dashboard in sorted(dashboards, key=lambda d: d.slug):
            title, charts = dashboard.dashboard_title, len(dashboard.slices)
            print(f"dashboard /superset/dashboard/{dashboard.slug}/ {title}: {charts} charts")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Run the API with uvicorn; settings from the environment and the repo-root .env.

python -m radar.api [--host 127.0.0.1] [--port 8000]
"""

import argparse
import logging
import sys

import uvicorn

from radar.api.app import JsonFormatter, create_app, services_from
from radar.api.settings import from_env
from radar.config import load_dotenv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.api", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    load_dotenv()
    app = create_app(services_from(from_env()))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

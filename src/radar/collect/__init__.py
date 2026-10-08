"""Collectors: sources to the raw layer `data/raw/<source>/dt=YYYY-MM-DD/`."""

import io
import logging
import sys


def setup_logging() -> None:
    """INFO to stdout in UTF-8: on Windows a redirected stdout would otherwise be cp1251."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)

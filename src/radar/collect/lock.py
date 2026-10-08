"""One running collector per source: two runs would write the same download or partition at once."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def single_instance(path: Path) -> Iterator[None]:
    """Hold `path` as a lock file with our PID; a lock left by a dead process is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(path.read_text(encoding="utf-8").strip() or 0)
            except (FileNotFoundError, ValueError):
                holder = 0
            if holder and psutil.pid_exists(holder):
                raise AlreadyRunning(f"{path} is held by PID {holder}") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        break
    try:
        yield
    finally:
        path.unlink(missing_ok=True)

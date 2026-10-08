import os
from pathlib import Path

import pytest

from radar.collect.lock import AlreadyRunning, single_instance


def test_second_holder_is_refused_and_lock_is_released(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    with single_instance(lock):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        with pytest.raises(AlreadyRunning), single_instance(lock):
            pass
    assert not lock.exists()


def test_lock_of_a_dead_process_is_taken_over(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    lock.write_text("999999999", encoding="utf-8")
    with single_instance(lock):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())

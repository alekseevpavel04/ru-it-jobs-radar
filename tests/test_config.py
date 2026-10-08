from pathlib import Path

import pytest

from radar import config


def test_require_fails_on_missing_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RADAR_USER_AGENT", raising=False)
    with pytest.raises(config.ConfigError, match="RADAR_USER_AGENT"):
        config.user_agent()
    monkeypatch.setenv("RADAR_USER_AGENT", "  ")
    with pytest.raises(config.ConfigError):
        config.user_agent()


def test_data_dir_relative_to_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", "somewhere")
    assert config.data_dir() == config.ROOT / "somewhere"
    monkeypatch.delenv("RADAR_DATA_DIR")
    assert config.data_dir() == config.ROOT / "data"


def test_dotenv_does_not_override_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nRADAR_A=from_file\nRADAR_B=from_file\n\nnot a pair\n", encoding="utf-8")
    monkeypatch.setenv("RADAR_A", "from_env")
    monkeypatch.delenv("RADAR_B", raising=False)
    config.load_dotenv(env)
    assert config.optional("RADAR_A") == "from_env"
    assert config.optional("RADAR_B") == "from_file"

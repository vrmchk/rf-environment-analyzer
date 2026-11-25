from pathlib import Path

import pytest

from rfea_backend.settings import SYSTEM_MODEL, Settings

ENV = ("RFEA_DATA_DIR", "RFEA_SCENES_DIR", "RFEA_OUTPUT_DIR", "RFEA_MODEL_DIR")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_take_the_system_model_from_the_outputs_folder(clean_env, tmp_path) -> None:
    clean_env.setenv("RFEA_OUTPUT_DIR", str(tmp_path / "out"))
    clean_env.setenv("RFEA_DATA_DIR", str(tmp_path / "data"))
    s = Settings.from_env()
    assert s.model_dir == tmp_path / "out" / "exp04_uniform_p20" / "spec_n32"
    assert s.scenes_dir == tmp_path / "data" / "scenes"


def test_without_env_the_folders_are_next_to_the_main_checkout(clean_env) -> None:
    s = Settings.from_env()
    assert s.model_dir.parts[-3:] == ("outputs", *SYSTEM_MODEL.parts)
    assert s.scenes_dir.parent.parent == s.model_dir.parents[2]  # the same parent folder


def test_explicit_dirs_win(clean_env, tmp_path) -> None:
    clean_env.setenv("RFEA_OUTPUT_DIR", str(tmp_path / "out"))
    clean_env.setenv("RFEA_MODEL_DIR", str(tmp_path / "m"))
    clean_env.setenv("RFEA_SCENES_DIR", str(tmp_path / "s"))
    s = Settings.from_env()
    assert (s.model_dir, s.scenes_dir) == (Path(tmp_path / "m"), Path(tmp_path / "s"))

"""Settings from environment variables (the same names in Docker and in local runs).

Defaults for a local run follow rfea_ml.paths: RFEA_DATA_DIR and RFEA_OUTPUT_DIR, or the
`data` and `outputs` folders next to the main checkout (found through git, so that every
worktree uses the same ones). The model of the system is SYSTEM_MODEL under the outputs
folder unless RFEA_MODEL_DIR names another one. In Docker the image sets RFEA_SCENES_DIR
and RFEA_MODEL_DIR to the mounted folders (docker-compose.yml).
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql+psycopg://rfea:rfea@localhost:5432/rfea"
# The model chosen for the system (thesis 5.5): TorchSig data with one measured-bandwidth
# distribution for all classes, early-stopping patience 20.
SYSTEM_MODEL = Path("exp04_uniform_p20") / "spec_n32"


def _shared_dir(env_var: str, name: str) -> Path:
    if env := os.environ.get(env_var):
        return Path(env)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        )  # fmt: skip
        return Path(out.stdout.strip()).parent.parent / name
    except OSError, subprocess.CalledProcessError:
        return Path(name)  # no git (e.g. an installed package): relative to the cwd


@dataclass(frozen=True)
class Settings:
    scenes_dir: Path  # SigMF scenes: RFEA_SCENES_DIR or <data>/scenes
    model_dir: Path  # model.pt and classifier.json: RFEA_MODEL_DIR or <outputs>/SYSTEM_MODEL
    database_url: str = DEFAULT_DATABASE_URL  # RFEA_DATABASE_URL

    @classmethod
    def from_env(cls) -> Settings:
        scenes = os.environ.get("RFEA_SCENES_DIR")
        model = os.environ.get("RFEA_MODEL_DIR")
        return cls(
            scenes_dir=Path(scenes) if scenes else _shared_dir("RFEA_DATA_DIR", "data") / "scenes",
            model_dir=Path(model)
            if model
            else _shared_dir("RFEA_OUTPUT_DIR", "outputs") / SYSTEM_MODEL,
            database_url=os.environ.get("RFEA_DATABASE_URL", DEFAULT_DATABASE_URL),
        )

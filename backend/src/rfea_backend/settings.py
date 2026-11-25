"""Settings from environment variables (the same names in Docker and in local runs)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql+psycopg://rfea:rfea@localhost:5432/rfea"


@dataclass(frozen=True)
class Settings:
    scenes_dir: Path  # SigMF scenes: <RFEA_DATA_DIR>/scenes or RFEA_SCENES_DIR
    model_dir: Path  # model.pt and classifier.json: RFEA_MODEL_DIR
    database_url: str = DEFAULT_DATABASE_URL  # RFEA_DATABASE_URL

    @classmethod
    def from_env(cls) -> Settings:
        data = Path(os.environ.get("RFEA_DATA_DIR", "data"))
        return cls(
            scenes_dir=Path(os.environ.get("RFEA_SCENES_DIR", data / "scenes")),
            model_dir=Path(os.environ.get("RFEA_MODEL_DIR", "model")),
            database_url=os.environ.get("RFEA_DATABASE_URL", DEFAULT_DATABASE_URL),
        )

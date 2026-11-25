import json
from pathlib import Path

import numpy as np
import pytest

FS = 1e6


def write_scene(directory: Path, name: str, x: np.ndarray, fs: float = FS) -> None:
    x.astype("<c8").tofile(directory / f"{name}.sigmf-data")
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": fs, "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": 100e6}],
        "annotations": [],
    }
    (directory / f"{name}.sigmf-meta").write_text(json.dumps(meta))


def tone(freq: float, seconds: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    n = int(seconds * FS)
    x = np.exp(2j * np.pi * freq / FS * np.arange(n))
    if noise:
        rng = np.random.default_rng(seed)
        x = x + noise * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return x.astype(np.complex64)


@pytest.fixture
def scenes_dir(tmp_path: Path) -> Path:
    write_scene(tmp_path, "tone", tone(100e3, 0.3, noise=0.1))
    return tmp_path

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rfea_core import SpectrogramCNN

FS = 1e6
CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64",
           "QPSK", "WBFM"]  # fmt: skip


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


def classifier_meta() -> dict:
    q = {"median": 0.218, "p25": 0.146, "p75": 0.306, "p05": 0.069, "p95": 0.431, "records": 1}
    return {
        "classes": CLASSES,
        "temperature": 1.0,
        "frame_len": 128,
        "stft": {"n_fft": 32, "hop": 8, "window": "hann", "out_size": [64, 64]},
        "signal_bandwidth_fraction": [0.15, 0.5],
        "measured_bandwidth_fraction": {"all": q, "by_group": {}},
        "model": "SpectrogramCNN",
        "weights": "model.pt",
        "commit": "test",
    }


def write_model(directory: Path, meta: dict | None = None) -> Path:
    """An untrained model with a valid classifier.json (tests do not need a good one)."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(SpectrogramCNN(num_classes=len(CLASSES)).state_dict(), directory / "model.pt")
    (directory / "classifier.json").write_text(json.dumps(meta or classifier_meta()))
    return directory


@pytest.fixture
def scenes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "scenes"
    d.mkdir()
    write_scene(d, "tone", tone(100e3, 0.3, noise=0.1))
    return d


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return write_model(tmp_path / "model")

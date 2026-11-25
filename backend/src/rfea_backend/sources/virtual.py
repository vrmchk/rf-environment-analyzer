"""Virtual Source: a SigMF scene file read as if it were a live receiver.

The file is memory-mapped, so a 60 s scene (480 MB) is not loaded at once. Real-time
pacing is done by the session loop (pipeline.py), not here, so tests can read fast.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_SUFFIX, META_SUFFIX = ".sigmf-data", ".sigmf-meta"


@dataclass(frozen=True)
class SceneInfo:
    name: str
    sample_rate: float
    center_freq: float
    duration_s: float
    n_signals: int

    @classmethod
    def read(cls, meta_path: Path) -> SceneInfo:
        meta = json.loads(meta_path.read_text())
        g = meta["global"]
        if g.get("core:datatype") != "cf32_le":
            raise ValueError(f"{meta_path}: only cf32_le is supported")
        fs = float(g["core:sample_rate"])
        n = meta_path.with_suffix(DATA_SUFFIX).stat().st_size // 8
        return cls(
            name=meta_path.name.removesuffix(META_SUFFIX),
            sample_rate=fs,
            center_freq=float(meta["captures"][0].get("core:frequency", 0.0)),
            duration_s=n / fs,
            n_signals=len(meta.get("annotations", [])),
        )


def list_scenes(scenes_dir: Path) -> list[SceneInfo]:
    """Scenes that have both SigMF files; unreadable ones are skipped."""
    out = []
    for meta in sorted(scenes_dir.glob(f"*{META_SUFFIX}")):
        if meta.with_suffix(DATA_SUFFIX).exists():
            try:
                out.append(SceneInfo.read(meta))
            except ValueError, KeyError, json.JSONDecodeError:
                continue
    return out


class VirtualSource:
    def __init__(self, scenes_dir: Path, name: str):
        meta_path = scenes_dir / f"{name}{META_SUFFIX}"
        if Path(name).name != name or not meta_path.exists():
            raise FileNotFoundError(f"scene {name!r} not found")
        self.info = SceneInfo.read(meta_path)
        self.sample_rate = self.info.sample_rate
        self.center_freq = self.info.center_freq
        self._data = np.memmap(meta_path.with_suffix(DATA_SUFFIX), dtype="<c8", mode="r")
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def read(self, n: int) -> np.ndarray | None:
        if self._pos >= len(self._data):
            return None
        block = np.array(self._data[self._pos : self._pos + n], dtype=np.complex64)
        self._pos += len(block)
        return block

    def close(self) -> None:
        del self._data

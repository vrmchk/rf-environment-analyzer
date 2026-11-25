"""Classifier model: loading and the compatibility check at start-up (NFR-8).

The model directory holds model.pt (state dict) and classifier.json written by experiment 04.
Every check below must pass, or the model is not loaded and the reason is reported by
/api/health and on session start.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rfea_core import SpectrogramCNN, StftConfig, log_spectrogram

FRAME_LEN = 128
STFT = StftConfig(n_fft=32)  # the approved spectrogram (method.md, section 1)
REQUIRED = ("classes", "temperature", "frame_len", "stft", "model", "weights",
            "measured_bandwidth_fraction")  # fmt: skip
ARCHITECTURES = {"SpectrogramCNN": SpectrogramCNN}


class ModelError(Exception):
    pass


@dataclass(frozen=True)
class BandwidthRange:
    median: float
    p05: float
    p95: float


class Classifier:
    def __init__(self, model: torch.nn.Module, meta: dict, sha256: str):
        self.model = model.eval()
        self.meta = meta
        self.classes: list[str] = list(meta["classes"])
        self.temperature = float(meta["temperature"])
        m = meta["measured_bandwidth_fraction"]["all"]
        self.bandwidth = BandwidthRange(float(m["median"]), float(m["p05"]), float(m["p95"]))
        self.sha256 = sha256

    @torch.inference_mode()
    def logits(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 2, FRAME_LEN) float IQ frames -> (B, n_classes) logits."""
        return self.model(log_spectrogram(frames, STFT))

    def describe(self) -> dict:
        return {"classes": self.classes, "temperature": self.temperature,
                "commit": self.meta.get("commit"), "sha256": self.sha256,
                "bandwidth_median": self.bandwidth.median}  # fmt: skip


def load_classifier(model_dir: Path) -> Classifier:
    meta_path = model_dir / "classifier.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ModelError(f"cannot read {meta_path}: {e}") from e
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ModelError(f"classifier.json lacks {', '.join(missing)}")
    stft = meta["stft"]
    expected = {"n_fft": STFT.n_fft, "hop": STFT.hop, "window": "hann",
                "out_size": list(STFT.out_size)}  # fmt: skip
    if {k: stft.get(k) for k in expected} != expected:
        raise ModelError(f"STFT {stft} differs from the system's {expected}")
    if meta["frame_len"] != FRAME_LEN:
        raise ModelError(f"frame_len {meta['frame_len']} != {FRAME_LEN}")
    arch = ARCHITECTURES.get(meta["model"])
    if arch is None:
        raise ModelError(f"unknown model {meta['model']!r}")
    t = meta["temperature"]
    if not isinstance(t, int | float) or not 0 < t < 10:
        raise ModelError(f"temperature {t!r} outside (0, 10)")
    try:
        bw = meta["measured_bandwidth_fraction"]["all"]
        if not 0 < bw["p05"] <= bw["median"] <= bw["p95"] < 1:
            raise ModelError(f"implausible measured bandwidth {bw}")
    except (KeyError, TypeError) as e:
        raise ModelError(f"measured_bandwidth_fraction is malformed: {e}") from e

    weights = model_dir / Path(str(meta["weights"])).name
    model = arch(num_classes=len(meta["classes"]))
    try:
        raw = weights.read_bytes()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, KeyError) as e:
        raise ModelError(f"cannot load {weights}: {e}") from e
    clf = Classifier(model, meta, hashlib.sha256(raw).hexdigest())
    out = clf.logits(torch.randn(1, 2, FRAME_LEN, generator=torch.Generator().manual_seed(0)))
    if out.shape != (1, len(clf.classes)):
        raise ModelError(f"model output {tuple(out.shape)} != (1, {len(clf.classes)})")
    return clf

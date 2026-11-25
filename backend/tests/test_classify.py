import json

import numpy as np
import pytest
import torch
from conftest import FS, classifier_meta, write_model

from rfea_backend.classify.model import ModelError, load_classifier
from rfea_backend.dsp.channelizer import FRAME_LEN, Band, estimate_band, frames
from rfea_core import occupied_fraction


def test_model_loads_and_classifies(model_dir) -> None:
    clf = load_classifier(model_dir)
    assert len(clf.classes) == 11 and clf.bandwidth.median == 0.218
    out = clf.logits(torch.randn(3, 2, FRAME_LEN))
    assert out.shape == (3, 11)


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda m: m.pop("measured_bandwidth_fraction"), "lacks"),
        (lambda m: m["stft"].update(n_fft=64), "STFT"),
        (lambda m: m.update(frame_len=256), "frame_len"),
        (lambda m: m.update(model="Other"), "unknown model"),
        (lambda m: m.update(temperature=0), "temperature"),
        (lambda m: m.update(classes=m["classes"][:5]), "cannot load"),
    ],
)
def test_incompatible_models_are_rejected(tmp_path, change, reason: str) -> None:
    meta = classifier_meta()
    change(meta)
    d = write_model(tmp_path / "m", meta)
    with pytest.raises(ModelError, match=reason):
        load_classifier(d)


def test_missing_model_is_reported(tmp_path) -> None:
    with pytest.raises(ModelError, match="cannot read"):
        load_classifier(tmp_path)


def test_band_estimate_from_a_flat_spectrum() -> None:
    n = 1024
    p = np.ones(n)  # noise 1 everywhere
    p[600:640] += 99  # 40 bins = 39 kHz of signal, centered at bin 619.5
    band = estimate_band(p, 1.0, 600, 639, FS, 0.22)
    assert band.bandwidth_hz == pytest.approx(0.9 * 40 * FS / n, rel=0.05)
    assert band.center_hz == pytest.approx((619.5 - n / 2) * FS / n, abs=FS / n)
    assert band.decimation == round(FS / (band.bandwidth_hz / 0.22))
    assert band.fraction == pytest.approx(0.22, rel=0.1)


def test_frames_bring_a_signal_to_the_training_bandwidth() -> None:
    rng = np.random.default_rng(0)
    # band-limited noise: 30 kHz wide at +150 kHz
    n = 50_000
    spec = np.zeros(n, complex)
    k = np.fft.fftfreq(n, 1 / FS)
    sel = np.abs(k - 150e3) < 15e3
    spec[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(spec).astype(np.complex64)
    band = Band(center_hz=150e3, bandwidth_hz=0.9 * 30e3, decimation=8, fraction=0.216)
    f = frames(x, band, FS, 4)
    assert f.shape == (4, 2, FRAME_LEN) and f.dtype == np.float32
    z = f[:, 0] + 1j * f[:, 1]
    p = (np.abs(np.fft.fft(z, axis=1)) ** 2).mean(0)  # averaged over the 4 frames
    measured = occupied_fraction(torch.from_numpy(p)).item()
    # 27 kHz of 125 kHz after decimation by 8: ~0.22
    assert 0.15 < measured < 0.3
    centroid = (p * np.fft.fftfreq(FRAME_LEN)).sum() / p.sum()
    assert abs(centroid) < 0.03


def test_classifier_json_of_exp04_shape_is_accepted(model_dir) -> None:
    meta = json.loads((model_dir / "classifier.json").read_text())
    assert set(meta["measured_bandwidth_fraction"]["all"]) >= {"median", "p05", "p95"}

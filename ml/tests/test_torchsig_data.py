import numpy as np
import pytest
import torch

from rfea_core import occupied_bandwidth
from rfea_ml.torchsig_data import (
    FRAME_LEN,
    MARGIN,
    RECORD_LEN,
    GenConfig,
    Impairments,
    add_noise,
    frames,
    generate_class,
    impair,
    split_by_record,
    to_measured,
)


def test_add_noise_sets_snr_by_total_power() -> None:
    rng = np.random.default_rng(0)
    s = 3.0 * np.exp(1j * rng.uniform(0, 2 * np.pi, 200_000))
    for snr in (-10, 0, 10):
        x = add_noise(s, snr, rng)
        noise = x - s / 3.0  # signal is normalized to unit power
        measured = 10 * np.log10(1.0 / np.mean(np.abs(noise) ** 2))
        assert measured == pytest.approx(snr, abs=0.05)


def test_impair_keeps_length_and_power() -> None:
    rng = np.random.default_rng(1)
    s = np.exp(1j * 2 * np.pi * 0.05 * np.arange(RECORD_LEN + 64))
    x = impair(s, Impairments(), rng)
    assert x.shape == (RECORD_LEN,)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=1e-3)


def test_split_by_record_is_disjoint_and_complete() -> None:
    parts = split_by_record(250, 20, (0.6, 0.2, 0.2))
    assert [len(p) for p in parts] == [150 * 20, 50 * 20, 50 * 20]
    assert len(np.unique(np.concatenate(parts))) == 250 * 20


def test_frames_are_consecutive_slices() -> None:
    x = (np.arange(3 * RECORD_LEN) + 1j * np.arange(3 * RECORD_LEN)).reshape(3, RECORD_LEN)
    f = frames(x, starts=np.array([0, 5, 31]), count=1)
    assert f.shape == (3, 1, 2, FRAME_LEN)
    assert f[1, 0, 0, 0] == x[1, 5 * FRAME_LEN].real
    f = frames(x, count=32)
    assert np.array_equal(f[0, :, 1].reshape(-1), x[0].imag)


def test_generate_class_is_deterministic_and_alternates_names() -> None:
    cfg = GenConfig(snrs=(0, 10))
    a = generate_class(2, ("am-usb", "am-lsb"), 2, cfg)
    b = generate_class(2, ("am-usb", "am-lsb"), 2, cfg)
    assert a["x"].shape == (4, RECORD_LEN)
    assert a["x"].dtype == np.complex64
    assert list(a["tsig_class"]) == ["am-usb", "am-lsb"] * 2
    assert list(a["snr"]) == [0, 0, 10, 10]
    assert np.all((a["bandwidth"] >= 0.15) & (a["bandwidth"] <= 0.5))
    assert np.array_equal(a["x"], b["x"])


@pytest.mark.parametrize("name", ["qpsk", "16qam", "fm", "am-usb"])
def test_generate_class_respects_bandwidth_range(name: str) -> None:
    cfg = GenConfig(snrs=(10,), bandwidth=(0.10, 0.20))
    bw = generate_class(0, (name,), 8, cfg)["bandwidth"]
    assert np.all((bw >= 0.10) & (bw <= 0.20))


@pytest.mark.parametrize("name", ["qpsk", "fm", "2fsk"])
def test_generate_class_sets_measured_bandwidth_for_any_class(name: str) -> None:
    # FM is ~3x narrower than QPSK at the same TorchSig bandwidth; --measured removes that
    cfg = GenConfig(snrs=(40,), measured=(0.25, 0.25))
    x = generate_class(0, (name,), 6, cfg)["x"]
    m = occupied_bandwidth(torch.from_numpy(x)).numpy()
    assert np.all(np.abs(m - 0.25) < 0.05)  # FM varies with its message


def test_to_measured_narrows_and_widens() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=20 * RECORD_LEN) + 1j * rng.normal(size=20 * RECORD_LEN)
    s = np.convolve(w, np.sinc(0.2 * np.arange(-200, 201)), mode="same")  # 0.2 of the rate
    for target in (0.1, 0.4):
        y, r = to_measured(s, target)
        assert y.shape == (RECORD_LEN + MARGIN,)
        m = float(occupied_bandwidth(torch.from_numpy(y[None].astype(np.complex64)))[0])
        assert m == pytest.approx(target, abs=0.02)

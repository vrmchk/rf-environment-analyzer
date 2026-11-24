import numpy as np
import pytest

from rfea_ml.features import FEATURE_NAMES, svm_features

N = 200_000  # long noise-free symbol streams, so sample moments match theory


def _iq(symbols: np.ndarray) -> np.ndarray:
    return np.stack([symbols.real, symbols.imag])[None].astype(np.float64)


def _qam(order: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.sqrt(order))
    levels = np.arange(side) * 2 - (side - 1)
    return rng.choice(levels, N) + 1j * rng.choice(levels, N)


@pytest.mark.parametrize(
    ("name", "make", "c42", "abs_c20"),
    [
        ("bpsk", lambda r: r.choice([-1.0, 1.0], N) + 0j, -2.0, 1.0),
        ("qpsk", lambda r: np.exp(1j * (np.pi / 4 + np.pi / 2 * r.integers(0, 4, N))), -1.0, 0.0),
        ("8psk", lambda r: np.exp(1j * np.pi / 4 * r.integers(0, 8, N)), -1.0, 0.0),
        ("qam16", lambda r: _qam(16, r), -0.68, 0.0),
        ("qam64", lambda r: _qam(64, r), -0.619, 0.0),
    ],
)
def test_cumulants_match_theory(name, make, c42, abs_c20) -> None:
    rng = np.random.default_rng(0)
    symbols = make(rng) * 0.01  # RadioML-like amplitude; features are scale-invariant
    f = dict(zip(FEATURE_NAMES, svm_features(_iq(symbols))[0], strict=True))
    assert f["c42"] == pytest.approx(c42, abs=0.02), name
    assert f["abs_c20"] == pytest.approx(abs_c20, abs=0.02), name


def test_constant_envelope_and_frequency_features() -> None:
    n = np.arange(1024)
    tone = np.exp(1j * (2 * np.pi * 0.05 * n + 0.7))  # frequency and phase offset
    f = dict(zip(FEATURE_NAMES, svm_features(_iq(tone))[0], strict=True))
    assert f["sigma_a"] == pytest.approx(0.0, abs=1e-9)
    assert f["gamma_max"] == pytest.approx(0.0, abs=1e-9)
    assert f["sigma_f"] == pytest.approx(0.0, abs=1e-9)  # constant phase increment
    assert f["c42"] == pytest.approx(-1.0, abs=1e-3)  # M20 ~ 0 up to finite-N leakage

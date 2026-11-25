import math

import pytest
import torch

from rfea_core import occupied_bandwidth, occupied_fraction

N = 4096


def band_noise(width: float, n: int = N, seed: int = 0) -> torch.Tensor:
    """Complex noise flat over |f| < width / 2 (fractions of the sample rate)."""
    g = torch.Generator().manual_seed(seed)
    spec = torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g))
    f = torch.fft.fftfreq(n)
    spec[f.abs() >= width / 2] = 0
    return torch.fft.ifft(spec)


def test_fraction_of_a_flat_power_spectrum() -> None:
    p = torch.zeros(1000, dtype=torch.float64)
    p[:250] = 1
    assert occupied_fraction(p).item() == pytest.approx(0.9 * 0.25, abs=1e-3)
    assert occupied_fraction(p, power_fraction=1.0).item() == pytest.approx(0.25)


def test_tone_takes_the_window_main_lobe() -> None:
    n = torch.arange(N)
    tone = torch.exp(1j * 2 * math.pi * (100.25 / N) * n)
    assert occupied_bandwidth(tone).item() <= 3 / 256


@pytest.mark.parametrize("width", [0.1, 0.25, 0.5])
def test_band_limited_noise_takes_ninety_percent_of_its_band(width: float) -> None:
    bw = occupied_bandwidth(band_noise(width)).item()
    assert bw == pytest.approx(0.9 * width, abs=0.03)


def test_white_noise_fills_the_band_and_batches() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.complex(torch.randn(3, N, generator=g), torch.randn(3, N, generator=g))
    bw = occupied_bandwidth(x)
    assert bw.shape == (3,)
    # 16 averaged segments: bins vary by ~25 %, so 90 % of the power takes ~83 % of the bins
    assert torch.all((bw > 0.78) & (bw < 0.9))


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        occupied_bandwidth(torch.ones(4096, dtype=torch.complex64), power_fraction=0)
    with pytest.raises(ValueError):
        occupied_bandwidth(torch.ones(100, dtype=torch.complex64))

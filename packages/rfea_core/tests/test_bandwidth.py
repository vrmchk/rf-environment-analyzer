import math

import pytest
import torch

from rfea_core import occupied_bandwidth

N = 4096


def test_tone_takes_one_bin() -> None:
    n = torch.arange(N)
    tone = torch.exp(1j * 2 * math.pi * (100 / N) * n)
    assert occupied_bandwidth(tone).item() == pytest.approx(1 / N)


def test_flat_band_takes_power_fraction_of_its_width() -> None:
    # Equal power in bins -B/2..B/2 (B = 0.25 fs): 90 % of it is 0.9 * 0.25 of the band.
    spec = torch.zeros(N, dtype=torch.complex128)
    width = N // 4
    spec[: width // 2] = 1
    spec[-width // 2 :] = 1
    x = torch.fft.ifft(spec)
    assert occupied_bandwidth(x).item() == pytest.approx(0.9 * 0.25, abs=2 / N)
    assert occupied_bandwidth(x, power_fraction=1.0).item() == pytest.approx(0.25, abs=1 / N)


def test_white_noise_fills_the_band_and_batches() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.complex(torch.randn(3, N, generator=g), torch.randn(3, N, generator=g))
    bw = occupied_bandwidth(x)
    assert bw.shape == (3,)
    # Bin powers are exponential; the top share q of bins holds q (1 - ln q) of the power,
    # which reaches 0.9 at q ~ 0.595.
    assert torch.all((bw > 0.57) & (bw < 0.62))


def test_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        occupied_bandwidth(torch.ones(8, dtype=torch.complex64), power_fraction=0)

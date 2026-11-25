"""Waterfall rows: averaged power spectrum of one block (Welch) and quantization to 0-255.

A row is the mean of the Hann-windowed N-point power spectra of the block's consecutive
non-overlapping windows (N = 1024 at 1 MHz: 977 Hz per bin; a 50 ms block averages 48
windows, so noise in a row varies by about 0.6 dB instead of 5.6 dB). Levels are quantized
relative to the noise: from P_noise - 10 dB to P_noise + 50 dB, P_noise being the row
median (thesis (3.9)) averaged over the last NOISE_ROWS rows, so colors do not drift.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class WaterfallConfig:
    n_fft: int = 1024
    row_samples: int = 50_000  # 50 ms at 1 MHz -> 20 rows/s
    below_noise_db: float = 10.0
    above_noise_db: float = 50.0
    noise_rows: int = 40  # 2 s at 20 rows/s


def row_power_db(block: np.ndarray, n_fft: int) -> np.ndarray:
    """(n_fft,) averaged power in dB, frequency ascending (-fs/2 ... fs/2)."""
    k = len(block) // n_fft
    if k == 0:
        raise ValueError("block shorter than one FFT window")
    w = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.fft(block[: k * n_fft].reshape(k, n_fft) * w, axis=1)
    p = np.mean(spec.real**2 + spec.imag**2, axis=0) / np.sum(w**2)
    return 10 * np.log10(np.fft.fftshift(p) + EPS)


class Waterfall:
    """Turns blocks into quantized rows; keeps the smoothed noise level."""

    def __init__(self, cfg: WaterfallConfig | None = None):
        self.cfg = cfg or WaterfallConfig()
        self._noise = deque(maxlen=cfg.noise_rows)

    def row(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Returns (power dB, levels uint8, lo_db, hi_db)."""
        p = row_power_db(block, self.cfg.n_fft)
        self._noise.append(float(np.median(p)))
        noise = float(np.mean(self._noise))
        lo, hi = noise - self.cfg.below_noise_db, noise + self.cfg.above_noise_db
        q = np.clip((p - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return p, q, lo, hi

"""Narrowband signal of a frame: bandwidth estimate, shift, low-pass, decimation, frames.

The classifier was trained on signals whose measured bandwidth (90 % of the power) is a
known share of the sample rate, median 0.22 (classifier.json). So for a tracked signal:
  1. its measured bandwidth is estimated from the track's averaged power spectrum with the
     noise level subtracted, by the same rule (rfea_core.occupied_fraction), over the frame
     widened by half its width on each side;
  2. the target rate is f_s' = bandwidth / 0.22; the decimation factor D = round(fs / f_s');
  3. a frame of 128 output samples is cut: the input is shifted by the power centroid of the
     signal (an SSB signal is not centered on its frame), low-pass filtered (windowed sinc,
     63 taps, cut-off f_s'/2) and every D-th sample is kept.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from rfea_core import occupied_fraction

TAPS = 63
FRAME_LEN = 128


@dataclass(frozen=True)
class Band:
    center_hz: float  # offset from the center frequency of the stream
    bandwidth_hz: float  # measured (90 % of the power)
    decimation: int
    fraction: float  # bandwidth / f_s' after the integer decimation


def estimate_band(
    power_mean: np.ndarray,
    noise_mean: float,
    k1: int,
    k2: int,
    fs: float,
    target_fraction: float,
) -> Band:
    """power_mean: (n,) linear row power averaged over the track; noise_mean: linear noise."""
    n = len(power_mean)
    w = k2 - k1 + 1
    a, b = max(0, k1 - w // 2), min(n, k2 + w // 2 + 1)
    p = np.clip(power_mean[a:b] - noise_mean, 0, None)
    df = fs / n
    if p.sum() <= 0:
        p = np.ones(b - a)
    bandwidth = float(occupied_fraction(torch.from_numpy(p)).item()) * (b - a) * df
    bins = np.arange(a, b)
    center = float(((bins - n / 2) * p).sum() / p.sum()) * df
    decimation = max(1, round(fs / (bandwidth / target_fraction)))
    return Band(center, bandwidth, decimation, bandwidth * decimation / fs)


def lowpass(decimation: int) -> np.ndarray:
    """Windowed-sinc low-pass with cut-off at half the output rate, unit DC gain."""
    m = np.arange(TAPS) - (TAPS - 1) / 2
    h = np.sinc(m / decimation) * np.hamming(TAPS)
    return (h / h.sum()).astype(np.float32)


def frames(block: np.ndarray, band: Band, fs: float, count: int) -> np.ndarray:
    """Up to `count` frames spread evenly over the block: (k, 2, FRAME_LEN) float32."""
    d = band.decimation
    need = (FRAME_LEN - 1) * d + TAPS
    if len(block) < need:
        return np.zeros((0, 2, FRAME_LEN), dtype=np.float32)
    starts = np.linspace(0, len(block) - need, count).astype(int) if count > 1 else [0]
    h = lowpass(d)[::-1]
    out = []
    for s in starts:
        seg = block[s : s + need]
        mix = np.exp(-2j * np.pi * band.center_hz / fs * (s + np.arange(need))).astype(np.complex64)
        y = sliding_window_view(seg * mix, TAPS)[::d][:FRAME_LEN] @ h
        out.append(np.stack([y.real, y.imag]))
    return np.asarray(out, dtype=np.float32)

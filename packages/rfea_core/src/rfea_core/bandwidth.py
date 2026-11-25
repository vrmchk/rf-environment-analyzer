"""Measured (occupied) bandwidth of a complex baseband signal.

The classifier is trained on signals that take a known share of the sample rate, and the
backend decimates a detected signal so that it takes the same share (method.md, section 1).
Generator metadata does not say how wide a signal looks: for FM, FSK and AM the occupied
band is 2-5 times narrower than the nominal one. So the training data and the backend use
one measured quantity: the fewest frequency bins of the averaged power spectrum that hold
`power_fraction` of the power, as a fraction of the sample rate.

The spectrum must be averaged (Welch: Hann-windowed segments of SEGMENT samples): bins of a
single periodogram of a noise-like signal fluctuate exponentially, and the fewest bins
holding 90 % of such a spectrum cover only ~2/3 of the band. The backend measures on the
waterfall rows, which are averaged too, so both sides see the same quantity.
"""

import torch

SEGMENT = 256


def occupied_fraction(power: torch.Tensor, power_fraction: float = 0.9) -> torch.Tensor:
    """Share of the bins of a power spectrum (..., n) that, taken from the strongest down,
    first reach `power_fraction` of the total. The bins need not be contiguous, so a
    signal with gaps in its spectrum is not counted as wider than it is."""
    if not 0 < power_fraction <= 1:
        raise ValueError("power_fraction must be in (0, 1]")
    s = power.sort(dim=-1, descending=True).values
    share = s.cumsum(dim=-1) / s.sum(dim=-1, keepdim=True)
    n_bins = (share < power_fraction).sum(dim=-1) + 1  # the bin that crosses the fraction
    return n_bins.to(torch.float64) / power.shape[-1]


def averaged_power(x: torch.Tensor, segment: int = SEGMENT) -> torch.Tensor:
    """Welch power spectrum (..., segment) of complex samples (..., n), n >= segment:
    mean of the periodograms of non-overlapping Hann-windowed segments."""
    n = x.shape[-1]
    if n < segment:
        raise ValueError(f"need at least {segment} samples")
    k = n // segment
    s = x[..., : k * segment].reshape(*x.shape[:-1], k, segment)
    w = torch.hann_window(segment, dtype=s.real.dtype, device=s.device)
    return (torch.fft.fft(s * w, dim=-1).abs() ** 2).mean(dim=-2)


def occupied_bandwidth(
    x: torch.Tensor, power_fraction: float = 0.9, segment: int = SEGMENT
) -> torch.Tensor:
    """Occupied bandwidth as a fraction of the sample rate.

    x: complex tensor (..., n) of baseband samples, n >= segment; returns (...,) values in
    (0, 1], resolution 1 / segment.
    """
    return occupied_fraction(averaged_power(x, segment), power_fraction)

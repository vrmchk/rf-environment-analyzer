"""Measured (occupied) bandwidth of a complex baseband signal.

The classifier is trained on signals that take a known share of the sample rate, and the
backend decimates a detected signal so that it takes the same share (method.md, section 1).
Generator metadata does not say how wide a signal looks: for FM, FSK and AM the occupied
band is 2-5 times narrower than the nominal one. So the training data and the backend use
one measured quantity: the fewest FFT bins that hold `power_fraction` of the signal power,
as a fraction of the sample rate.
"""

import torch


def occupied_bandwidth(x: torch.Tensor, power_fraction: float = 0.9) -> torch.Tensor:
    """Occupied bandwidth as a fraction of the sample rate.

    x: complex tensor (..., n) of baseband samples.
    Returns (...,) float tensor in (0, 1]: the share of the n FFT bins that, taken from the
    strongest down, first reach `power_fraction` of the total power. The bins need not be
    contiguous, so a signal with gaps in its spectrum is not counted as wider than it is.
    """
    if not 0 < power_fraction <= 1:
        raise ValueError("power_fraction must be in (0, 1]")
    p = torch.fft.fft(x, dim=-1).abs() ** 2
    s = p.sort(dim=-1, descending=True).values
    share = s.cumsum(dim=-1) / s.sum(dim=-1, keepdim=True)
    n_bins = (share < power_fraction).sum(dim=-1) + 1  # the bin that crosses the fraction
    return n_bins.to(torch.float64) / x.shape[-1]

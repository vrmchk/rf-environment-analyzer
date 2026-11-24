"""STFT spectrogram of complex IQ samples.

The same function is used for training and inference, so the model always sees
spectrograms computed in exactly the same way.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPS = 1e-12


@dataclass(frozen=True)
class StftConfig:
    """Parameters of the time-frequency transform.

    n_fft: window length in samples (also the number of frequency bins).
    hop_length: step between windows; defaults to n_fft // 4 (75% overlap).
    out_size: (frequency, time) size the spectrogram is resized to, so that
        every window length feeds the same network input.
    """

    n_fft: int
    hop_length: int | None = None
    out_size: tuple[int, int] = (64, 64)

    @property
    def hop(self) -> int:
        return self.hop_length if self.hop_length is not None else self.n_fft // 4


def iq_to_complex(iq: torch.Tensor) -> torch.Tensor:
    """(..., 2, N) real tensor with I and Q rows -> (..., N) complex tensor."""
    return torch.complex(iq[..., 0, :], iq[..., 1, :])


def log_spectrogram(iq: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """Batch of IQ examples -> normalized log-power spectrograms.

    iq: (B, 2, N) float tensor.
    Returns (B, 1, F, T) float tensor with F, T = cfg.out_size and values in [0, 1].

    Steps: S(m, k) = |STFT(m, k)|^2 with a Hann window, two-sided spectrum with the
    zero frequency in the middle, power in dB, per-example min-max scaling to [0, 1],
    bilinear resize to cfg.out_size.
    """
    x = iq_to_complex(iq.float())
    window = torch.hann_window(cfg.n_fft, device=x.device)
    spec = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        window=window,
        center=True,
        return_complex=True,
        onesided=False,  # complex input: negative and positive frequencies differ
    )
    power = spec.real.square() + spec.imag.square()  # (B, F, T)
    power = torch.fft.fftshift(power, dim=-2)
    db = 10.0 * torch.log10(power + _EPS)

    lo = db.amin(dim=(-2, -1), keepdim=True)
    hi = db.amax(dim=(-2, -1), keepdim=True)
    norm = (db - lo) / (hi - lo + _EPS)

    return F.interpolate(norm.unsqueeze(1), size=cfg.out_size, mode="bilinear", align_corners=False)

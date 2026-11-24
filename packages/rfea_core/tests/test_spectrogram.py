import math

import pytest
import torch

from rfea_core import SpectrogramCNN, StftConfig, log_spectrogram


@pytest.mark.parametrize("n_fft", [16, 32, 64])
def test_output_shape_and_range(n_fft: int) -> None:
    iq = torch.randn(4, 2, 128)
    out = log_spectrogram(iq, StftConfig(n_fft=n_fft))
    assert out.shape == (4, 1, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


def test_tone_lands_in_expected_frequency_bin() -> None:
    # Complex tone at +fs/4 must appear in the upper half of the shifted spectrum,
    # at 3/4 of the frequency axis; -fs/4 at 1/4.
    n = torch.arange(128)
    for freq, expected in [(0.25, 0.75), (-0.25, 0.25)]:
        phase = 2 * math.pi * freq * n
        iq = torch.stack([torch.cos(phase), torch.sin(phase)]).unsqueeze(0)
        cfg = StftConfig(n_fft=32, out_size=(32, 17))
        spec = log_spectrogram(iq, cfg)[0, 0]
        peak_bin = spec.mean(dim=1).argmax().item()
        assert abs(peak_bin / 32 - expected) <= 1 / 32


def test_cnn_forward() -> None:
    model = SpectrogramCNN(num_classes=11)
    logits = model(torch.rand(2, 1, 64, 64))
    assert logits.shape == (2, 11)

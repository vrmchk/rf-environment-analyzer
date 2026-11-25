"""Shared signal processing and model code used by both training (ml/) and the backend."""

from rfea_core.accumulation import (
    Decision,
    DecisionAccumulator,
    Status,
    StoppingResult,
    accumulate,
    apply_stopping_rule,
)
from rfea_core.bandwidth import occupied_bandwidth
from rfea_core.calibration import calibrated_log_probs, fit_temperature
from rfea_core.models import SpectrogramCNN
from rfea_core.spectrogram import StftConfig, iq_to_complex, log_spectrogram

__all__ = [
    "Decision",
    "DecisionAccumulator",
    "SpectrogramCNN",
    "Status",
    "StftConfig",
    "StoppingResult",
    "accumulate",
    "apply_stopping_rule",
    "calibrated_log_probs",
    "fit_temperature",
    "iq_to_complex",
    "log_spectrogram",
    "occupied_bandwidth",
]

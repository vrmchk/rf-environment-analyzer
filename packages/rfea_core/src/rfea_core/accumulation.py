"""Accumulation of per-frame decisions and the stopping rule (thesis, subsection 3.7).

For one tracked signal with calibrated per-frame distributions p_k(c):
  Lambda_K(c) = sum_{k<=K} log p_k(c) = Lambda_{K-1}(c) + log p_K(c)      (3.21)
  q_K(c) = softmax(Lambda_K)(c),  y_K = argmax_c Lambda_K(c)               (3.22)
Stopping rule (3.23): decide y_K as soon as max_c q_K(c) >= tau; if K reaches K_max
without that, the signal is "unknown"; otherwise keep observing.

DecisionAccumulator is the streaming form used per signal by the backend;
accumulate() and apply_stopping_rule() are the same computation vectorized over many
sequences, used in experiments.
"""

from dataclasses import dataclass
from enum import StrEnum

import torch

from rfea_core.calibration import calibrated_log_probs


class Status(StrEnum):
    OBSERVING = "observing"
    IDENTIFIED = "identified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    status: Status
    predicted_class: int  # current argmax of the accumulated evidence
    confidence: float  # max_c q_K(c)
    n_frames: int  # K


class DecisionAccumulator:
    """Accumulates calibrated log-probabilities of one signal frame by frame."""

    def __init__(self, n_classes: int, threshold: float, max_frames: int, temperature: float = 1.0):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.threshold = threshold
        self.max_frames = max_frames
        self.temperature = temperature
        self.log_evidence = torch.zeros(n_classes, dtype=torch.float64)  # Lambda_K
        self.n_frames = 0
        self.decision: Decision | None = None

    @property
    def done(self) -> bool:
        return self.decision is not None and self.decision.status != Status.OBSERVING

    def update(self, logits: torch.Tensor) -> Decision:
        """Add one frame (raw network logits, shape (M,)) and apply the stopping rule."""
        if self.done:
            raise RuntimeError("decision already made for this signal")
        log_p = calibrated_log_probs(logits.detach().to("cpu", torch.float64), self.temperature)
        self.log_evidence += log_p
        self.n_frames += 1

        q = torch.softmax(self.log_evidence, dim=0)
        confidence, predicted = q.max(dim=0)
        if confidence >= self.threshold:
            status = Status.IDENTIFIED
        elif self.n_frames >= self.max_frames:
            status = Status.UNKNOWN
        else:
            status = Status.OBSERVING
        self.decision = Decision(status, int(predicted), float(confidence), self.n_frames)
        return self.decision


def accumulate(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """(S, K, M) per-frame logits of S sequences -> (S, K, M) accumulated q_K(c)."""
    log_evidence = calibrated_log_probs(logits.double(), temperature).cumsum(dim=1)
    return torch.softmax(log_evidence, dim=-1)


@dataclass
class StoppingResult:
    predicted_class: torch.Tensor  # (S,) class at the stopping frame (at K_max if unknown)
    n_frames: torch.Tensor  # (S,) frames observed until the decision (K_max if unknown)
    unknown: torch.Tensor  # (S,) bool, threshold not reached within K_max frames


def apply_stopping_rule(q: torch.Tensor, threshold: float, max_frames: int) -> StoppingResult:
    """Stopping rule (3.23) on accumulated confidences q of shape (S, K >= max_frames, M)."""
    q = q[:, :max_frames]
    confidence, predicted = q.max(dim=-1)  # (S, K)
    reached = confidence >= threshold
    unknown = ~reached.any(dim=1)
    first = torch.where(unknown, max_frames - 1, reached.int().argmax(dim=1))
    rows = torch.arange(q.shape[0])
    return StoppingResult(predicted[rows, first], first + 1, unknown)

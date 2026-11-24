import math

import pytest
import torch

from rfea_core import (
    DecisionAccumulator,
    Status,
    accumulate,
    apply_stopping_rule,
    calibrated_log_probs,
    fit_temperature,
)


def test_fit_temperature_recovers_known_temperature() -> None:
    # Labels are sampled from softmax(z / 2), so the NLL-optimal temperature is 2.
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(50_000, 5, generator=gen) * 4
    y = torch.multinomial(torch.softmax(z / 2.0, dim=1), 1, generator=gen).squeeze(1)
    assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.05)


def test_temperature_keeps_predicted_class() -> None:
    z = torch.randn(100, 11)
    for t in (0.5, 1.0, 3.0):
        assert torch.equal(calibrated_log_probs(z, t).argmax(1), z.argmax(1))


def test_accumulated_class_does_not_depend_on_temperature() -> None:
    z = torch.randn(200, 10, 11)
    classes = [accumulate(z, t).argmax(-1) for t in (0.5, 1.0, 4.0)]
    assert torch.equal(classes[0], classes[1])
    assert torch.equal(classes[1], classes[2])


def test_streaming_accumulator_matches_vectorized() -> None:
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(300, 20, 11, generator=gen) * 2
    t, tau, k_max = 1.5, 0.95, 20
    batch = apply_stopping_rule(accumulate(z, t), tau, k_max)
    for s in range(z.shape[0]):
        acc = DecisionAccumulator(11, tau, k_max, t)
        for k in range(k_max):
            d = acc.update(z[s, k])
            if acc.done:
                break
        assert d.n_frames == batch.n_frames[s].item()
        assert d.predicted_class == batch.predicted_class[s].item()
        assert (d.status == Status.UNKNOWN) == batch.unknown[s].item()


def test_stopping_rule_cases() -> None:
    # Two classes; per-frame probabilities (0.8, 0.2) give q_K(0) = 4^K / (4^K + 1).
    z = torch.log(torch.tensor([0.8, 0.2])).expand(1, 5, 2)
    q = accumulate(z)
    assert q[0, 0, 0].item() == pytest.approx(0.8)
    assert q[0, 1, 0].item() == pytest.approx(16 / 17)
    stop = apply_stopping_rule(q, threshold=0.95, max_frames=5)  # 64/65 at K = 3
    assert (stop.n_frames.item(), stop.predicted_class.item(), stop.unknown.item()) == (3, 0, False)
    stop = apply_stopping_rule(q, threshold=0.999, max_frames=4)  # 256/257 < 0.999
    assert (stop.n_frames.item(), stop.unknown.item()) == (4, True)

    acc = DecisionAccumulator(2, threshold=0.95, max_frames=5)
    statuses = [acc.update(z[0, k]).status for k in range(3)]
    assert statuses == [Status.OBSERVING, Status.OBSERVING, Status.IDENTIFIED]
    assert acc.decision.confidence == pytest.approx(64 / 65)
    with pytest.raises(RuntimeError):
        acc.update(z[0, 3])


def test_conflicting_frames_end_unknown() -> None:
    # Frames alternately favor two classes: evidence never accumulates.
    a, b = math.log(0.9), math.log(0.1)
    frames = torch.tensor([[a, b], [b, a]] * 5)
    acc = DecisionAccumulator(2, threshold=0.95, max_frames=10)  # q_K is 0.9 or 0.5
    decisions = [acc.update(f) for f in frames]
    assert decisions[-1].status == Status.UNKNOWN
    assert decisions[-1].n_frames == 10

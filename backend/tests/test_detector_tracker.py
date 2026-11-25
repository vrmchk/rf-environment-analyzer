import numpy as np
import pytest

from rfea_backend.dsp.detector import DetectionSmoother, DetectorConfig, detect
from rfea_backend.dsp.tracker import Tracker, TrackerConfig, iou


def row_with(signals: list[tuple[int, int]], level: float = 20.0) -> np.ndarray:
    p = np.zeros(1024)
    for k1, k2 in signals:
        p[k1 : k2 + 1] = level
    return p


def test_detect_finds_intervals_bridges_gaps_and_drops_spikes() -> None:
    p = row_with([(100, 120), (123, 130), (500, 501), (800, 850)])
    noise, intervals = detect(p, DetectorConfig(gamma_db=6))
    assert noise == 0
    # 100-120 and 123-130 are 2 free bins apart -> one signal; 500-501 is a spike
    assert intervals == [(100, 130), (800, 850)]


def test_detect_respects_threshold() -> None:
    p = row_with([(100, 120)], level=5.0)
    assert detect(p, DetectorConfig(gamma_db=6))[1] == []
    assert detect(p, DetectorConfig(gamma_db=3))[1] == [(100, 120)]


def test_detect_drops_side_lobes_of_a_strong_neighbor() -> None:
    p = row_with([(200, 260)], level=30.0)
    p[240 + 30 : 240 + 36] = 12.0  # 10 bins past the main lobe, 18 dB weaker: side lobe
    p[600:620] = 12.0  # equally weak but far away: a signal of its own
    assert detect(p, DetectorConfig(gamma_db=6))[1] == [(200, 260), (600, 619)]


def test_smoothing_joins_a_flickering_signal() -> None:
    rng = np.random.default_rng(0)
    cfg = DetectorConfig(gamma_db=3, merge_gap=2, smooth_rows=4, smooth_bins=5)
    sm = DetectionSmoother(cfg)
    raw_pieces, smooth_pieces = [], []
    for _ in range(12):
        p = rng.normal(0, 0.6, 1024)  # noise in dB around 0
        p[300:340] = 3.5 + rng.normal(0, 2.5, 40)  # weak signal, bins dip below 3 dB
        raw = DetectorConfig(gamma_db=3, merge_gap=2, smooth_bins=1)
        raw_pieces.append(len(detect(p, raw)[1]))
        smooth_pieces.append(len(detect(sm(p), cfg)[1]))
    assert sum(n > 1 for n in raw_pieces) >= 8  # raw rows mostly break the signal up
    assert sum(n == 1 for n in smooth_pieces) >= 10  # smoothed rows keep it whole


def test_iou() -> None:
    assert iou((0, 9), (0, 9)) == 1
    assert iou((0, 9), (5, 14)) == pytest.approx(5 / 15)
    assert iou((0, 9), (20, 29)) == 0


def test_tracker_follows_a_signal_and_closes_it_after_the_gap() -> None:
    tr = Tracker(TrackerConfig(rho=0.5, max_gap=2, min_rows=3))
    tr.update(0, [(100, 120)])
    [t] = tr.active.values()
    assert not tr.confirmed(t)
    tr.update(1, [(101, 121)])
    tr.update(2, [(100, 119), (600, 640)])  # a second signal appears
    assert tr.confirmed(t) and t.rows_seen == 3
    assert len(tr.active) == 2
    closed = []
    for row in range(3, 7):
        closed += tr.update(row, [(600, 640)])
    assert [c.id for c in closed] == [t.id]  # missed rows 3, 4, 5 > max_gap = 2
    assert list(tr.active) == [2]


def test_tracker_opens_a_new_track_when_overlap_is_small() -> None:
    tr = Tracker(TrackerConfig(rho=0.5))
    tr.update(0, [(100, 120)])
    tr.update(1, [(115, 140)])  # IoU 6/41 < 0.5
    assert sorted(tr.active) == [1, 2]


def test_tracker_accumulates_power_for_bandwidth() -> None:
    tr = Tracker()
    p = row_with([(100, 120)])
    tr.update(0, [(100, 120)], p, 0.0)
    tr.update(1, [(100, 120)], p, 0.0)
    [t] = tr.active.values()
    assert t.noise_sum == pytest.approx(2.0)
    assert t.power_sum[110] == pytest.approx(200.0)

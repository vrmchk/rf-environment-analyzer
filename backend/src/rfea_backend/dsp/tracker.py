"""Tracking detected intervals between rows (thesis (3.11)).

An interval joins the active track with the largest frequency IoU, if that IoU >= rho
(one interval per track, best pairs first); the track takes the interval's bounds.
A track left without an interval is still continued, with its own bounds kept, when an
interval of the row contains it entirely: two neighboring signals whose intervals merge in
some rows (the detector bridges gaps of up to merge_gap bins) then keep their tracks
instead of one of them being closed and reopened as a second event. An interval that
matched no track opens a new track unless it contains an active track. A track that gets
no interval for more than `max_gap` rows is closed. Tracks seen in fewer than `min_rows`
rows are noise: they are not shown and do not go to the event log.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class TrackState(StrEnum):
    OBSERVING = "observing"
    IDENTIFIED = "identified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrackerConfig:
    rho: float = 0.5
    max_gap: int = 5  # rows (0.25 s at 20 rows/s)
    min_rows: int = 3


@dataclass
class Track:
    id: int
    k1: int  # current interval, inclusive bins
    k2: int
    first_row: int
    last_row: int
    rows_seen: int = 1
    missed: int = 0
    state: TrackState = TrackState.OBSERVING
    label: str | None = None  # class name once decided
    confidence: float | None = None
    n_frames: int = 0
    power_sum: np.ndarray | None = field(default=None, repr=False)  # linear, over rows
    noise_sum: float = 0.0  # linear noise level summed over the same rows


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.active: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self,
        row: int,
        intervals: list[tuple[int, int]],
        p_db: np.ndarray | None = None,
        noise_db: float | None = None,
    ) -> list[Track]:
        """Match the row's intervals; returns the tracks closed in this row."""
        pairs = sorted(
            ((iou((t.k1, t.k2), iv), tid, j) for tid, t in self.active.items()
             for j, iv in enumerate(intervals)),
            reverse=True,
        )  # fmt: skip
        matched_tracks: set[int] = set()
        matched_intervals: set[int] = set()
        for score, tid, j in pairs:
            if score < self.cfg.rho:
                break
            if tid in matched_tracks or j in matched_intervals:
                continue
            t = self.active[tid]
            t.k1, t.k2 = intervals[j]
            self._continue(t, row, p_db, noise_db)
            matched_tracks.add(tid)
            matched_intervals.add(j)
        holders: set[int] = set()  # intervals that contain a confirmed track
        for tid, t in self.active.items():
            if not self.confirmed(t):  # pieces of a signal still taking shape are not kept
                continue
            inside = [j for j, iv in enumerate(intervals) if contains(iv, (t.k1, t.k2))]
            holders.update(inside)
            if inside and tid not in matched_tracks:
                self._continue(t, row, p_db, noise_db)  # bounds kept
                matched_tracks.add(tid)
        for j, (k1, k2) in enumerate(intervals):
            if j not in matched_intervals and j not in holders:
                t = Track(self._next_id, k1, k2, row, row)
                self._accumulate(t, p_db, noise_db)
                self.active[t.id] = t
                self._next_id += 1
        closed = []
        for tid in list(self.active):
            t = self.active[tid]
            if tid not in matched_tracks and t.first_row != row:
                t.missed += 1
                if t.missed > self.cfg.max_gap:
                    closed.append(self.active.pop(tid))
        return closed

    def confirmed(self, t: Track) -> bool:
        """Seen in enough rows to be a signal rather than a noise spike."""
        return t.rows_seen >= self.cfg.min_rows

    def close_all(self) -> list[Track]:
        closed = list(self.active.values())
        self.active.clear()
        return closed

    def _continue(self, t: Track, row: int, p_db: np.ndarray | None, noise_db: float | None):
        t.last_row, t.missed = row, 0
        t.rows_seen += 1
        self._accumulate(t, p_db, noise_db)

    @staticmethod
    def _accumulate(t: Track, p_db: np.ndarray | None, noise_db: float | None) -> None:
        if p_db is None or noise_db is None:
            return
        p = 10 ** (p_db / 10)
        t.power_sum = p if t.power_sum is None else t.power_sum + p
        t.noise_sum += 10 ** (noise_db / 10)

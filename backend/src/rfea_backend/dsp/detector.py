"""Signal detection in a waterfall row (thesis 3.4, FR-7).

The noise level of a row is its median over frequency (3.9); a bin is occupied when it is
more than gamma dB above it (3.10). Occupied bins form intervals; gaps of up to
`merge_gap` free bins inside a signal are bridged and intervals narrower than `min_bins`
are dropped as noise spikes. Connectivity in time is the tracker's job (tracker.py).

Strong linearly modulated signals have spectral side lobes above the threshold next to the
main lobe. An interval whose peak is at least `sidelobe_db` below that of a neighbor no
farther away than the neighbor's own width is taken as its side lobe and dropped. The
price: a separate weak signal right next to a much stronger one is not detected.

Near the threshold the spectrum of a weak signal flickers from row to row and breaks into
pieces, each of which would become a track and an event. So detection runs on a smoothed
copy of the rows (DetectionSmoother): an exponential average over about `smooth_rows`
rows and a moving average over `smooth_bins` bins. The display and the bandwidth estimate
use the raw rows. The `edge_bins` outermost bins on each side are not searched: a real
receiver's anti-aliasing filter rolls off there, and spectral leakage of strong signals
near the edge produced false detections.

Defaults (gamma 3 dB, 5-bin frequency smoothing, gaps up to 8 bins bridged, no time
smoothing) were chosen by the FR-13 evaluation on scenes eval1 and eval2 and checked on
eval3 and eval4.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    gamma_db: float = 3.0  # chosen by FR-13 on scenes eval1, eval2 (3 vs 6 dB)
    merge_gap: int = 8  # free bins bridged inside one signal (~8 kHz)
    min_bins: int = 3  # ~3 kHz at 977 Hz per bin
    sidelobe_db: float = 10.0
    smooth_rows: float = 1.0  # time constant of the exponential average, rows (1 = none)
    smooth_bins: int = 5  # width of the moving average over frequency (1 = none)
    edge_bins: int = 8  # bins ignored at each band edge (anti-aliasing roll-off)


class DetectionSmoother:
    """Smoothed rows for detection; keeps the running average between rows."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._avg: np.ndarray | None = None

    def __call__(self, p_db: np.ndarray) -> np.ndarray:
        p = 10 ** (p_db / 10)
        a = 1 / max(self.cfg.smooth_rows, 1.0)
        self._avg = p if self._avg is None else (1 - a) * self._avg + a * p
        s = self._avg
        k = self.cfg.smooth_bins
        if k > 1:
            s = np.convolve(s, np.ones(k) / k, mode="same")
            s[: k // 2] = self._avg[: k // 2]  # no shrinking toward the band edges
            s[-(k // 2) :] = self._avg[-(k // 2) :]
        return 10 * np.log10(s)


def detect(p_db: np.ndarray, cfg: DetectorConfig) -> tuple[float, list[tuple[int, int]]]:
    """Noise level (dB) and occupied intervals [k1, k2] (inclusive bin indices)."""
    noise = float(np.median(p_db))
    mask = p_db > noise + cfg.gamma_db
    if cfg.edge_bins:
        mask[: cfg.edge_bins] = mask[-cfg.edge_bins :] = False
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    runs = list(zip(edges[::2], edges[1::2] - 1, strict=True))
    merged: list[list[int]] = []
    for k1, k2 in runs:
        if merged and k1 - merged[-1][1] - 1 <= cfg.merge_gap:
            merged[-1][1] = int(k2)
        else:
            merged.append([int(k1), int(k2)])
    found = [(a, b) for a, b in merged if b - a + 1 >= cfg.min_bins]
    peaks = [float(p_db[a : b + 1].max()) for a, b in found]
    return noise, [
        iv for j, iv in enumerate(found)
        if not any(
            peaks[i] - peaks[j] >= cfg.sidelobe_db
            and _gap(found[i], iv) <= found[i][1] - found[i][0] + 1
            for i in range(len(found)) if i != j
        )
    ]  # fmt: skip


def _gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(a[0], b[0]) - min(a[1], b[1]) - 1

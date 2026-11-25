"""Signal detection in a waterfall row (thesis 3.4, FR-7).

The noise level of a row is its median over frequency (3.9); a bin is occupied when it is
more than gamma dB above it (3.10). Occupied bins form intervals; gaps of up to
`merge_gap` free bins inside a signal are bridged and intervals narrower than `min_bins`
are dropped as noise spikes. Connectivity in time is the tracker's job (tracker.py).

Strong linearly modulated signals have spectral side lobes above the threshold next to the
main lobe. An interval whose peak is at least `sidelobe_db` below that of a neighbor no
farther away than the neighbor's own width is taken as its side lobe and dropped. The
price: a separate weak signal right next to a much stronger one is not detected.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    gamma_db: float = 6.0
    merge_gap: int = 2  # free bins bridged inside one signal
    min_bins: int = 3  # ~3 kHz at 977 Hz per bin
    sidelobe_db: float = 10.0


def detect(p_db: np.ndarray, cfg: DetectorConfig) -> tuple[float, list[tuple[int, int]]]:
    """Noise level (dB) and occupied intervals [k1, k2] (inclusive bin indices)."""
    noise = float(np.median(p_db))
    mask = p_db > noise + cfg.gamma_db
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

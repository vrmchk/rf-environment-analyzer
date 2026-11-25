"""IQ data source interface: the rest of the system does not depend on the source type."""

from typing import Protocol

import numpy as np


class IqSource(Protocol):
    sample_rate: float  # Hz
    center_freq: float  # Hz

    @property
    def position(self) -> int:
        """Samples delivered so far."""
        ...

    def read(self, n: int) -> np.ndarray | None:
        """Next n complex64 samples; fewer at the end; None when the source is exhausted."""
        ...

    def close(self) -> None: ...

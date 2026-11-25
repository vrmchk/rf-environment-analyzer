"""WebSocket messages (architecture.md, sections 3 and 11).

Client -> server, JSON: {"type": "start", "scene": str, "params": {...}}, {"type": "pause"},
{"type": "resume"}, {"type": "stop"}.
Server -> client, JSON: session_started, tracks, event, session_ended, error.
Server -> client, MessagePack: the waterfall row (levels as a bin field -> Uint8Array).
"""

from dataclasses import dataclass

import msgpack
import numpy as np


@dataclass(frozen=True)
class Row:
    row: int  # row number in the session
    t: float  # scene time at the end of the row, s
    f0_hz: float  # center frequency
    fs_hz: float  # sample rate = displayed span
    lo_db: float  # level of 0
    hi_db: float  # level of 255
    wall: float  # Unix time when the row's IQ block arrived (latency measurement, NFR-4)
    levels: np.ndarray  # (n,) uint8, frequency ascending

    def encode(self) -> bytes:
        return msgpack.packb(
            {
                "type": "row",
                "row": self.row,
                "t": self.t,
                "f0_hz": self.f0_hz,
                "fs_hz": self.fs_hz,
                "n": len(self.levels),
                "lo_db": self.lo_db,
                "hi_db": self.hi_db,
                "wall": self.wall,
                "levels": self.levels.astype(np.uint8).tobytes(),
            },
            use_bin_type=True,
        )


def decode(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


def error(message: str) -> dict:
    return {"type": "error", "message": message}

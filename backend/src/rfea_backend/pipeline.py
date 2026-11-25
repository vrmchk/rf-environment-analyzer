"""Monitoring session: source -> waterfall rows (-> detection -> tracking -> classification).

A session runs in its own thread, paced to real time: a block of the source "arrives" when
its last sample would have been received live. It lives on the server, not in a client
connection: messages go to `emit`, which the app fans out to whatever clients are
connected (NFR-8).
"""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum

from rfea_backend.api import protocol
from rfea_backend.api.protocol import Row
from rfea_backend.dsp.detector import DetectorConfig, detect
from rfea_backend.dsp.tracker import Track, Tracker, TrackerConfig
from rfea_backend.dsp.waterfall import Waterfall, WaterfallConfig
from rfea_backend.sources.base import IqSource


@dataclass(frozen=True)
class SessionParams:
    gamma_db: float = 6.0  # detection threshold over noise (FR-7)
    tau: float = 0.95  # confidence threshold of the stopping rule (FR-8)
    k_max: int = 20  # max frames per signal (FR-8)


class EndReason(StrEnum):
    FINISHED = "finished"
    STOPPED = "stopped"
    ERROR = "error"


Emit = Callable[[dict | bytes], None]


class Session:
    def __init__(
        self,
        source: IqSource,
        scene: str,
        params: SessionParams,
        emit: Emit,
        waterfall: WaterfallConfig | None = None,
        realtime: bool = True,
    ):
        self.id = str(uuid.uuid4())
        self.scene = scene
        self.params = params
        self.source = source
        self.emit = emit
        self.wf_cfg = waterfall or WaterfallConfig()
        self.realtime = realtime
        self.paused = False
        self.ended: EndReason | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread = threading.Thread(target=self._run, name=f"session-{self.id}", daemon=True)

    def started_message(self) -> dict:
        return {
            "type": "session_started",
            "session_id": self.id,
            "scene": self.scene,
            "fs_hz": self.source.sample_rate,
            "f0_hz": self.source.center_freq,
            "n": self.wf_cfg.n_fft,
            "rows_per_s": self.source.sample_rate / self.wf_cfg.row_samples,
            "params": asdict(self.params),
            "paused": self.paused,
        }

    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        self.paused = True
        self._resume.clear()

    def resume(self) -> None:
        self.paused = False
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        reason, message = EndReason.FINISHED, None
        try:
            self._loop()
            if self._stop.is_set():
                reason = EndReason.STOPPED
        except Exception as e:  # the session ends cleanly on a source or processing error
            reason, message = EndReason.ERROR, f"{type(e).__name__}: {e}"
        finally:
            self.source.close()
        self.ended = reason
        self.emit({"type": "session_ended", "session_id": self.id, "reason": str(reason),
                   "error": message})  # fmt: skip

    def _loop(self) -> None:
        fs, n = self.source.sample_rate, self.wf_cfg.row_samples
        waterfall = Waterfall(self.wf_cfg)
        detector = DetectorConfig(gamma_db=self.params.gamma_db)
        tracker = Tracker(TrackerConfig())
        t0 = time.monotonic()
        row = 0
        while not self._stop.is_set():
            if not self._resume.is_set():
                paused_at = time.monotonic()
                self._resume.wait()
                t0 += time.monotonic() - paused_at
                continue
            block = self.source.read(n)
            if block is None or len(block) < self.wf_cfg.n_fft:
                return
            end = self.source.position
            if self.realtime:
                delay = t0 + end / fs - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    return
            wall = time.time()
            p_db, levels, lo, hi = waterfall.row(block)
            t = end / fs
            self.emit(Row(row, t, self.source.center_freq, fs, lo, hi, wall, levels).encode())
            noise_db, intervals = detect(p_db, detector)
            tracker.update(row, intervals, p_db, noise_db)
            self.emit(protocol.tracks(row, t, [
                self._track_item(tr, fs) for tr in tracker.active.values()
                if tracker.confirmed(tr)
            ]))  # fmt: skip
            row += 1

    def _track_item(self, tr: Track, fs: float) -> dict:
        n, f0 = self.wf_cfg.n_fft, self.source.center_freq
        f_lo, f_hi = protocol.bins_to_hz(tr.k1, tr.k2, n, f0, fs)
        return {
            "id": tr.id,
            "f_lo_hz": f_lo,
            "f_hi_hz": f_hi,
            "t0": tr.first_row * self.wf_cfg.row_samples / fs,  # start of its first row
            "state": str(tr.state),
            "label": tr.label,
            "confidence": tr.confidence,
            "n_frames": tr.n_frames,
        }

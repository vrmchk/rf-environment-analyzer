"""Monitoring session: source -> waterfall rows -> detection -> tracking -> classification.

A session runs in its own thread, paced to real time: a block of the source "arrives" when
its last sample would have been received live. It lives on the server, not in a client
connection: messages go to `emit`, which the app fans out to whatever clients are
connected (NFR-8).

Classification of a confirmed track: every row, the narrowband signal of its frame gives up
to FRAMES_PER_ROW frames (channelizer.py); their logits go into the track's
DecisionAccumulator (tau, K_max of the session, T* of the model). When the stopping rule
decides (identified, or unknown after K_max frames) an event is emitted. A track lost
before a decision is logged as unknown with the frames it got.
"""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
import torch

from rfea_backend.api import protocol
from rfea_backend.api.protocol import Row
from rfea_backend.classify.model import Classifier
from rfea_backend.dsp import channelizer
from rfea_backend.dsp.channelizer import Band
from rfea_backend.dsp.detector import DetectorConfig, detect
from rfea_backend.dsp.tracker import Track, Tracker, TrackerConfig, TrackState
from rfea_backend.dsp.waterfall import Waterfall, WaterfallConfig
from rfea_backend.sources.base import IqSource
from rfea_core import DecisionAccumulator, Status

FRAMES_PER_ROW = 4


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


@dataclass
class _Signal:
    """Per-track classification state."""

    acc: DecisionAccumulator
    detected_at_s: float
    detected_wall: float
    band: Band | None = None
    snr_db: float | None = None
    logged: bool = False


def _iso(wall: float) -> str:
    return datetime.fromtimestamp(wall, UTC).isoformat()


class Session:
    def __init__(
        self,
        source: IqSource,
        scene: str,
        params: SessionParams,
        emit: Emit,
        classifier: Classifier | None = None,
        waterfall: WaterfallConfig | None = None,
        realtime: bool = True,
    ):
        self.id = str(uuid.uuid4())
        self.scene = scene
        self.params = params
        self.source = source
        self.emit = emit
        self.classifier = classifier
        self.wf_cfg = waterfall or WaterfallConfig()
        self.realtime = realtime
        self.paused = False
        self.ended: EndReason | None = None
        self.started_wall = time.time()
        self._signals: dict[int, _Signal] = {}
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
        row, t = 0, 0.0
        while not self._stop.is_set():
            if not self._resume.is_set():
                paused_at = time.monotonic()
                self._resume.wait()
                t0 += time.monotonic() - paused_at
                continue
            block = self.source.read(n)
            if block is None or len(block) < self.wf_cfg.n_fft:
                break
            end = self.source.position
            if self.realtime:
                delay = t0 + end / fs - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
            wall = time.time()
            p_db, levels, lo, hi = waterfall.row(block)
            t = end / fs
            self.emit(Row(row, t, self.source.center_freq, fs, lo, hi, wall, levels).encode())
            noise_db, intervals = detect(p_db, detector)
            closed = tracker.update(row, intervals, p_db, noise_db)
            for tr in tracker.active.values():
                if tr.id not in self._signals and self.classifier is not None:
                    self._signals[tr.id] = _Signal(
                        DecisionAccumulator(len(self.classifier.classes), self.params.tau,
                                            self.params.k_max, self.classifier.temperature),
                        detected_at_s=(end - len(block)) / fs, detected_wall=wall,
                    )  # fmt: skip
            confirmed = [tr for tr in tracker.active.values() if tracker.confirmed(tr)]
            if self.classifier is not None:
                self._classify(block, confirmed, fs, t, wall)
                for tr in closed:
                    if tracker.confirmed(tr):
                        self._lost(tr, t, wall)
                    self._signals.pop(tr.id, None)
            self.emit(protocol.tracks(row, t, [self._track_item(tr, fs) for tr in confirmed]))
            row += 1
        if self.classifier is not None:  # signals still on the air when the session ends
            wall = time.time()
            for tr in tracker.close_all():
                if tracker.confirmed(tr):
                    self._lost(tr, t, wall)

    def _classify(self, block: np.ndarray, tracks: list[Track], fs: float, t: float,
                  wall: float) -> None:  # fmt: skip
        clf = self.classifier
        assert clf is not None
        jobs = []
        for tr in tracks:
            sig = self._signals[tr.id]
            if tr.state != TrackState.OBSERVING:
                continue
            power = tr.power_sum / tr.rows_seen
            noise = tr.noise_sum / tr.rows_seen
            sig.band = channelizer.estimate_band(power, noise, tr.k1, tr.k2, fs,
                                                 clf.bandwidth.median)  # fmt: skip
            p_frame = float(power[tr.k1 : tr.k2 + 1].mean())
            sig.snr_db = float(10 * np.log10(max(p_frame - noise, 1e-12) / noise))
            first = sig.acc.n_frames == 0  # the first frame of a signal is skipped (exp04)
            need = self.params.k_max - sig.acc.n_frames
            f = channelizer.frames(block, sig.band, fs, FRAMES_PER_ROW + first)[first:][:need]
            if len(f):
                jobs.append((tr, sig, f))
        if not jobs:
            return
        logits = clf.logits(torch.from_numpy(np.concatenate([f for _, _, f in jobs])))
        i = 0
        for tr, sig, f in jobs:
            for k in range(len(f)):
                d = sig.acc.update(logits[i + k])
                tr.n_frames, tr.confidence = d.n_frames, d.confidence
                if d.status != Status.OBSERVING:
                    tr.state = TrackState(str(d.status))
                    tr.label = clf.classes[d.predicted_class] if d.status == Status.IDENTIFIED \
                        else None  # fmt: skip
                    self._log(tr, sig, t, wall)
                    break
            i += len(f)

    def _lost(self, tr: Track, t: float, wall: float) -> None:
        sig = self._signals.get(tr.id)
        if sig is None or sig.logged:
            return
        tr.state, tr.label = TrackState.UNKNOWN, None
        self._log(tr, sig, t, wall)

    def _log(self, tr: Track, sig: _Signal, t: float, wall: float) -> None:
        sig.logged = True
        fs, n, f0 = self.source.sample_rate, self.wf_cfg.n_fft, self.source.center_freq
        lo, hi = protocol.bins_to_hz(tr.k1, tr.k2, n, f0, fs)
        band = sig.band
        bw = self.classifier.bandwidth if self.classifier else None
        self.emit({
            "type": "event",
            "session_id": self.id,
            "track_id": tr.id,
            "status": str(tr.state),
            "label": tr.label,
            "confidence": tr.confidence,
            "n_frames": tr.n_frames,
            "detected_at_s": round(sig.detected_at_s, 3),
            "decided_at_s": round(t, 3),
            "observation_s": round(t - sig.detected_at_s, 3),
            "detected_at": _iso(sig.detected_wall),
            "decided_at": _iso(wall),
            "center_freq_hz": f0 + band.center_hz if band else (lo + hi) / 2,
            "bandwidth_hz": band.bandwidth_hz if band else hi - lo,
            "snr_db": sig.snr_db,
            "bandwidth_out_of_range": bool(band and bw and not bw.p05 <= band.fraction <= bw.p95),
        })  # fmt: skip

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

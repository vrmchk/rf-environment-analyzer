"""Run a session on a scene as fast as possible and write its event log as CSV.

The same Session, model check and CSV export as the live system (FR-12), only without
real-time pacing and with a throwaway SQLite log, so that FR-13 evaluation and the choice
of the detection threshold do not take a minute of playback per scene and setting.

Usage: uv run python -m rfea_backend.offline --scene eval1 --gamma 6 --out eval1_g6.csv
Env: RFEA_DATA_DIR or RFEA_SCENES_DIR, RFEA_MODEL_DIR (as for the server).
"""

import argparse
import hashlib
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from rfea_backend.classify.model import load_classifier
from rfea_backend.db.repo import Repo
from rfea_backend.dsp.detector import DetectorConfig
from rfea_backend.pipeline import Session, SessionParams
from rfea_backend.settings import Settings
from rfea_backend.sources.virtual import META_SUFFIX, VirtualSource


def run(settings: Settings, scene: str, params: SessionParams, out: Path,
        detector: DetectorConfig | None = None) -> dict:  # fmt: skip
    clf = load_classifier(settings.model_dir)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Repo(f"sqlite:///{(Path(tmp) / 'log.db').as_posix()}")
        repo.migrate()
        source = VirtualSource(settings.scenes_dir, scene)
        ended: dict = {}
        session = Session(source, scene, params, lambda m: None, clf, detector=detector,
                          realtime=False, on_event=repo.add_event)  # fmt: skip
        session.on_end = lambda reason, error: ended.update(reason=reason, error=error)
        meta = settings.scenes_dir / f"{scene}{META_SUFFIX}"
        repo.create_session(session.id, scene, hashlib.sha256(meta.read_bytes()).hexdigest(),
                            session.stored_params(), clf.describe(), datetime.now(UTC))  # fmt: skip
        t0 = time.perf_counter()
        session.start()
        session.join()
        repo.end_session(session.id, ended.get("reason", "error"), ended.get("error"))
        out.write_text(repo.events_csv(session.id) or "", encoding="utf-8")
        n = len(repo.list_events(session.id))
        repo.engine.dispose()
    return {"events": n, "seconds": round(time.perf_counter() - t0, 1), **ended}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--gamma", type=float, default=SessionParams.gamma_db)
    p.add_argument("--tau", type=float, default=SessionParams.tau)
    p.add_argument("--k-max", type=int, default=SessionParams.k_max)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smooth-rows", type=float, default=DetectorConfig.smooth_rows)
    p.add_argument("--smooth-bins", type=int, default=DetectorConfig.smooth_bins)
    p.add_argument("--merge-gap", type=int, default=DetectorConfig.merge_gap)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    det = DetectorConfig(smooth_rows=a.smooth_rows, smooth_bins=a.smooth_bins,
                         merge_gap=a.merge_gap)  # fmt: skip
    r = run(Settings.from_env(), a.scene, SessionParams(a.gamma, a.tau, a.k_max), a.out, det)
    print(f"{a.out}: {r}")


if __name__ == "__main__":
    main()

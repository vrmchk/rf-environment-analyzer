"""FR-13: compare a session's CSV export (FR-12) with the reference annotations of its scene.

Matching. An event has no end time (it is written when the stopping rule decides), so the
box IoU of thesis (3.25) cannot be computed from the log. An event is matched to an
annotated signal when
  - their frequency intervals overlap with IoU_f >= 0.5 (event: center +- measured
    bandwidth / 2; annotation: the edges holding 5-95 % of the power), and
  - the event was detected while the signal was on the air (from TIME_TOLERANCE_S before
    its start, one waterfall row of slack, to its end).
Pairs are taken greedily by IoU_f, one event per signal.

Metrics:
  detection: precision and recall of events vs annotated signals, recall by SNR group,
    detection delay (detected - signal start);
  classification, signals of the training classes that were detected: share identified
    correctly, wrong and unknown; accuracy of the decided ones; frames and time to the
    decision; by SNR group and by class; confusion matrix;
  signals outside the training classes: share unknown and what they were called.
SNR is the annotation's, i.e. the SNR the classifier sees after decimation (scenes.py).

Usage: uv run python -m rfea_ml.evaluate_session a.csv [b.csv ...] [--name pooled]
       [--scenes DIR] [--out DIR]
Outputs <out>/<csv stem>/metrics.json and summary.md (default out: <output_dir>/fr13).
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rfea_ml.paths import output_dir
from rfea_ml.radioml import CLASSES
from rfea_ml.scenes import scenes_dir

IOU_F_MIN = 0.5
TIME_TOLERANCE_S = 0.05
SNR_GROUPS = {"SNR >= 10": (10, 99), "0..10": (0, 10), "< 0": (-99, 0)}


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """Comment header (session, scene, sha256, params) and the event rows."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {}
    for line in lines:
        if line.startswith("# scene "):
            parts = line.split()
            header["scene"], header["scene_sha256"] = parts[2], parts[4]
        elif line.startswith("# session "):
            header["session"] = line.split()[2]
        elif line.startswith("# params "):
            header["params"] = dict(kv.split("=", 1) for kv in line[len("# params ") :].split())
    rows = list(csv.DictReader(x for x in lines if not x.startswith("#")))
    events = []
    for r in rows:
        events.append({
            "track_id": int(r["track_id"]),
            "detected_at_s": float(r["detected_at_s"]),
            "decided_at_s": float(r["decided_at_s"]),
            "center_freq_hz": float(r["center_freq_hz"]),
            "bandwidth_hz": float(r["bandwidth_hz"]),
            "status": r["status"],
            "label": r["label"] or None,
            "confidence": float(r["confidence"]) if r["confidence"] else None,
            "n_frames": int(r["n_frames"]),
            "observation_s": float(r["observation_s"]),
        })  # fmt: skip
    return header, events


def read_annotations(meta_path: Path) -> list[dict]:
    meta = json.loads(meta_path.read_text())
    fs = meta["global"]["core:sample_rate"]
    return [{
        "start_s": a["core:sample_start"] / fs,
        "end_s": (a["core:sample_start"] + a["core:sample_count"]) / fs,
        "f_lo": a["core:freq_lower_edge"],
        "f_hi": a["core:freq_upper_edge"],
        "label": a["core:label"],
        "known": a["core:label"] in CLASSES,
        "snr_db": a["rfea:snr_db"],
    } for a in meta["annotations"]]  # fmt: skip


def iou_f(e: dict, a: dict) -> float:
    half = e["bandwidth_hz"] / 2
    lo, hi = e["center_freq_hz"] - half, e["center_freq_hz"] + half
    inter = max(0.0, min(hi, a["f_hi"]) - max(lo, a["f_lo"]))
    union = (hi - lo) + (a["f_hi"] - a["f_lo"]) - inter
    return inter / union if union > 0 else 0.0


def match(events: list[dict], refs: list[dict]) -> dict[int, int]:
    """Event index -> annotation index."""
    pairs = sorted(  # best overlap first; on a tie, the earlier event
        ((-iou_f(e, a), e["detected_at_s"], i, j) for i, e in enumerate(events)
         for j, a in enumerate(refs)
         if a["start_s"] - TIME_TOLERANCE_S <= e["detected_at_s"] <= a["end_s"]),
    )  # fmt: skip
    out: dict[int, int] = {}
    used: set[int] = set()
    for neg_score, _, i, j in pairs:
        if -neg_score < IOU_F_MIN:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def group_of(snr: float) -> str:
    return next(g for g, (lo, hi) in SNR_GROUPS.items() if lo <= snr < hi)


def _share(n: int, d: int) -> float | None:
    return n / d if d else None


def evaluate(events: list[dict], refs: list[dict]) -> dict:
    m = match(events, refs)
    by_ref = {j: events[i] for i, j in m.items()}
    det: dict = {"events": len(events), "signals": len(refs), "matched": len(m),
                 "precision": _share(len(m), len(events)), "recall": _share(len(m), len(refs)),
                 "false_events": len(events) - len(m)}  # fmt: skip
    det["recall_by_group"] = {
        g: _share(sum(1 for j, a in enumerate(refs) if group_of(a["snr_db"]) == g and j in by_ref),
                  sum(1 for a in refs if group_of(a["snr_db"]) == g))
        for g in SNR_GROUPS
    }  # fmt: skip
    delays = [by_ref[j]["detected_at_s"] - refs[j]["start_s"] for j in by_ref]
    det["delay_s_median"] = round(float(np.median(delays)), 3) if delays else None

    def outcomes(idx: list[int]) -> dict:
        found = [j for j in idx if j in by_ref]
        res = Counter()
        frames, times = [], []
        for j in found:
            e = by_ref[j]
            if e["status"] == "unknown":
                res["unknown"] += 1
            else:
                res["correct" if e["label"] == refs[j]["label"] else "wrong"] += 1
                frames.append(e["n_frames"])
                times.append(e["observation_s"])
        decided = res["correct"] + res["wrong"]
        return {
            "signals": len(idx), "detected": len(found),
            "correct": _share(res["correct"], len(found)),
            "wrong": _share(res["wrong"], len(found)),
            "unknown": _share(res["unknown"], len(found)),
            "decided_accuracy": _share(res["correct"], decided),
            "frames_median": float(np.median(frames)) if frames else None,
            "time_s_median": float(np.median(times)) if times else None,
        }  # fmt: skip

    known = [j for j, a in enumerate(refs) if a["known"]]
    cls = {"all": outcomes(known)}
    cls |= {g: outcomes([j for j in known if group_of(refs[j]["snr_db"]) == g]) for g in SNR_GROUPS}
    per_class = {c: outcomes([j for j in known if refs[j]["label"] == c]) for c in CLASSES}
    confusion: dict[str, Counter] = defaultdict(Counter)
    for j, e in by_ref.items():
        confusion[refs[j]["label"]][e["label"] if e["status"] == "identified" else "unknown"] += 1
    ood = [j for j, a in enumerate(refs) if not a["known"]]
    ood_found = [j for j in ood if j in by_ref]
    return {
        "detection": det,
        "classification": cls,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "outside_training": {
            "signals": len(ood),
            "detected": len(ood_found),
            "unknown": _share(
                sum(by_ref[j]["status"] == "unknown" for j in ood_found), len(ood_found)
            ),
            "called": dict(Counter(by_ref[j]["label"] or "unknown" for j in ood_found)),
        },  # fmt: skip
    }


def summary(name: str, header: dict, r: dict) -> str:
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    d = r["detection"]
    lines = [
        f"# FR-13: {name}", "",
        f"Scene {header.get('scene')}, session {header.get('session')}, "
        f"params {header.get('params')}", "",
        "## Detection", "",
        f"Signals {d['signals']}, events {d['events']}, matched {d['matched']}, false events "
        f"{d['false_events']}; precision {pct(d['precision'])} %, recall {pct(d['recall'])} %; "
        f"median delay {d['delay_s_median']} s.", "",
        "| SNR | Recall, % |", "|---|---|",
        *(f"| {g} | {pct(v)} |" for g, v in d["recall_by_group"].items()), "",
        "## Classification of detected signals of the training classes", "",
        "| Group | Signals | Detected | Correct, % | Wrong, % | Unknown, % | Decided accuracy, % "
        "| Frames (median) | Time, s (median) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    rows = list(r["classification"].items()) + list(r["per_class"].items())
    for g, o in rows:
        lines.append(
            f"| {g} | {o['signals']} | {o['detected']} | {pct(o['correct'])} | {pct(o['wrong'])} "
            f"| {pct(o['unknown'])} | {pct(o['decided_accuracy'])} | {o['frames_median']} "
            f"| {o['time_s_median']} |"
        )
    ood = r["outside_training"]
    lines += ["", f"## Outside the training classes: {ood['detected']} of {ood['signals']} "
              f"detected, unknown {pct(ood['unknown'])} %, called {ood['called']}", "",
              "## Confusion (true -> decided)", ""]  # fmt: skip
    for t, row in sorted(r["confusion"].items()):
        lines.append(f"- {t}: " + ", ".join(f"{k} {v}" for k, v in sorted(row.items())))
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, nargs="+")
    p.add_argument("--scenes", type=Path, default=None, help="default: <data_dir>/scenes")
    p.add_argument("--out", type=Path, default=None, help="default: <output_dir>/fr13")
    p.add_argument("--name", default=None, help="also write pooled results under this name")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    root = a.out or output_dir() / "fr13"
    all_events: list[dict] = []
    all_refs: list[dict] = []
    for k, path in enumerate(a.csv):
        header, events = read_log(path)
        meta = (a.scenes or scenes_dir()) / f"{header['scene']}.sigmf-meta"
        if hashlib.sha256(meta.read_bytes()).hexdigest() != header["scene_sha256"]:
            raise SystemExit(f"{meta} differs from the scene of {path} (sha256)")
        refs = read_annotations(meta)
        _write(root / path.stem, path.stem, header, evaluate(events, refs))
        # pooled over the scenes: shift each scene in time so that they cannot match across
        shift = 1e6 * k
        all_events += [e | {"detected_at_s": e["detected_at_s"] + shift} for e in events]
        all_refs += [r | {"start_s": r["start_s"] + shift, "end_s": r["end_s"] + shift}
                     for r in refs]  # fmt: skip
    if a.name and len(a.csv) > 1:
        header = {"scene": ", ".join(p.stem for p in a.csv)}
        _write(root / a.name, a.name, header, evaluate(all_events, all_refs))


def _write(out: Path, name: str, header: dict, r: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"header": header} | r, indent=1))
    text = summary(name, header, r)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

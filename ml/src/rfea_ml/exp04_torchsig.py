"""Experiment 04: the classifier of the system on TorchSig and decision accumulation on
correlated frames of one signal.

Data: rfea_ml.torchsig_data records (4096 samples = 32 frames of 128, split by record).
Model: SpectrogramCNN with the approved STFT (N = 32, H = 8), trained from scratch with the
experiment 01 training code (Adam, early stopping with patience 8, max 60 epochs, seed 0).
Train, val and single-frame test use FRAMES_PER_RECORD random distinct frames of every
record (4 x 33 000 = 132 000 training frames, as many as RadioML train).

A. Single frame: accuracy vs SNR, confusion, T* on val frames, ECE (experiment 03 code).
B. Accumulation, the main comparison, on the same model and test records:
   - correlated: the first K_MAX = 20 consecutive frames of each test record
     (50 records per (class, SNR) pair, 11 000 sequences);
   - independent: K_MAX frames from K_MAX different test records of the same class and
     SNR (as in experiment 03), 50 sequences per pair.
   Metrics as in experiment 03 (accuracy of argmax Lambda_K, ECE of q_K, stopping rule).
C. Unknown signals: the first K_MAX frames of each OOD record (4 classes outside the
   training set, test only); share of "unknown" vs tau and what the decided ones were called.

Outputs in <output_dir>/exp04/: spec_n32/ (model.pt, history.json, logits.npz,
classifier.json for the backend), metrics.json, summary.md, plots.

Usage: uv run python -m rfea_ml.exp04_torchsig [--analyze-only]
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfea_core import (
    SpectrogramCNN,
    StftConfig,
    accumulate,
    apply_stopping_rule,
    log_spectrogram,
)
from rfea_ml import torchsig_data as tsd
from rfea_ml.exp03_accumulation import (
    K_MAX,
    K_VALUES,
    SNR_GROUPS,
    TAUS,
    accumulation_report,
    calibration_report,
    plot_stopping,
)
from rfea_ml.paths import git_commit, output_dir
from rfea_ml.plots import INK, plot_confusions, style_axes
from rfea_ml.train import TrainConfig, evaluate, fit, predict_logits

N_FFT = 32
FRAMES_PER_RECORD = 4
N_SEQUENCES = 50  # independent sequences per (class, SNR); equals test records per pair
SEED = 0
RUN = "spec_n32"


def exp_dir() -> Path:
    return output_dir() / "exp04"


def frame_starts(n_records: int, rng: np.random.Generator) -> np.ndarray:
    """(n, FRAMES_PER_RECORD) distinct random frame indices per record."""
    n_frames = tsd.RECORD_LEN // tsd.FRAME_LEN
    return np.stack(
        [rng.choice(n_frames, FRAMES_PER_RECORD, replace=False) for _ in range(n_records)]
    )


def sampled_frames(rec: tsd.Records, rng: np.random.Generator):
    """FRAMES_PER_RECORD frames of every record, flattened: x (n*F, 2, 128), y, snr."""
    starts = frame_starts(len(rec.x), rng)
    x = np.concatenate([tsd.frames(rec.x, starts[:, j])[:, 0] for j in range(FRAMES_PER_RECORD)])
    rep = partial(np.tile, reps=FRAMES_PER_RECORD)
    return x, rep(rec.y), rep(rec.snr)


def all_frames_logits(model, rec: tsd.Records, count: int, transform, device) -> np.ndarray:
    """(n, count, M) logits of the first `count` consecutive frames of every record."""
    out = []
    for i in range(0, len(rec.x), 2000):
        f = tsd.frames(rec.x[i : i + 2000], 0, count)  # (b, count, 2, 128)
        x = torch.from_numpy(f.reshape(-1, 2, tsd.FRAME_LEN)).to(device)
        out.append(predict_logits(model, x, transform).cpu().numpy().reshape(len(f), count, -1))
    return np.concatenate(out)


def train(device: torch.device) -> None:
    out = exp_dir() / RUN
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    parts = {p: sampled_frames(tsd.load(p), rng) for p in ("train", "val", "test")}
    stft = StftConfig(n_fft=N_FFT)
    transform = partial(log_spectrogram, cfg=stft)

    cfg = TrainConfig(seed=SEED)
    torch.manual_seed(cfg.seed)
    model = SpectrogramCNN(len(tsd.CLASSES)).to(device)
    dev = {
        p: (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        for p, (x, y, _) in parts.items()
    }
    t0 = time.perf_counter()
    history = fit(model, *dev["train"], *dev["val"], cfg, transform)
    seconds = time.perf_counter() - t0

    logits = {p: predict_logits(model, dev[p][0], transform).cpu().numpy() for p in ("val", "test")}
    model.eval()
    test_seq = all_frames_logits(
        model, tsd.load("test"), tsd.RECORD_LEN // tsd.FRAME_LEN, transform, device
    )
    ood_seq = all_frames_logits(model, tsd.load("ood_test"), K_MAX, transform, device)
    np.savez(
        out / "logits.npz",
        val_logits=logits["val"], val_y=parts["val"][1], val_snr=parts["val"][2],
        test_logits=logits["test"], test_y=parts["test"][1], test_snr=parts["test"][2],
        test_seq_logits=test_seq, ood_seq_logits=ood_seq,
    )  # fmt: skip
    torch.save(model.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "train.json").write_text(
        json.dumps(
            {
                "train_config": vars(cfg),
                "epochs": len(history),
                "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1),
                "train_seconds": seconds,
                "frames": {p: len(v[1]) for p, v in parts.items()},
                "commit": git_commit(),
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
            },
            indent=1,
        )
    )


def independent_sequences(y: np.ndarray, snr: np.ndarray, rng: np.random.Generator):
    """(S, K_MAX) record indices and (S, K_MAX) frame indices: frames of different records."""
    rec_idx, frame_idx = [], []
    n_frames = tsd.RECORD_LEN // tsd.FRAME_LEN
    for c in np.unique(y):
        for s in np.unique(snr):
            pool = np.flatnonzero((y == c) & (snr == s))
            for _ in range(N_SEQUENCES):
                rec_idx.append(rng.choice(pool, K_MAX, replace=False))
                frame_idx.append(rng.integers(0, n_frames, K_MAX))
    return np.stack(rec_idx), np.stack(frame_idx)


def ood_report(z: torch.Tensor, names: np.ndarray, snr: np.ndarray, t_star: float) -> dict:
    """Stopping rule on OOD sequences: share of "unknown" and the classes given instead."""
    q = accumulate(z, t_star)
    out = {}
    for tau in TAUS:
        r = apply_stopping_rule(q, tau, K_MAX)
        unknown, pred = r.unknown.numpy(), r.predicted_class.numpy()
        frames = r.n_frames.numpy()
        per = {}
        for name in np.unique(names):
            m = names == name
            decided = m & ~unknown
            counts = np.bincount(pred[decided], minlength=len(tsd.CLASSES))
            per[str(name)] = {
                "unknown_share": float(unknown[m].mean()),
                "unknown_by_group": {
                    g: float(unknown[m & (snr >= lo) & (snr <= hi)].mean())
                    for g, (lo, hi) in SNR_GROUPS.items()
                },
                "mean_frames_decided": float(frames[decided].mean()) if decided.any() else None,
                "decided_as": {
                    tsd.CLASSES[i]: int(counts[i]) for i in np.argsort(-counts)[:3] if counts[i] > 0
                },
            }
        out[str(tau)] = per
    return out


def analyze() -> dict:
    out = exp_dir() / RUN
    with np.load(out / "logits.npz") as d:
        val_z, val_y = torch.from_numpy(d["val_logits"]), torch.from_numpy(d["val_y"])
        test_z = torch.from_numpy(d["test_logits"]).double()
        test_y, test_snr = d["test_y"], d["test_snr"]
        seq_z = torch.from_numpy(d["test_seq_logits"]).double()  # (records, 32, M)
        ood_z = torch.from_numpy(d["ood_seq_logits"]).double()
    rec, ood = tsd.load("test"), tsd.load("ood_test")

    single = evaluate(test_z.numpy(), test_y, test_snr, tsd.CLASSES)
    t_star, calib = calibration_report(val_z, val_y, test_z, test_y, test_snr)
    temps = {"T=1": 1.0, "T*": t_star}

    correlated = accumulation_report(seq_z[:, :K_MAX], rec.y, rec.snr, temps)
    r_idx, f_idx = independent_sequences(rec.y, rec.snr, np.random.default_rng(SEED))
    indep_z = seq_z[torch.from_numpy(r_idx), torch.from_numpy(f_idx)]
    independent = accumulation_report(indep_z, rec.y[r_idx[:, 0]], rec.snr[r_idx[:, 0]], temps)

    train_info = json.loads((out / "train.json").read_text())
    classifier = {
        "classes": list(tsd.CLASSES),
        "temperature": t_star,
        "frame_len": tsd.FRAME_LEN,
        "stft": {"n_fft": N_FFT, "hop": N_FFT // 4, "window": "hann", "out_size": [64, 64]},
        "signal_bandwidth_fraction": list(tsd.GenConfig.bandwidth),
        "model": "SpectrogramCNN",
        "weights": "model.pt",
        "commit": train_info["commit"],
    }
    (out / "classifier.json").write_text(json.dumps(classifier, indent=1))
    return {
        "run": RUN,
        "temperature": t_star,
        "train": train_info,
        "single_frame": single,
        "calibration": calib,
        "correlated": {"run": "ts_corr", "temperature": t_star, "accumulation": correlated},
        "independent": {"run": "ts_indep", "temperature": t_star, "accumulation": independent},
        "ood": ood_report(ood_z, ood.tsig_class, ood.snr, t_star),
        "settings": {
            "frames_per_record": FRAMES_PER_RECORD,
            "k_max": K_MAX,
            "independent_sequences_per_pair": N_SEQUENCES,
            "seed": SEED,
            "data": json.loads((tsd.dataset_dir() / "config.json").read_text()),
        },
        "commit": git_commit(),
    }


def plot_accuracy_vs_snr(m: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for key, label, color, style in (
        ("independent", "independent frames", "#2a78d6", "--"),
        ("correlated", "consecutive frames of one signal", "#eb6834", "-"),
    ):
        for k, marker in ((1, None), (16, "o")):
            by_snr = m[key]["accumulation"]["accuracy_by_k"][str(k)]["by_snr"]
            if k == 1 and key == "correlated":
                continue  # K = 1 is the same single-frame curve for both
            ax.plot(
                [int(s) for s in by_snr], [100 * v for v in by_snr.values()],
                color=color if k == 16 else "#52514e", linestyle=style if k == 16 else ":",
                marker=marker, markersize=4, linewidth=2,
                label=f"K = 16, {label}" if k == 16 else "K = 1 (single frame)",
            )  # fmt: skip
    ax.set_xlabel("SNR, dB", color=INK)
    ax.set_ylabel("Accuracy, %", color=INK)
    ax.set_title("TorchSig: accuracy of argmax Λ_K", color=INK, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(-20, 20, 2))
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary(m: dict) -> str:
    groups = list(SNR_GROUPS)
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    sf = m["single_frame"]
    lines = [
        "# Experiment 04: TorchSig, correlated vs independent frames",
        "",
        f"Training: {m['train']['epochs']} epochs (best {m['train']['best_epoch']}), "
        f"{m['train']['train_seconds'] / 60:.1f} min, frames {m['train']['frames']}.",
        "",
        f"Single frame (test): all SNR {pct(sf['accuracy_overall'])}, SNR >= 0 "
        f"{pct(sf['accuracy_snr_ge_0'])}, SNR >= 10 {pct(sf['accuracy_snr_ge_10'])} %.",
        "",
        f"T* = {m['temperature']:.3f}; ECE T=1 -> T*: {pct(m['calibration']['T=1']['ece'])} -> "
        f"{pct(m['calibration']['T*']['ece'])} %; mean confidence "
        f"{pct(m['calibration']['T=1']['mean_confidence'])} -> "
        f"{pct(m['calibration']['T*']['mean_confidence'])} %.",
        "",
        "## Accuracy of argmax Λ_K, %",
        "",
        "| Frames | K | " + " | ".join(groups) + " |",
        "|---|---|" + "---|" * len(groups),
    ]
    for key in ("independent", "correlated"):
        acc = m[key]["accumulation"]["accuracy_by_k"]
        for k in K_VALUES:
            g = acc[str(k)]["by_group"]
            lines.append(f"| {key} | {k} | " + " | ".join(pct(g[x]) for x in groups) + " |")

    lines += ["", "## ECE of q_K with T*, % (mean q_K / accuracy, %)", "",
              "| Frames | K | All |", "|---|---|---|"]  # fmt: skip
    for key in ("independent", "correlated"):
        for k, e in m[key]["accumulation"]["T*"]["ece_qk"].items():
            lines.append(
                f"| {key} | {k} | {pct(e['all'])} "
                f"({pct(e['mean_confidence'])} / {pct(e['accuracy'])}) |"
            )

    lines += ["", "## Stopping rule, T*: decided accuracy, % / mean frames / unknown, %", "",
              "| Frames | τ | All | " + " | ".join(groups) + " |",
              "|---|---|---|" + "---|" * len(groups)]  # fmt: skip
    for key in ("independent", "correlated"):
        for tau in TAUS:
            st = m[key]["accumulation"]["T*"]["stopping"][str(tau)]
            cells = [st["all"], *(st["by_group"][g] for g in groups)]
            fmt = [
                f"{pct(c['decided_accuracy'])} / "
                f"{'—' if c['mean_frames_decided'] is None else f'{c["mean_frames_decided"]:.1f}'}"
                f" / {pct(c['unknown_share'])}"
                for c in cells
            ]
            lines.append(f"| {key} | {tau} | " + " | ".join(fmt) + " |")

    lines += ["", "## Unknown signals (OOD), T*: share of \"unknown\", % (decided as)", "",
              "| Class | τ | All | " + " | ".join(groups) + " | Decided as |",
              "|---|---|---|" + "---|" * (len(groups) + 1)]  # fmt: skip
    for tau, per in m["ood"].items():
        for name, r in per.items():
            ub = r["unknown_by_group"]
            dec = ", ".join(f"{c} {n}" for c, n in r["decided_as"].items())
            lines.append(
                f"| {name} | {tau} | {pct(r['unknown_share'])} | "
                + " | ".join(pct(ub[g]) for g in groups)
                + f" | {dec} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true", help="reuse saved logits")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if not args.analyze_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        train(device)
    torch.set_num_threads(4)
    m = analyze()
    exp = exp_dir()
    (exp / "metrics.json").write_text(json.dumps(m, indent=1))
    m = json.loads((exp / "metrics.json").read_text())  # string keys, as the plots expect
    plot_accuracy_vs_snr(m, exp / "accuracy_vs_k.png")
    plot_confusions({RUN: m["single_frame"]}, exp / "confusion_snr_ge_10.png")
    for key in ("correlated", "independent"):
        plot_stopping(m[key], exp / f"stopping_{key}.png")
    text = summary(m)
    (exp / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

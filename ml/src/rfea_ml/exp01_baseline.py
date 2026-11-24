"""Experiment 01: STFT window length for the spectrogram CNN vs. the O'Shea IQ CNN on
RadioML 2016.10A.

Runs:
  spec_n16, spec_n32, spec_n64 - SpectrogramCNN on STFT spectrograms (Hann window,
      hop = n_fft / 4, resized to 64x64), window length 16 / 32 / 64 samples;
  oshea_iq - OSheaCNN on raw IQ samples.

Each run uses the same stratified 60/20/20 split. Outputs go to ml/outputs/exp01/<run>/:
model.pt, history.json, metrics.json, logits.npz (val and test logits, kept for the
calibration and decision accumulation experiments), plus summary plots in ml/outputs/exp01/.

Usage: uv run python -m rfea_ml.exp01_baseline [--runs spec_n16 oshea_iq] [--plot-only]
"""

import argparse
import json
import sys
import time
from functools import partial

import matplotlib
import numpy as np
import torch

from rfea_core import SpectrogramCNN, StftConfig, log_spectrogram
from rfea_ml import radioml
from rfea_ml.baselines import OSheaCNN
from rfea_ml.paths import OUTPUTS_DIR, git_commit
from rfea_ml.train import TrainConfig, evaluate, fit, predict_logits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP_DIR = OUTPUTS_DIR / "exp01"
RUNS = ("spec_n16", "spec_n32", "spec_n64", "oshea_iq")
LABELS = {
    "spec_n16": "Spectrogram CNN, window 16",
    "spec_n32": "Spectrogram CNN, window 32",
    "spec_n64": "Spectrogram CNN, window 64",
    "oshea_iq": "IQ CNN (O'Shea et al.)",
}
# Categorical slots 1-4 of the validated reference palette (light mode),
# plus marker and line style so series are not distinguished by color alone.
STYLE = {
    "spec_n16": dict(color="#2a78d6", marker="o", linestyle="-"),
    "spec_n32": dict(color="#eb6834", marker="s", linestyle="-"),
    "spec_n64": dict(color="#1baf7a", marker="^", linestyle="-"),
    "oshea_iq": dict(color="#eda100", marker="D", linestyle="--"),
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def build(run: str, n_classes: int):
    if run == "oshea_iq":
        return OSheaCNN(n_classes), None, {"model": "OSheaCNN", "input": "iq 2x128"}
    n_fft = int(run.removeprefix("spec_n"))
    cfg = StftConfig(n_fft=n_fft)
    params = {
        "model": "SpectrogramCNN",
        "input": "stft",
        "n_fft": n_fft,
        "hop": cfg.hop,
        "window": "hann",
        "out_size": list(cfg.out_size),
        "normalization": "dB, per-example min-max to [0, 1]",
    }
    return SpectrogramCNN(n_classes), partial(log_spectrogram, cfg=cfg), params


def run_one(run: str, data: radioml.RadioML, split, device: torch.device, cfg: TrainConfig):
    train, val, test = split
    out = EXP_DIR / run
    out.mkdir(parents=True, exist_ok=True)

    x = torch.from_numpy(data.x).to(device)
    y = torch.from_numpy(data.y).to(device)
    model, transform, params = build(run, len(data.classes))
    model.to(device)
    transform = transform or (lambda t: t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run}] {params} params={n_params:,}", flush=True)

    t0 = time.perf_counter()
    history = fit(model, x[train], y[train], x[val], y[val], cfg, transform)
    train_seconds = time.perf_counter() - t0

    val_logits = predict_logits(model, x[val], transform).cpu().numpy()
    test_logits = predict_logits(model, x[test], transform).cpu().numpy()
    metrics = evaluate(test_logits, data.y[test], data.snr[test], data.classes)
    metrics.update(
        run=run,
        params=params,
        n_params=n_params,
        train_config=vars(cfg),
        epochs=len(history),
        best_epoch=int(np.argmin([h["val_loss"] for h in history]) + 1),
        train_seconds=train_seconds,
        commit=git_commit(),
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        torch=torch.__version__,
        split={"train": len(train), "val": len(val), "test": len(test), "seed": 0},
    )

    torch.save(model.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(
        out / "logits.npz",
        val_logits=val_logits, val_y=data.y[val], val_snr=data.snr[val],
        test_logits=test_logits, test_y=data.y[test], test_snr=data.snr[test],
    )  # fmt: skip
    print(
        f"[{run}] overall {metrics['accuracy_overall']:.4f}  SNR>=0 "
        f"{metrics['accuracy_snr_ge_0']:.4f}  SNR>=10 {metrics['accuracy_snr_ge_10']:.4f}  "
        f"{train_seconds / 60:.1f} min",
        flush=True,
    )


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_summary() -> None:
    metrics = {}
    for run in RUNS:
        path = EXP_DIR / run / "metrics.json"
        if path.exists():
            metrics[run] = json.loads(path.read_text())
    if not metrics:
        return

    # Accuracy vs SNR.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for run, m in metrics.items():
        snrs = [int(s) for s in m["accuracy_by_snr"]]
        acc = [100 * v for v in m["accuracy_by_snr"].values()]
        ax.plot(snrs, acc, label=LABELS[run], linewidth=2, markersize=5, **STYLE[run])
    ax.set_xlabel("SNR, dB", color=INK)
    ax.set_ylabel("Accuracy, %", color=INK)
    ax.set_title("RadioML 2016.10A: test accuracy vs SNR", color=INK, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(-20, 20, 2))
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "accuracy_vs_snr.png")
    plt.close(fig)

    # Confusion matrices at SNR >= 10 dB, row-normalized (recall per true class).
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), dpi=150, squeeze=False)
    for ax, (run, m) in zip(axes[0], metrics.items(), strict=True):
        cm = sum(np.array(c) for s, c in m["confusion_by_snr"].items() if int(s) >= 10)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        classes = m["classes"]
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7, color=INK_2)
        ax.set_yticks(range(len(classes)), classes, fontsize=7, color=INK_2)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if cm[i, j] > 0.5 else INK)  # fmt: skip
        ax.set_title(f"{LABELS[run]}\nSNR ≥ 10 dB", fontsize=9, color=INK)
        ax.set_xlabel("Predicted", fontsize=8, color=INK)
        ax.set_ylabel("True", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "confusion_snr_ge_10.png")
    plt.close(fig)

    # Summary table.
    rows = ["| Run | Params | Epochs | All SNR | SNR ≥ 0 | SNR ≥ 10 | Peak (SNR) | Train, min |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for run, m in metrics.items():
        peak_snr, peak = max(m["accuracy_by_snr"].items(), key=lambda kv: kv[1])
        rows.append(
            f"| {run} | {m['n_params']:,} | {m['epochs']} (best {m['best_epoch']}) | "
            f"{100 * m['accuracy_overall']:.1f} | {100 * m['accuracy_snr_ge_0']:.1f} | "
            f"{100 * m['accuracy_snr_ge_10']:.1f} | {100 * peak:.1f} ({peak_snr}) | "
            f"{m['train_seconds'] / 60:.1f} |"
        )
    (EXP_DIR / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        data = radioml.load()
        split = radioml.split_indices(data.y, data.snr, seed=0)
        cfg = TrainConfig(max_epochs=args.max_epochs)
        for run in args.runs:
            run_one(run, data, split, device, cfg)
    plot_summary()


if __name__ == "__main__":
    main()

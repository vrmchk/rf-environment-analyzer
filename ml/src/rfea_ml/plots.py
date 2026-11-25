"""Shared plots and summary tables for experiment metrics (format of train.evaluate)."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "spec_n16": "Spectrogram CNN, window 16",
    "spec_n32": "Spectrogram CNN, window 32",
    "spec_n64": "Spectrogram CNN, window 64",
    "oshea_iq": "IQ CNN (O'Shea et al.)",
    "svm": "SVM, 7 features",
}
# Categorical slots 1-5 of the validated reference palette (light mode), in fixed order,
# plus marker and line style so series are not distinguished by color alone.
STYLE = {
    "spec_n16": dict(color="#2a78d6", marker="o", linestyle="-"),
    "spec_n32": dict(color="#eb6834", marker="s", linestyle="-"),
    "spec_n64": dict(color="#1baf7a", marker="^", linestyle="-"),
    "oshea_iq": dict(color="#eda100", marker="D", linestyle="--"),
    "svm": dict(color="#e87ba4", marker="v", linestyle=":"),
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_metrics(paths: dict[str, Path]) -> dict[str, dict]:
    """{run: path to metrics.json} -> {run: metrics}, skipping missing files."""
    return {run: json.loads(p.read_text()) for run, p in paths.items() if p.exists()}


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy_vs_snr(metrics: dict[str, dict], path: Path) -> None:
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
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(metrics: dict[str, dict], path: Path, min_snr: int = 10) -> None:
    """Row-normalized confusion matrices (recall per true class) at SNR >= min_snr."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), dpi=150, squeeze=False)
    for ax, (run, m) in zip(axes[0], metrics.items(), strict=True):
        cm = sum(np.array(c) for s, c in m["confusion_by_snr"].items() if int(s) >= min_snr)
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
        ax.set_title(f"{LABELS[run]}\nSNR ≥ {min_snr} dB", fontsize=9, color=INK)
        ax.set_xlabel("Predicted", fontsize=8, color=INK)
        ax.set_ylabel("True", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_table(metrics: dict[str, dict]) -> str:
    rows = ["| Run | Size | Epochs | All SNR | SNR ≥ 0 | SNR ≥ 10 | Peak (SNR) | Train, min |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for run, m in metrics.items():
        peak_snr, peak = max(m["accuracy_by_snr"].items(), key=lambda kv: kv[1])
        size = f"{m['n_params']:,} params" if "n_params" in m else f"{m['n_support']:,} SVs"
        epochs = f"{m['epochs']} (best {m['best_epoch']})" if "epochs" in m else "—"
        rows.append(
            f"| {run} | {size} | {epochs} | "
            f"{100 * m['accuracy_overall']:.1f} | {100 * m['accuracy_snr_ge_0']:.1f} | "
            f"{100 * m['accuracy_snr_ge_10']:.1f} | {100 * peak:.1f} ({peak_snr}) | "
            f"{m['train_seconds'] / 60:.1f} |"
        )
    return "\n".join(rows) + "\n"

"""Experiment 01, repeated runs: mean ± standard deviation over training seeds.

Reads <output_dir>/exp01/<run>/metrics.json (seed 0) and exp01/seed<N>/<run>/metrics.json
(see exp01_baseline) and writes to <output_dir>/exp01/seeds/:
  summary.md - accuracy (all SNR, SNR >= 0, SNR >= 10, each SNR) as mean ± std, the
      per-seed values and the differences between the runs;
  summary.json - the same numbers;
  accuracy_vs_snr.png - mean accuracy vs SNR with a ±1 std band.

The standard deviation is the sample one (ddof = 1). Only the training seed differs
between repeats; the split is always seed 0.

Usage: uv run python -m rfea_ml.exp01_seeds [--seed 0 1 2]
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from rfea_ml.exp01_baseline import RUNS, exp_dir
from rfea_ml.paths import output_dir
from rfea_ml.plots import INK, LABELS, STYLE, load_metrics, style_axes

TOTALS = {
    "accuracy_overall": "All SNR",
    "accuracy_snr_ge_0": "SNR ≥ 0",
    "accuracy_snr_ge_10": "SNR ≥ 10",
}
REFERENCE = "spec_n32"  # approved model; the other runs are compared with it


def load(seeds: list[int]) -> dict[str, list[dict]]:
    """{run: [metrics for each seed]}; fails if any run/seed is missing."""
    paths = {(run, s): exp_dir(s) / run / "metrics.json" for run in RUNS for s in seeds}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("missing results:\n  " + "\n  ".join(missing))
    loaded = load_metrics({f"{run}/{s}": p for (run, s), p in paths.items()})
    return {run: [loaded[f"{run}/{s}"] for s in seeds] for run in RUNS}


def aggregate(metrics: dict[str, list[dict]], seeds: list[int]) -> dict:
    out: dict = {"seeds": seeds, "std": "sample (ddof=1)", "runs": {}}
    for run, ms in metrics.items():
        snrs = list(ms[0]["accuracy_by_snr"])
        by_snr = np.array([[m["accuracy_by_snr"][s] for s in snrs] for m in ms])
        out["runs"][run] = {
            "commits": [m["commit"] for m in ms],
            "best_epochs": [m["best_epoch"] for m in ms],
            "train_minutes": [round(m["train_seconds"] / 60, 1) for m in ms],
            **{k: _stats([m[k] for m in ms]) for k in TOTALS},
            "accuracy_by_snr": {s: _stats(by_snr[:, i]) for i, s in enumerate(snrs)},
        }
    out["differences_vs_" + REFERENCE] = {
        run: {
            k: _stats([a[k] - b[k] for a, b in zip(metrics[REFERENCE], ms, strict=True)])
            for k in TOTALS
        }
        for run, ms in metrics.items()
        if run != REFERENCE
    }
    return out


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}


def _ms(s: dict, signed: bool = False) -> str:
    mean = f"{100 * s['mean']:+.1f}" if signed else f"{100 * s['mean']:.1f}"
    return f"{mean} ± {100 * s['std']:.1f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(map(str, r)) + " |" for r in rows),
    ]


def summary(agg: dict) -> str:
    seeds, runs = agg["seeds"], agg["runs"]
    totals = list(TOTALS.values())
    lines = [
        f"# Experiment 01: {len(seeds)} training seeds ({', '.join(map(str, seeds))})",
        "",
        "Accuracy on test, %, mean ± sample std over seeds; split fixed (seed 0).",
        "",
        *_table(
            ["Run", *totals, "Best epoch per seed"],
            [
                [run, *(_ms(r[k]) for k in TOTALS), ", ".join(map(str, r["best_epochs"]))]
                for run, r in runs.items()
            ],
        ),
        "",
        "Per seed, %:",
        "",
        *_table(
            ["Run", "Seed", *totals, "Commit"],
            [
                [run, s, *(f"{100 * r[k]['values'][i]:.1f}" for k in TOTALS), r["commits"][i]]
                for run, r in runs.items()
                for i, s in enumerate(seeds)
            ],
        ),
        "",
        f"Difference {REFERENCE} − run (same seed), percentage points, mean ± std:",
        "",
        *_table(
            ["Run", *totals],
            [
                [run, *(_ms(d[k], signed=True) for k in TOTALS)]
                for run, d in agg["differences_vs_" + REFERENCE].items()
            ],
        ),
        "",
        "Accuracy by SNR, %, mean ± std:",
        "",
        *_table(
            ["SNR, dB", *runs],
            [
                [s, *(_ms(r["accuracy_by_snr"][s]) for r in runs.values())]
                for s in next(iter(runs.values()))["accuracy_by_snr"]
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def plot(agg: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for run, r in agg["runs"].items():
        snrs = [int(s) for s in r["accuracy_by_snr"]]
        mean = np.array([100 * v["mean"] for v in r["accuracy_by_snr"].values()])
        std = np.array([100 * v["std"] for v in r["accuracy_by_snr"].values()])
        style = STYLE[run]
        ax.fill_between(snrs, mean - std, mean + std, color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(snrs, mean, label=LABELS[run], linewidth=2, markersize=5, **style)
    n = len(agg["seeds"])
    ax.set_xlabel("SNR, dB", color=INK)
    ax.set_ylabel("Accuracy, %", color=INK)
    ax.set_title(f"RadioML 2016.10A: test accuracy vs SNR, mean ± std over {n} seeds",
                 color=INK, loc="left")  # fmt: skip
    ax.set_ylim(0, 100)
    ax.set_xticks(range(-20, 20, 2))
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    agg = aggregate(load(args.seed), args.seed)
    out = output_dir() / "exp01" / "seeds"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(agg, indent=1))
    text = summary(agg)
    (out / "summary.md").write_text(text, encoding="utf-8")
    plot(agg, out / "accuracy_vs_snr.png")
    print(text)


if __name__ == "__main__":
    main()

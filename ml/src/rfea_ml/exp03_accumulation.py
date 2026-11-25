"""Experiment 03: temperature scaling and decision accumulation on RadioML 2016.10A.

No training: uses the saved val/test logits of experiment 01 (seed 0) for the spectrogram
CNN (spec_n32, main model) and the O'Shea IQ CNN (oshea_iq).

A. Calibration (thesis 3.6): T* fitted on val logits (all SNR); NLL and ECE (15 equal bins)
   on test before (T = 1) and after (T*); reliability diagrams; ECE per SNR.
B. Accumulation (thesis 3.7): RadioML has no consecutive frames of one signal, so the
   "frames" of a signal are different test examples of the same class and SNR: for each
   (class, SNR) pair, 100 sequences of K_max = 20 examples, without repeats inside a
   sequence (seed 0). The examples are independent, so the frame independence assumption
   holds exactly: this is the optimistic case.
   - accuracy of argmax Lambda_K for K = 1, 2, 4, 8, 16 (K = 1 is single-frame);
   - ECE of the accumulated confidence q_K for K = 1, 4, 16 with T = 1 and T*;
   - stopping rule with K_max = 20, tau in {0.9, 0.95, 0.99}, T = 1 and T*: accuracy of
     decided signals, mean frames until the decision, share of "unknown".

Outputs in <output_dir>/exp03/: <run>/metrics.json, plots, summary.md.

Usage: uv run python -m rfea_ml.exp03_accumulation [--plot-only]
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import torch

from rfea_core import accumulate, apply_stopping_rule, calibrated_log_probs, fit_temperature
from rfea_ml.paths import git_commit, output_dir
from rfea_ml.plots import INK, INK_2, LABELS, style_axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = ("spec_n32", "oshea_iq")
K_VALUES = (1, 2, 4, 8, 16)
ECE_K = (1, 4, 16)
TAUS = (0.9, 0.95, 0.99)
K_MAX = 20
N_SEQUENCES = 100
N_BINS = 15
SEED = 0
SNR_GROUPS = {
    "SNR >= 10": (10, 18),
    "0..8": (0, 8),
    "-10..-2": (-10, -2),
    "<= -12": (-20, -12),
}
# Categorical slots 1-3 (fixed order) for tau / temperature series; blue ramp for K.
CAT = ("#2a78d6", "#eb6834", "#1baf7a")
K_RAMP = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281")


def exp_dir() -> Path:
    return output_dir() / "exp03"


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error (3.20) with equal-width confidence bins."""
    return float(
        sum(
            b["count"] * abs(b["acc"] - b["conf"]) for b in reliability(confidence, correct, n_bins)
        )
        / len(confidence)
    )


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Per bin: mean confidence, accuracy, count (non-empty bins only)."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    "bin": b,
                    "conf": float(confidence[m].mean()),
                    "acc": float(correct[m].mean()),
                    "count": int(m.sum()),
                }
            )
    return out


def make_sequences(y: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """(S, K_MAX) test indices: N_SEQUENCES sequences per (class, SNR) pair."""
    rng = np.random.default_rng(SEED)
    seqs = []
    for c in np.unique(y):
        for s in np.unique(snr):
            pool = np.flatnonzero((y == c) & (snr == s))
            seqs.extend(rng.choice(pool, K_MAX, replace=False) for _ in range(N_SEQUENCES))
    return np.stack(seqs)


def _by_snr(values: np.ndarray, snr: np.ndarray) -> dict[int, float]:
    return {int(s): float(values[snr == s].mean()) for s in np.unique(snr)}


def _by_group(values: np.ndarray, snr: np.ndarray) -> dict[str, float]:
    return {
        g: float(values[(snr >= lo) & (snr <= hi)].mean()) for g, (lo, hi) in SNR_GROUPS.items()
    }


def _stopping_stats(
    mask: np.ndarray, decided: np.ndarray, ok: np.ndarray, frames: np.ndarray
) -> dict:
    """Stopping rule outcome over the sequences selected by mask."""
    dm = mask & decided
    return {
        "decided_accuracy": float(ok[dm].mean()) if dm.any() else None,
        "mean_frames_decided": float(frames[dm].mean()) if dm.any() else None,
        "mean_frames_all": float(frames[mask].mean()),
        "unknown_share": float(1 - decided[mask].mean()),
        "wrong_share": float((dm & ~ok)[mask].mean()),
    }


def fit_evidence_scale(z: torch.Tensor, y: np.ndarray, temperature: float) -> float:
    """beta minimizing NLL of softmax(beta * Lambda_K) over all K of (S, K, M) sequences."""
    log_evidence = calibrated_log_probs(z, temperature).cumsum(dim=1)
    target = torch.from_numpy(y)[:, None].expand(-1, z.shape[1]).reshape(-1)
    log_beta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_beta], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        scaled = (log_evidence * log_beta.exp()).reshape(-1, z.shape[-1])
        loss = torch.nn.functional.cross_entropy(scaled, target)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_beta.detach().exp())


def calibration_report(
    val_z: torch.Tensor, val_y: torch.Tensor, test_z: torch.Tensor, test_y, test_snr
) -> tuple[float, dict]:
    """T* on val logits; NLL, ECE, reliability on test for T = 1 and T* (thesis 3.6)."""
    t_star = fit_temperature(val_z, val_y)
    calib = {}
    for name, t in {"T=1": 1.0, "T*": t_star}.items():
        log_p = calibrated_log_probs(test_z, t)
        conf, pred = log_p.exp().max(1)
        conf, correct = conf.numpy(), (pred.numpy() == test_y)
        calib[name] = {
            "temperature": t,
            "nll": float(-log_p[torch.arange(len(test_y)), test_y].mean()),
            "ece": ece(conf, correct),
            "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()),
            "ece_by_snr": {
                int(s): ece(conf[test_snr == s], correct[test_snr == s])
                for s in np.unique(test_snr)
            },
            "reliability": reliability(conf, correct),
            "by_group": {
                g: {
                    "mean_confidence": float(conf[m].mean()),
                    "accuracy": float(correct[m].mean()),
                    "ece": ece(conf[m], correct[m]),
                }
                for g, (lo, hi) in SNR_GROUPS.items()
                for m in [(test_snr >= lo) & (test_snr <= hi)]
            },
        }
    return t_star, calib


def accumulation_report(
    z: torch.Tensor, seq_y: np.ndarray, seq_snr: np.ndarray, temps: dict[str, float]
) -> dict:
    """Accuracy of argmax Lambda_K, ECE of q_K and the stopping rule (thesis 3.7).

    z: (S, K_MAX, M) logits of S frame sequences; seq_y, seq_snr: class and SNR per sequence.
    """
    accum = {}
    for name, t in temps.items():
        q = accumulate(z, t)
        conf, pred = q.max(dim=-1)
        conf, correct = conf.numpy(), pred.numpy() == seq_y[:, None]
        res = {
            "ece_qk": {
                k: {
                    "all": ece(conf[:, k - 1], correct[:, k - 1]),
                    "mean_confidence": float(conf[:, k - 1].mean()),
                    "accuracy": float(correct[:, k - 1].mean()),
                    "by_group": {
                        g: ece(conf[m, k - 1], correct[m, k - 1])
                        for g, (lo, hi) in SNR_GROUPS.items()
                        for m in [(seq_snr >= lo) & (seq_snr <= hi)]
                    },
                }
                for k in ECE_K
            },
            "stopping": {},
        }
        if "accuracy_by_k" not in accum:  # argmax Lambda_K does not depend on T
            accum["accuracy_by_k"] = {
                k: {
                    "by_snr": _by_snr(correct[:, k - 1], seq_snr),
                    "by_group": _by_group(correct[:, k - 1], seq_snr),
                }
                for k in K_VALUES
            }
        for tau in TAUS:
            r = apply_stopping_rule(q, tau, K_MAX)
            decided = ~r.unknown.numpy()
            ok = r.predicted_class.numpy() == seq_y
            frames = r.n_frames.numpy().astype(float)
            stats = partial(_stopping_stats, decided=decided, ok=ok, frames=frames)

            res["stopping"][str(tau)] = {
                "all": stats(np.ones_like(decided)),
                "by_group": {
                    g: stats((seq_snr >= lo) & (seq_snr <= hi))
                    for g, (lo, hi) in SNR_GROUPS.items()
                },
                "by_snr": {int(s): stats(seq_snr == s) for s in np.unique(seq_snr)},
            }
        accum[name] = res
    return accum


def analyze(run: str) -> dict:
    with np.load(output_dir() / "exp01" / run / "logits.npz") as d:
        val_z, val_y = torch.from_numpy(d["val_logits"]), torch.from_numpy(d["val_y"])
        test_z = torch.from_numpy(d["test_logits"]).double()
        test_y, test_snr = d["test_y"], d["test_snr"]
        val_snr = d["val_snr"]

    # A. Calibration.
    t_star, calib = calibration_report(val_z, val_y, test_z, test_y, test_snr)

    # B. Accumulation over pseudo-sequences.
    seq = make_sequences(test_y, test_snr)
    seq_y, seq_snr = test_y[seq[:, 0]], test_snr[seq[:, 0]]
    z = test_z[torch.from_numpy(seq)]  # (S, K_MAX, M)
    accum = accumulation_report(z, seq_y, seq_snr, {"T=1": 1.0, "T*": t_star})

    # Diagnostic (not part of the method): one factor beta on the accumulated evidence,
    # q_K = softmax(beta * Lambda_K), fitted on val pseudo-sequences.
    val_seq = make_sequences(val_y.numpy(), val_snr)
    beta = fit_evidence_scale(
        val_z.double()[torch.from_numpy(val_seq)], val_y.numpy()[val_seq[:, 0]], t_star
    )
    q_beta = torch.softmax(beta * calibrated_log_probs(z, t_star).cumsum(dim=1), dim=-1)
    conf_b, pred_b = q_beta.max(dim=-1)
    conf_b, correct_b = conf_b.numpy(), pred_b.numpy() == seq_y[:, None]
    diagnostic = {
        "beta": beta,
        "ece_qk": {
            k: {
                "all": ece(conf_b[:, k - 1], correct_b[:, k - 1]),
                "by_group": {
                    g: {
                        "ece": ece(conf_b[m, k - 1], correct_b[m, k - 1]),
                        "mean_confidence": float(conf_b[m, k - 1].mean()),
                        "accuracy": float(correct_b[m, k - 1].mean()),
                    }
                    for g, (lo, hi) in SNR_GROUPS.items()
                    for m in [(seq_snr >= lo) & (seq_snr <= hi)]
                },
            }
            for k in ECE_K
        },
    }

    return {
        "run": run,
        "logits": f"exp01/{run}/logits.npz (seed 0)",
        "temperature": t_star,
        "calibration": calib,
        "accumulation": accum,
        "diagnostic_evidence_scale": diagnostic,
        "settings": {
            "k_values": K_VALUES,
            "ece_k": ECE_K,
            "taus": TAUS,
            "k_max": K_MAX,
            "n_sequences_per_pair": N_SEQUENCES,
            "n_sequences": len(seq),
            "ece_bins": N_BINS,
            "seed": SEED,
            "snr_groups": SNR_GROUPS,
        },
        "commit": git_commit(),
    }


def plot_reliability(metrics: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.6), dpi=150)
    for ax, (run, m) in zip(np.atleast_1d(axes), metrics.items(), strict=True):
        ax.plot(
            [0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=":", label="Perfect calibration"
        )
        for (name, c), color, marker in zip(
            m["calibration"].items(), CAT, ("o", "s"), strict=False
        ):
            bins = c["reliability"]
            label = (
                f"T = 1, ECE {100 * c['ece']:.1f}%"
                if name == "T=1"
                else (f"T* = {c['temperature']:.2f}, ECE {100 * c['ece']:.1f}%")
            )
            ax.plot(
                [b["conf"] for b in bins],
                [b["acc"] for b in bins],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=label,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Confidence", color=INK)
        ax.set_ylabel("Accuracy", color=INK)
        ax.set_title(f"{LABELS[run]}: reliability (test)", color=INK, fontsize=10, loc="left")
        style_axes(ax)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_accuracy_vs_k(metrics: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.8), dpi=150)
    for ax, (run, m) in zip(np.atleast_1d(axes), metrics.items(), strict=True):
        for k, color in zip(K_VALUES, K_RAMP, strict=True):
            by_snr = m["accumulation"]["accuracy_by_k"][str(k)]["by_snr"]
            label = "K = 1 (single frame)" if k == 1 else f"K = {k}"
            ax.plot(
                [int(s) for s in by_snr],
                [100 * v for v in by_snr.values()],
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=label,
            )
        ax.set_xlabel("SNR, dB", color=INK)
        ax.set_ylabel("Accuracy, %", color=INK)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(-20, 20, 4))
        ax.set_title(
            f"{LABELS[run]}: accumulated over K frames", color=INK, fontsize=10, loc="left"
        )
        style_axes(ax)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stopping(m: dict, path: Path) -> None:
    panels = (
        ("decided_accuracy", "Accuracy of decided signals, %", 100),
        ("mean_frames_decided", "Mean frames until decision", 1),
        ("unknown_share", 'Share of "unknown", %', 100),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    for ax, (key, ylabel, scale) in zip(axes, panels, strict=True):
        for tau, color in zip(TAUS, CAT, strict=True):
            for name, style in (("T*", "-"), ("T=1", "--")):
                by_snr = m["accumulation"][name]["stopping"][str(tau)]["by_snr"]
                pts = [(int(s), v[key]) for s, v in by_snr.items() if v[key] is not None]
                t_label = f"T* = {m['temperature']:.2f}" if name == "T*" else "T = 1"
                ax.plot(
                    [p[0] for p in pts],
                    [scale * p[1] for p in pts],
                    color=color,
                    linestyle=style,
                    marker="o" if name == "T*" else None,
                    markersize=3,
                    linewidth=2 if name == "T*" else 1.5,
                    label=f"τ = {tau}, {t_label}",
                )
        ax.set_xlabel("SNR, dB", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_xticks(range(-20, 20, 4))
        if key != "mean_frames_decided":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, K_MAX)
        style_axes(ax)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=7, loc="lower right")
    fig.suptitle(
        f"{LABELS[m['run']]}: stopping rule, K_max = {K_MAX}",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}"


def _slash(d: dict, *keys: str) -> str:
    """'a / b / c' of percentages d[key] for the given keys."""
    return " / ".join(_pct(d[k]) for k in keys)


def _table(title: str, header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["", f"## {title}", "", "| " + " | ".join(header) + " |"]
    out.append("|" + "---|" * len(header))
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def summary(metrics: dict[str, dict]) -> str:
    groups = list(SNR_GROUPS)
    lines: list[str] = []

    header = [
        "Run",
        "T*",
        "NLL T=1",
        "NLL T*",
        "ECE T=1, %",
        "ECE T*, %",
        "Mean conf T=1, %",
        "Mean conf T*, %",
        "Accuracy, %",
    ]
    rows = []
    for run, m in metrics.items():
        a, b = m["calibration"]["T=1"], m["calibration"]["T*"]
        rows.append(
            [
                run,
                f"{m['temperature']:.3f}",
                f"{a['nll']:.4f}",
                f"{b['nll']:.4f}",
                _pct(a["ece"]),
                _pct(b["ece"]),
                _pct(a["mean_confidence"]),
                _pct(b["mean_confidence"]),
                _pct(a["accuracy"]),
            ]
        )
    lines += _table("Calibration, single frame (test)", header, rows)

    rows = []
    for run, m in metrics.items():
        bg = m["calibration"]["T*"]["by_group"]
        rows.append([run, *(_slash(bg[g], "mean_confidence", "accuracy", "ece") for g in groups)])
    lines += _table(
        "Single frame with T*: mean confidence / accuracy / ECE, %", ["Run", *groups], rows
    )

    rows = []
    for run, m in metrics.items():
        for k in K_VALUES:
            g = m["accumulation"]["accuracy_by_k"][str(k)]["by_group"]
            rows.append([run, str(k), *(_pct(g[x]) for x in groups)])
    lines += _table("Accuracy of argmax Lambda_K, %", ["Run", "K", *groups], rows)

    rows = []
    for run, m in metrics.items():
        for k in ECE_K:
            for name in ("T=1", "T*"):
                e = m["accumulation"][name]["ece_qk"][str(k)]
                rows.append(
                    [
                        run,
                        str(k),
                        name,
                        _pct(e["all"]),
                        _pct(e["mean_confidence"]),
                        _pct(e["accuracy"]),
                        *(_pct(e["by_group"][x]) for x in groups),
                    ]
                )
    lines += _table(
        "ECE of accumulated confidence q_K, %",
        ["Run", "K", "T", "All", "Mean q_K", "Accuracy", *groups],
        rows,
    )

    def cell(s: dict) -> str:
        f = s["mean_frames_decided"]
        frames = "—" if f is None else f"{f:.1f}"
        return f"{_pct(s['decided_accuracy'])} / {frames} / {_pct(s['unknown_share'])}"

    rows = []
    for run, m in metrics.items():
        for name in ("T=1", "T*"):
            for tau in TAUS:
                st = m["accumulation"][name]["stopping"][str(tau)]
                rows.append(
                    [
                        run,
                        name,
                        str(tau),
                        cell(st["all"]),
                        *(cell(st["by_group"][x]) for x in groups),
                    ]
                )
    lines += _table(
        f"Stopping rule, K_max = {K_MAX}: decided accuracy % / mean frames / unknown %",
        ["Run", "T", "τ", "All", *groups],
        rows,
    )

    rows = []
    for run, m in metrics.items():
        dg = m["diagnostic_evidence_scale"]
        for k in ECE_K:
            e = dg["ece_qk"][str(k)]
            rows.append(
                [
                    run,
                    f"{dg['beta']:.3f}",
                    str(k),
                    _pct(e["all"]),
                    *(_slash(e["by_group"][x], "mean_confidence", "accuracy") for x in groups),
                ]
            )
    lines += _table(
        "Diagnostic: q_K = softmax(beta * Lambda_K), beta on val sequences; "
        "ECE % and mean q_K / accuracy %",
        ["Run", "beta", "K", "ECE all", *groups],
        rows,
    )
    return "\n".join(lines).lstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    torch.set_num_threads(4)  # leave CPU for a concurrent GPU training run

    exp = exp_dir()
    metrics = {}
    for run in RUNS:
        path = exp / run / "metrics.json"
        if not args.plot_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(analyze(run), indent=1))
            print(f"[{run}] done", flush=True)
        metrics[run] = json.loads(path.read_text())

    plot_reliability(metrics, exp / "reliability.png")
    plot_accuracy_vs_k(metrics, exp / "accuracy_vs_k.png")
    plot_stopping(metrics["spec_n32"], exp / "stopping_spec_n32.png")
    plot_stopping(metrics["oshea_iq"], exp / "stopping_oshea_iq.png")
    table = summary(metrics)
    (exp / "summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

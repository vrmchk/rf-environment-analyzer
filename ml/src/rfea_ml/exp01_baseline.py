"""Experiment 01: STFT window length for the spectrogram CNN vs. the O'Shea IQ CNN on
RadioML 2016.10A.

Runs:
  spec_n16, spec_n32, spec_n64 - SpectrogramCNN on STFT spectrograms (Hann window,
      hop = n_fft / 4, resized to 64x64), window length 16 / 32 / 64 samples;
  oshea_iq - OSheaCNN on raw IQ samples.

Each run uses the same stratified 60/20/20 split. Outputs go to <output_dir>/exp01/<run>/
(see rfea_ml.paths): model.pt, history.json, metrics.json, logits.npz (val and test
logits, kept for the calibration and decision accumulation experiments), plus summary
plots in <output_dir>/exp01/.

Usage: uv run python -m rfea_ml.exp01_baseline [--runs spec_n16 oshea_iq] [--plot-only]
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from rfea_core import SpectrogramCNN, StftConfig, log_spectrogram
from rfea_ml import radioml
from rfea_ml.baselines import OSheaCNN
from rfea_ml.paths import git_commit, output_dir
from rfea_ml.plots import load_metrics, plot_accuracy_vs_snr, plot_confusions, summary_table
from rfea_ml.train import TrainConfig, evaluate, fit, predict_logits

RUNS = ("spec_n16", "spec_n32", "spec_n64", "oshea_iq")


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


def exp_dir() -> Path:
    return output_dir() / "exp01"


def run_one(run: str, data: radioml.RadioML, split, device: torch.device, cfg: TrainConfig):
    train, val, test = split
    out = exp_dir() / run
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


def plot_summary() -> None:
    exp = exp_dir()
    metrics = load_metrics({run: exp / run / "metrics.json" for run in RUNS})
    if not metrics:
        return
    plot_accuracy_vs_snr(metrics, exp / "accuracy_vs_snr.png")
    plot_confusions(metrics, exp / "confusion_snr_ge_10.png")
    table = summary_table(metrics)
    (exp / "summary.md").write_text(table, encoding="utf-8")
    print(table)


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

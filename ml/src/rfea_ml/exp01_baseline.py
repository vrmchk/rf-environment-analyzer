"""Experiment 01: STFT window length for the spectrogram CNN vs. the O'Shea IQ CNN on
RadioML 2016.10A.

Runs:
  spec_n16, spec_n32, spec_n64 - SpectrogramCNN on STFT spectrograms (Hann window,
      hop = n_fft / 4, resized to 64x64), window length 16 / 32 / 64 samples;
  oshea_iq - OSheaCNN on raw IQ samples.

Each run uses the same stratified 60/20/20 split (seed 0). --seed changes only the
training seed (weight initialization, batch order). Outputs go to <output_dir>/exp01/<run>/
for seed 0 and to <output_dir>/exp01/seed<N>/<run>/ otherwise (see rfea_ml.paths):
model.pt, history.json, metrics.json, logits.npz (val and test logits, kept for the
calibration and decision accumulation experiments), plus summary plots in the same folder.

Epoch limits: 60 for the spectrogram CNNs, 200 for the O'Shea CNN (it converges slowly);
early stopping usually ends training much earlier.

Usage: uv run python -m rfea_ml.exp01_baseline [--runs spec_n16 oshea_iq] [--seed 1 2]
    [--skip-existing] [--plot-only]
(several seeds run one after another in one process; --skip-existing resumes an
interrupted job: runs that already have metrics.json are not retrained)
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
MAX_EPOCHS = {"spec_n16": 60, "spec_n32": 60, "spec_n64": 60, "oshea_iq": 200}


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


def exp_dir(seed: int = 0) -> Path:
    base = output_dir() / "exp01"
    return base if seed == 0 else base / f"seed{seed}"


def run_one(
    run: str, data: radioml.RadioML, split, device: torch.device, cfg: TrainConfig, commit: str
):
    train, val, test = split
    out = exp_dir(cfg.seed) / run
    out.mkdir(parents=True, exist_ok=True)

    x = torch.from_numpy(data.x).to(device)
    y = torch.from_numpy(data.y).to(device)
    torch.manual_seed(cfg.seed)  # weight initialization
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
        commit=commit,
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


def plot_summary(seed: int = 0) -> None:
    exp = exp_dir(seed)
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
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[0], help="training seeds; the split is fixed"
    )
    parser.add_argument("--max-epochs", type=int, help="override the per-model epoch limit")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not retrain runs that already have metrics.json (resume an interrupted job)",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if not args.plot_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        data = radioml.load()
        split = radioml.split_indices(data.y, data.snr, seed=0)
        commit = git_commit()  # code is imported once; later edits do not affect this run
        for seed in args.seed:
            for run in args.runs:
                done = exp_dir(seed) / run / "metrics.json"
                if args.skip_existing and done.exists():
                    print(f"[{run}] seed {seed}: {done} exists, skipped", flush=True)
                    continue
                cfg = TrainConfig(max_epochs=args.max_epochs or MAX_EPOCHS[run], seed=seed)
                run_one(run, data, split, device, cfg, commit)
            plot_summary(seed)
    else:
        for seed in args.seed:
            plot_summary(seed)


if __name__ == "__main__":
    main()

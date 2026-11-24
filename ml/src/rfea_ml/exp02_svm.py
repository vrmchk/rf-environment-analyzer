"""Experiment 02: SVM baseline on 7 hand-crafted features (thesis, subsection 3.5).

Features: rfea_ml.features.svm_features (normalized cumulants, amplitude and phase
increment statistics). Standardized with StandardScaler fitted on train only, then
SVC(kernel="rbf") with library defaults (C = 1, gamma = "scale", one-vs-one); one
model for all SNR levels. Same stratified 60/20/20 split as experiment 01 (seed 0);
the validation part is not used, since no hyperparameters are tuned.

Outputs in <output_dir>/exp02/: svm/metrics.json, svm/model.joblib (scaler + SVM),
svm/features.npz, and comparison plots and table together with the experiment 01 runs.

Usage: uv run python -m rfea_ml.exp02_svm [--plot-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rfea_ml import radioml
from rfea_ml.exp01_baseline import RUNS as EXP01_RUNS
from rfea_ml.features import FEATURE_NAMES, svm_features
from rfea_ml.paths import git_commit, output_dir
from rfea_ml.plots import load_metrics, plot_accuracy_vs_snr, plot_confusions, summary_table
from rfea_ml.train import evaluate_predictions


def exp_dir() -> Path:
    return output_dir() / "exp02"


def run() -> None:
    out = exp_dir() / "svm"
    out.mkdir(parents=True, exist_ok=True)
    data = radioml.load()
    train, val, test = radioml.split_indices(data.y, data.snr, seed=0)

    t0 = time.perf_counter()
    features = svm_features(data.x)
    feature_seconds = time.perf_counter() - t0
    print(f"features {features.shape} in {feature_seconds:.1f}s", flush=True)

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", cache_size=2000))
    t0 = time.perf_counter()
    model.fit(features[train], data.y[train])
    train_seconds = time.perf_counter() - t0
    svc = model[-1]
    print(f"fit {train_seconds / 60:.1f} min, support vectors {svc.n_support_.sum():,}", flush=True)

    t0 = time.perf_counter()
    pred = model.predict(features[test])
    predict_seconds = time.perf_counter() - t0

    metrics = evaluate_predictions(pred, data.y[test], data.snr[test], data.classes)
    metrics.update(
        run="svm",
        params={
            "model": "StandardScaler + SVC",
            "features": list(FEATURE_NAMES),
            "kernel": svc.kernel,
            "C": svc.C,
            "gamma": svc.gamma,
            "gamma_value": float(svc._gamma),
            "decision": "one-vs-one",
        },
        n_support=int(svc.n_support_.sum()),
        train_seconds=train_seconds,
        feature_seconds=feature_seconds,
        predict_seconds=predict_seconds,
        commit=git_commit(),
        split={"train": len(train), "val": len(val), "test": len(test), "seed": 0},
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    joblib.dump(model, out / "model.joblib")
    np.savez(
        out / "features.npz",
        features=features, names=np.array(FEATURE_NAMES), y=data.y, snr=data.snr,
        train=train, val=val, test=test, test_pred=pred,
    )  # fmt: skip
    print(
        f"[svm] overall {metrics['accuracy_overall']:.4f}  SNR>=0 "
        f"{metrics['accuracy_snr_ge_0']:.4f}  SNR>=10 {metrics['accuracy_snr_ge_10']:.4f}  "
        f"predict {predict_seconds / 60:.1f} min",
        flush=True,
    )


def plot_summary() -> None:
    exp1, exp2 = output_dir() / "exp01", exp_dir()
    paths = {r: exp1 / r / "metrics.json" for r in EXP01_RUNS}
    paths["svm"] = exp2 / "svm" / "metrics.json"
    metrics = load_metrics(paths)
    plot_accuracy_vs_snr(metrics, exp2 / "accuracy_vs_snr.png")
    compare = {r: metrics[r] for r in ("spec_n32", "oshea_iq", "svm") if r in metrics}
    plot_confusions(compare, exp2 / "confusion_snr_ge_10.png")
    table = summary_table(metrics)
    (exp2 / "summary.md").write_text(table, encoding="utf-8")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if not args.plot_only:
        run()
    plot_summary()


if __name__ == "__main__":
    main()

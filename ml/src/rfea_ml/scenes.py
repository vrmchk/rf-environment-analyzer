"""Wideband scenes for Virtual Source (FR-4) in SigMF, with reference annotations.

A scene is a complex baseband recording at SCENE_FS = 1 MHz with several narrowband signals
at different frequencies and times over white noise. Every signal is built like a training
record of rfea_ml.torchsig_data, so the classifier sees the same domain:
  1. a TorchSig class generator makes the clean signal at its own rate f_sig, with the
     bandwidth a random fraction 0.15-0.5 of f_sig (as in training);
  2. our impairments (carrier offset and drift, phase noise, clock offset);
  3. upsampling by an integer U to the scene rate (f_sig = SCENE_FS / U); U is chosen so
     that the measured bandwidth (90 % of the power, rfea_core.occupied_bandwidth) lands
     in SceneConfig.bandwidth_hz;
  4. shift to a free frequency channel and placement in a random time interval;
  5. power set for the SNR the classifier will see: SNR over the band f_s' = measured
     bandwidth / 0.285, the rate the backend decimates to (method.md, section 1), with the
     scene noise of unit power over SCENE_FS. So annotation SNRs are on the same scale
     as the experiment 04 tables.

Signals never overlap in frequency: the band is split into N_CHANNELS channels and a burst
takes a channel that is free for its whole duration. Time placement: LANES independent
lanes of bursts with random gaps, so 3-6 signals are on the air at once.

Output in <data_dir>/scenes/: <name>.sigmf-data (cf32_le) and <name>.sigmf-meta (JSON,
annotations: sample start/count, frequency edges holding 5-95 % of the power, label,
SNR, measured bandwidth).

Usage: uv run python -m rfea_ml.scenes --name demo --duration 20 --snr 10 18
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from rfea_core import occupied_bandwidth
from rfea_ml import torchsig_data as tsd
from rfea_ml.paths import data_dir, git_commit
from rfea_ml.radioml import CLASSES

SCENE_FS = 1e6
# Median measured bandwidth / f_s' of the model the scenes demo, eval1-eval4 were made for
# (exp04); kept so they regenerate identically. The system model has 0.316: the classifier
# sees a 10 lg(0.316 / 0.285) = 0.45 dB lower SNR than annotated.
DECIMATION_FRACTION = 0.285
N_CHANNELS = 8  # 125 kHz each at 1 MHz
OOD_SCENE_CLASSES = ("ofdm-64", "lfm-data")
RAMP_S = 1e-3  # raised-cosine on/off ramp of a burst
PROBE_LEN = 8192


@dataclass(frozen=True)
class SceneConfig:
    name: str = "demo"
    duration_s: float = 60.0
    seed: int = 0
    lanes: int = 6
    burst_s: tuple[float, float] = (2.0, 15.0)
    gap_s: tuple[float, float] = (0.5, 4.0)
    snr_db: tuple[float, float] = (-4.0, 18.0)
    bandwidth_hz: tuple[float, float] = (10e3, 60e3)
    signal_bandwidth: tuple[float, float] = (0.15, 0.5)  # of f_sig, as in training
    ood_share: float = 0.1
    center_freq_hz: float = 100e6
    noise: bool = True


def scenes_dir() -> Path:
    return data_dir() / "scenes"


def _class_cycle(rng: np.random.Generator):
    """Training classes in shuffled rounds, so they appear equally often."""
    while True:
        yield from (CLASSES[i] for i in rng.permutation(len(CLASSES)))


def schedule(cfg: SceneConfig, rng: np.random.Generator) -> list[dict]:
    """Bursts: start/end (s), channel, label and TorchSig class; no frequency overlap."""
    bursts: list[dict] = []
    classes = _class_cycle(rng)
    for _ in range(cfg.lanes):
        t = rng.uniform(0, cfg.gap_s[1])
        while t + cfg.burst_s[0] <= cfg.duration_s:
            end = min(t + rng.uniform(*cfg.burst_s), cfg.duration_s)
            busy = {b["channel"] for b in bursts if b["start_s"] < end and t < b["end_s"]}
            free = [c for c in range(N_CHANNELS) if c not in busy]
            if free:
                if rng.random() < cfg.ood_share:
                    label = tsig = str(rng.choice(OOD_SCENE_CLASSES))
                else:
                    label = next(classes)
                    tsig = str(rng.choice(tsd.TORCHSIG_CLASSES[label]))
                bursts.append({"start_s": t, "end_s": end, "channel": int(rng.choice(free)),
                               "label": label, "tsig_class": tsig})  # fmt: skip
            t = end + rng.uniform(*cfg.gap_s)
    return sorted(bursts, key=lambda b: b["start_s"])


def _clean(tsig: str, bandwidth: float, length: int, seed: int) -> np.ndarray:
    gen_cfg = tsd.GenConfig(bandwidth=(bandwidth, bandwidth))
    ds = tsd._torchsig_generators((tsig,), gen_cfg, seed, length=length)
    return np.asarray(ds.signal_generators[0]().data, dtype=np.complex128)


def _measured(x: np.ndarray) -> float:
    """Occupied bandwidth (fraction of the rate) from 4096-sample chunks."""
    n = len(x) // 4096 * 4096
    chunks = torch.from_numpy(x[: min(n, 64 * 4096)].reshape(-1, 4096))
    return float(occupied_bandwidth(chunks).median())


def _edges(x: np.ndarray) -> tuple[float, float]:
    """Frequencies (fraction of the rate) holding 5 % and 95 % of the power."""
    n = len(x) // 4096 * 4096
    p = np.abs(np.fft.fftshift(np.fft.fft(x[: min(n, 64 * 4096)].reshape(-1, 4096)), -1)) ** 2
    c = np.cumsum(p.mean(0))
    c /= c[-1]
    f = np.fft.fftshift(np.fft.fftfreq(4096))
    return float(f[np.searchsorted(c, 0.05)]), float(f[np.searchsorted(c, 0.95)])


def make_burst(b: dict, cfg: SceneConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Burst samples at SCENE_FS (zero frequency, unit noise scale) and its annotation."""
    n_out = int(round((b["end_s"] - b["start_s"]) * SCENE_FS))
    bw = float(rng.uniform(*cfg.signal_bandwidth))
    seed = int(rng.integers(2**31))
    target = rng.uniform(*cfg.bandwidth_hz)
    m_probe = _measured(_clean(b["tsig_class"], bw, PROBE_LEN, seed))
    up = int(np.clip(round(m_probe * SCENE_FS / target), 2, 100))

    n_sig = math.ceil(n_out / up) + 64
    imp = tsd.Impairments()
    # Training records are RECORD_LEN long; a linear drift over a whole burst would sweep
    # the carrier far beyond what the classifier saw, so the total drift of a burst equals
    # that of one training record. The clock offset needs extra source samples.
    imp = tsd.Impairments(cfo=imp.cfo, drift=imp.drift * tsd.RECORD_LEN / n_sig,
                          phase_noise=imp.phase_noise, clock=imp.clock)  # fmt: skip
    s = _clean(b["tsig_class"], bw, n_sig + math.ceil(n_sig * imp.clock), seed)
    s = tsd.impair(s, imp, rng, length=n_sig)
    m = _measured(s)
    lo, hi = _edges(s)
    x = resample_poly(s, up, 1)[:n_out]
    ramp = min(int(RAMP_S * SCENE_FS), n_out // 2)
    w = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    x[:ramp] *= w
    x[n_out - ramp :] *= w[::-1]

    bandwidth_hz = m * SCENE_FS / up
    snr_db = float(rng.uniform(*cfg.snr_db))
    # Noise has unit power over SCENE_FS; the classifier sees it over f_s'.
    f_dec = bandwidth_hz / DECIMATION_FRACTION
    p_signal = 10 ** (snr_db / 10) * f_dec / SCENE_FS
    x *= np.sqrt(p_signal / np.mean(np.abs(x[ramp : n_out - ramp]) ** 2))

    width = SCENE_FS / N_CHANNELS
    slack = max(width - (hi - lo) * SCENE_FS / up, 0) / 2 * 0.8
    offset = -SCENE_FS / 2 + (b["channel"] + 0.5) * width + rng.uniform(-slack, slack)
    offset -= (lo + hi) / 2 * SCENE_FS / up  # center the occupied band in the channel
    x = x * np.exp(2j * np.pi * offset / SCENE_FS * np.arange(n_out))

    start = int(round(b["start_s"] * SCENE_FS))
    ann = {
        "core:sample_start": start,
        "core:sample_count": n_out,
        "core:freq_lower_edge": cfg.center_freq_hz + offset + lo * SCENE_FS / up,
        "core:freq_upper_edge": cfg.center_freq_hz + offset + hi * SCENE_FS / up,
        "core:label": b["label"],
        "rfea:tsig_class": b["tsig_class"],
        "rfea:in_training_set": b["label"] in CLASSES,
        "rfea:snr_db": round(snr_db, 2),
        "rfea:measured_bandwidth_hz": round(bandwidth_hz, 1),
        "rfea:upsampling": up,
    }
    return x.astype(np.complex64), ann


def generate(cfg: SceneConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / cfg.name
    data_path, meta_path = base.with_suffix(".sigmf-data"), base.with_suffix(".sigmf-meta")
    if meta_path.exists():
        raise FileExistsError(f"{meta_path} exists; choose another --name")
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.duration_s * SCENE_FS))
    scene = np.zeros(n, dtype=np.complex64)
    annotations = []
    for b in schedule(cfg, rng):
        x, ann = make_burst(b, cfg, rng)
        s0 = ann["core:sample_start"]
        scene[s0 : s0 + len(x)] += x
        annotations.append(ann)
    if cfg.noise:
        for i in range(0, n, 1_000_000):
            k = min(1_000_000, n - i)
            w = rng.normal(0, np.sqrt(0.5), (2, k)).astype(np.float32)
            scene[i : i + k] += w[0] + 1j * w[1]
    scene.astype("<c8").tofile(data_path)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": SCENE_FS,
            "core:version": "1.0.0",
            "core:description": f"rfea synthetic scene {cfg.name}",
            "core:recorder": "rfea_ml.scenes",
            "rfea:config": asdict(cfg),
            "rfea:noise_power": 1.0 if cfg.noise else 0.0,
            "rfea:snr_definition": "signal power over noise power in f_s' = measured "
            f"bandwidth / {DECIMATION_FRACTION} (thesis (3.3), method.md section 1)",
            "rfea:commit": git_commit(),
            "rfea:seconds": round(time.perf_counter() - t0, 1),
        },
        "captures": [{"core:sample_start": 0, "core:frequency": cfg.center_freq_hz}],
        "annotations": annotations,
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=SceneConfig.name)
    parser.add_argument("--duration", type=float, default=SceneConfig.duration_s)
    parser.add_argument("--seed", type=int, default=SceneConfig.seed)
    parser.add_argument("--snr", type=float, nargs=2, default=SceneConfig.snr_db)
    parser.add_argument("--ood-share", type=float, default=SceneConfig.ood_share)
    parser.add_argument("--out", type=Path, default=None, help="default: <data_dir>/scenes")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    cfg = SceneConfig(name=args.name, duration_s=args.duration, seed=args.seed,
                      snr_db=tuple(args.snr), ood_share=args.ood_share)  # fmt: skip
    meta = generate(cfg, args.out or scenes_dir())
    g = json.loads(meta.read_text())
    print(f"{meta}: {len(g['annotations'])} signals, {g['global']['rfea:seconds']} s")


if __name__ == "__main__":
    main()

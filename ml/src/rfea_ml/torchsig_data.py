"""TorchSig 2.2 records for the classifier of the system and for decision accumulation.

A record is one continuous narrowband signal at zero frequency, RECORD_LEN = 4096 samples
(32 frames of FRAME_LEN = 128 samples, the length of a RadioML 2016.10A example). Frames
of one record share the carrier offset, phase, drift and bandwidth, so they are correlated
the same way as consecutive frames of one detected signal in the system.

Generation of one record:
  1. TorchSig builds a clean baseband signal of the class (no noise, no TorchSig
     impairments); its bandwidth is a random fraction of the sample rate, uniform in
     GenConfig.bandwidth (default 0.15-0.5; --bandwidth changes it);
  1a. only with --measured LO HI: the clean signal is resampled so that its measured
     bandwidth (90 % of the power, rfea_core.occupied_bandwidth) is uniform in [LO, HI]
     for every class. With TorchSig's own bandwidth the measured width depends on the
     class (WBFM ~0.11, FSK ~0.19, PSK/QAM ~0.37 of the rate), the network learns it as
     a class cue, and the backend, which decimates every signal to one fraction, breaks
     it (WBFM taken for GFSK). The source is generated RESAMPLE_CAP times longer so that
     narrow signals can be widened; `bandwidth` is then TorchSig's value at the new rate;
  2. our impairments, close to the RadioML channel: carrier frequency offset and linear
     drift, initial phase, Wiener phase noise, sample clock offset (see Impairments);
  3. power normalized to 1 and complex white Gaussian noise added so that
     SNR = 10 lg(P_s / P_w) over the whole sampled band (thesis, (3.3)); TorchSig's own
     SNR is defined differently (peak of the averaged spectrum over one noise bin).

Classes: the 11 RadioML classes in the same order (radioml.CLASSES) with their TorchSig
counterparts; AM-SSB alternates am-usb and am-lsb. Four classes outside the training set
(OOD_CLASSES) are generated for the test only, to check the "unknown" decision.

Split 60/20/20 by record within every (class, SNR) pair, so frames of one signal never
appear in two parts. Output in <data_dir>/<name>/ (default name: torchsig): train.npz,
val.npz, test.npz, ood_test.npz (x complex64 (n, RECORD_LEN), y, snr, bandwidth,
tsig_class) and config.json. Different bandwidth ranges go to different names, e.g.

  uv run python -m rfea_ml.torchsig_data --bandwidth 0.10 0.20 --name torchsig_narrow
  uv run python -m rfea_ml.torchsig_data --measured 0.15 0.45 --name torchsig_uniform

Usage: uv run python -m rfea_ml.torchsig_data [--records 250] [--workers 4]
       [--bandwidth LO HI] [--measured LO HI] [--name torchsig]
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from rfea_core import occupied_bandwidth
from rfea_ml.paths import data_dir, git_commit
from rfea_ml.radioml import CLASSES

RECORD_LEN = 4096
FRAME_LEN = 128
SNRS = tuple(range(-20, 20, 2))
# RadioML class -> TorchSig 2.2 classes (docs: datasets.md, section 4); several names are used
# in turn.
TORCHSIG_CLASSES = {
    "8PSK": ("8psk",),
    "AM-DSB": ("am-dsb",),
    "AM-SSB": ("am-usb", "am-lsb"),
    "BPSK": ("bpsk",),
    "CPFSK": ("2fsk",),
    "GFSK": ("2gfsk",),
    "PAM4": ("4ask",),
    "QAM16": ("16qam",),
    "QAM64": ("64qam",),
    "QPSK": ("qpsk",),
    "WBFM": ("fm",),
}
OOD_CLASSES = ("ofdm-64", "lfm-data", "4fsk", "32qam")  # test only: two far, two near
SAMPLE_RATE = 1e6  # TorchSig needs Hz; everything below is relative to it
MARGIN = 64  # extra samples generated so that clock-offset resampling stays inside the signal
DEFAULT_NAME = "torchsig"  # dataset folder in <data_dir>
RESAMPLE_CAP = 16  # source length, in records, available for widening a signal (--measured)
RESAMPLE_EDGE = 64  # output samples dropped at each end of a resampled record (transients)


@dataclass(frozen=True)
class Impairments:
    """Per-record random impairments; ranges are uniform, frequencies relative to fs."""

    cfo: float = 0.01  # carrier offset in [-cfo, cfo] (residual error of the channelizer)
    drift: float = 1e-7  # carrier drift per sample in [-drift, drift] (0.04 % fs over a record)
    phase_noise: float = 0.01  # Wiener phase noise: increment std in [0, phase_noise] rad
    clock: float = 1e-4  # sample clock offset in [-clock, clock] (0.4 sample over a record)


@dataclass(frozen=True)
class GenConfig:
    records: int = 250  # per (class, SNR) pair
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    ood_records: int = 50  # per (OOD class, SNR), test only
    bandwidth: tuple[float, float] = (0.15, 0.5)  # fraction of the sample rate
    # Measured (90 % power) bandwidth range, fraction of the rate; None keeps TorchSig's own.
    measured: tuple[float, float] | None = None
    snrs: tuple[int, ...] = SNRS
    seed: int = 0
    impairments: Impairments = field(default_factory=Impairments)


def dataset_dir(name: str = DEFAULT_NAME) -> Path:
    return data_dir() / name


def read_config(name: str = DEFAULT_NAME) -> dict:
    """config.json written by generate(): GenConfig, counts, commit."""
    return json.loads((dataset_dir(name) / "config.json").read_text())


def _torchsig_generators(
    names: tuple[str, ...], cfg: GenConfig, seed: int, length: int = RECORD_LEN
):
    from torchsig.datasets.datasets import TorchSigIterableDataset  # heavy import

    n, fs = length + MARGIN, SAMPLE_RATE
    metadata = dict(
        num_iq_samples_dataset=n, num_signals_min=1, num_signals_max=1,
        fft_size=64, fft_stride=64, sample_rate=fs, noise_power_db=0.0,
        snr_db_min=0.0, snr_db_max=0.0, cochannel_overlap_probability=0.0,
        signal_duration_in_samples_min=n, signal_duration_in_samples_max=n,
        bandwidth_min=cfg.bandwidth[0] * fs, bandwidth_max=cfg.bandwidth[1] * fs,
        signal_center_freq_min=0.0, signal_center_freq_max=0.0,
        frequency_min=-fs / 2, frequency_max=fs / 2 - 1,
    )  # fmt: skip
    # The dataset is used only to configure the generators and seed them; calling a
    # generator directly returns the clean signal (the dataset would add its own noise).
    return TorchSigIterableDataset(signal_generators=list(names), metadata=metadata, seed=seed)


def impair(
    s: np.ndarray, imp: Impairments, rng: np.random.Generator, length: int = RECORD_LEN
) -> np.ndarray:
    """Carrier offset and drift, initial phase, phase noise, clock offset; returns `length`
    samples (s needs MARGIN extra samples for the clock offset)."""
    n = np.arange(length)
    t = n * (1 + rng.uniform(-imp.clock, imp.clock))  # resample at the drifted clock
    src = np.arange(len(s))
    s = np.interp(t, src, s.real) + 1j * np.interp(t, src, s.imag)
    f0 = rng.uniform(-imp.cfo, imp.cfo)
    df = rng.uniform(-imp.drift, imp.drift)
    sigma = rng.uniform(0, imp.phase_noise)
    phase = (
        2 * np.pi * (f0 * n + df * n**2 / 2)
        + rng.uniform(0, 2 * np.pi)
        + np.cumsum(rng.normal(0, sigma, length))
    )
    return s * np.exp(1j * phase)


def add_noise(s: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Normalize power to 1 and add complex AWGN: SNR = 10 lg(P_s / P_w) (thesis, (3.3))."""
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    p_w = 10 ** (-snr_db / 10)
    w = rng.normal(0, np.sqrt(p_w / 2), (2, len(s)))
    return (s + w[0] + 1j * w[1]).astype(np.complex64)


def to_measured(
    s: np.ndarray, target: float, length: int = RECORD_LEN + MARGIN
) -> tuple[np.ndarray, float]:
    """Resample a clean signal so that its measured bandwidth is `target` (fraction of the
    rate). Returns `length` samples and the rate ratio r = new fraction / old fraction.

    The source must hold about length * r samples (r is capped by what it holds)."""
    m = float(occupied_bandwidth(torch.from_numpy(s[None, :RECORD_LEN].astype(np.complex64)))[0])
    r = min(target / m, (len(s) - 1) / (length + 2 * RESAMPLE_EDGE))
    q = Fraction(r).limit_denominator(32)  # new rate = old * up / down, fraction * down / up
    up, down = q.denominator, q.numerator
    src = s[: math.ceil((length + 2 * RESAMPLE_EDGE) * down / up) + 1]
    y = resample_poly(src, up, down)[RESAMPLE_EDGE : RESAMPLE_EDGE + length]
    return y, down / up


def generate_class(label: int, names: tuple[str, ...], n_records: int, cfg: GenConfig) -> dict:
    """All records of one class: n_records per SNR, TorchSig names used in turn.

    With cfg.measured, every clean signal is generated longer (RESAMPLE_CAP times) and
    resampled so that its measured bandwidth is uniform in cfg.measured for every class."""
    seed = cfg.seed * 1000 + label
    n_src = RECORD_LEN
    if cfg.measured:
        n_src = math.ceil((RECORD_LEN + MARGIN + 2 * RESAMPLE_EDGE) * RESAMPLE_CAP)
    ds = _torchsig_generators(names, cfg, seed, length=n_src)
    gens = {g.class_name: g for g in ds.signal_generators}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("x", "snr", "bandwidth", "tsig_class")}
    for snr in cfg.snrs:
        for i in range(n_records):
            name = names[i % len(names)]
            sig = gens[name]()
            s, r = np.asarray(sig.data, dtype=np.complex128), 1.0
            if cfg.measured:
                s, r = to_measured(s, rng.uniform(*cfg.measured))
            x = impair(s, cfg.impairments, rng)
            out["x"].append(add_noise(x, snr, rng))
            out["snr"].append(snr)
            out["bandwidth"].append(sig["bandwidth"] / SAMPLE_RATE * r)
            out["tsig_class"].append(name)
    return {
        "x": np.stack(out["x"]),
        "y": np.full(len(out["x"]), label, dtype=np.int64),
        "snr": np.array(out["snr"], dtype=np.int64),
        "bandwidth": np.array(out["bandwidth"], dtype=np.float32),
        "tsig_class": np.array(out["tsig_class"]),
    }


def split_by_record(n_records: int, n_snrs: int, fractions) -> list[np.ndarray]:
    """Indices into one class block (SNR-major, n_records per SNR) for train/val/test.
    Records within a pair are i.i.d., so the first ones go to train, then val, then test."""
    n_train = round(n_records * fractions[0])
    n_val = round(n_records * fractions[1])
    bounds = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_records)]
    return [
        np.concatenate([np.arange(s * n_records + a, s * n_records + b) for s in range(n_snrs)])
        for a, b in bounds
    ]


def _concat(blocks: list[dict]) -> dict:
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def generate(cfg: GenConfig, out: Path, workers: int = 4) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(c, TORCHSIG_CLASSES[name], cfg.records) for c, name in enumerate(CLASSES)]
    jobs += [(len(CLASSES) + i, (name,), cfg.ood_records) for i, name in enumerate(OOD_CLASSES)]
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(generate_class, c, names, n, cfg) for c, names, n in jobs]
        blocks = [f.result() for f in futures]

    parts = {"train": [], "val": [], "test": []}
    for block in blocks[: len(CLASSES)]:
        idx = split_by_record(cfg.records, len(cfg.snrs), cfg.fractions)
        for part, i in zip(parts, idx, strict=True):
            parts[part].append({k: v[i] for k, v in block.items()})
    for part, b in parts.items():
        np.savez(out / f"{part}.npz", **_concat(b))
    ood = _concat(blocks[len(CLASSES) :])
    ood["y"] = np.full(len(ood["y"]), -1, dtype=np.int64)  # no training class
    np.savez(out / "ood_test.npz", **ood)

    import torchsig

    meta = {
        "config": asdict(cfg),
        "record_len": RECORD_LEN,
        "frame_len": FRAME_LEN,
        "classes": list(CLASSES),
        "torchsig_classes": TORCHSIG_CLASSES,
        "ood_classes": list(OOD_CLASSES),
        "torchsig": torchsig.__version__,
        "commit": git_commit(),
        "seconds": round(time.perf_counter() - t0, 1),
        "counts": {p: int(sum(len(b["y"]) for b in v)) for p, v in parts.items()}
        | {"ood_test": len(ood["y"])},
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("counts", "seconds", "commit")}))


@dataclass
class Records:
    x: np.ndarray  # (n, RECORD_LEN) complex64
    y: np.ndarray  # (n,) class index into radioml.CLASSES, -1 for OOD
    snr: np.ndarray  # (n,) dB
    bandwidth: np.ndarray  # (n,) fraction of the sample rate
    tsig_class: np.ndarray  # (n,) TorchSig class name


def load(part: str, name: str = DEFAULT_NAME) -> Records:
    """part: train, val, test or ood_test; name: dataset folder in <data_dir>."""
    path = dataset_dir(name) / f"{part}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: uv run python -m rfea_ml.torchsig_data")
    with np.load(path, allow_pickle=False) as d:
        return Records(*(d[k] for k in ("x", "y", "snr", "bandwidth", "tsig_class")))


def frames(x: np.ndarray, starts: np.ndarray | int = 0, count: int = 1) -> np.ndarray:
    """Consecutive frames as RadioML-shaped I/Q arrays.

    x: (n, RECORD_LEN) complex records; starts: first frame index per record (or one for all).
    Returns (n, count, 2, FRAME_LEN) float32.
    """
    starts = np.broadcast_to(np.asarray(starts), (len(x),))
    offsets = (starts[:, None] + np.arange(count)) * FRAME_LEN  # (n, count)
    idx = offsets[..., None] + np.arange(FRAME_LEN)  # (n, count, FRAME_LEN)
    z = np.take_along_axis(x[:, None, :], idx, axis=2)
    return np.stack([z.real, z.imag], axis=2).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=GenConfig.records)
    parser.add_argument("--ood-records", type=int, default=GenConfig.ood_records)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--bandwidth", type=float, nargs=2, default=GenConfig.bandwidth, metavar=("LO", "HI"),
        help="signal bandwidth range, fraction of the sample rate (default %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--measured", type=float, nargs=2, default=None, metavar=("LO", "HI"),
        help="resample every record to a measured bandwidth uniform in [LO, HI], fraction "
        "of the sample rate, the same for all classes (default: off)",
    )  # fmt: skip
    parser.add_argument("--name", default=DEFAULT_NAME, help="dataset folder in <data_dir>")
    parser.add_argument("--out", type=Path, default=None, help="default: <data_dir>/<name>")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    lo, hi = args.bandwidth
    if not 0 < lo < hi <= 1:
        parser.error("--bandwidth needs 0 < LO < HI <= 1")
    measured = None
    if args.measured:
        measured = tuple(args.measured)
        if not 0 < measured[0] < measured[1] < 1:
            parser.error("--measured needs 0 < LO < HI < 1")
    cfg = GenConfig(records=args.records, ood_records=args.ood_records, bandwidth=(lo, hi),
                    measured=measured)  # fmt: skip
    out = args.out or dataset_dir(args.name)
    if (out / "config.json").exists():
        parser.error(f"{out} already holds a dataset; choose another --name")
    generate(cfg, out, args.workers)


if __name__ == "__main__":
    main()

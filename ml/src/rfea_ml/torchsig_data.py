"""TorchSig 2.2 records for the classifier of the system and for decision accumulation.

A record is one continuous narrowband signal at zero frequency, RECORD_LEN = 4096 samples
(32 frames of FRAME_LEN = 128 samples, the length of a RadioML 2016.10A example). Frames
of one record share the carrier offset, phase, drift and bandwidth, so they are correlated
the same way as consecutive frames of one detected signal in the system.

Generation of one record:
  1. TorchSig builds a clean baseband signal of the class (no noise, no TorchSig
     impairments); its bandwidth is a random fraction 0.15-0.5 of the sample rate;
  2. our impairments, close to the RadioML channel: carrier frequency offset and linear
     drift, initial phase, Wiener phase noise, sample clock offset (see Impairments);
  3. power normalized to 1 and complex white Gaussian noise added so that
     SNR = 10 lg(P_s / P_w) over the whole sampled band (thesis, (3.3)); TorchSig's own
     SNR is defined differently (peak of the averaged spectrum over one noise bin).

Classes: the 11 RadioML classes in the same order (radioml.CLASSES) with their TorchSig
counterparts; AM-SSB alternates am-usb and am-lsb. Four classes outside the training set
(OOD_CLASSES) are generated for the test only, to check the "unknown" decision.

Split 60/20/20 by record within every (class, SNR) pair, so frames of one signal never
appear in two parts. Output in <data_dir>/torchsig/: train.npz, val.npz, test.npz,
ood_test.npz (x complex64 (n, RECORD_LEN), y, snr, bandwidth, tsig_class) and config.json.

Usage: uv run python -m rfea_ml.torchsig_data [--records 250] [--workers 4]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

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
    snrs: tuple[int, ...] = SNRS
    seed: int = 0
    impairments: Impairments = field(default_factory=Impairments)


def dataset_dir() -> Path:
    return data_dir() / "torchsig"


def _torchsig_generators(names: tuple[str, ...], cfg: GenConfig, seed: int):
    from torchsig.datasets.datasets import TorchSigIterableDataset  # heavy import

    n, fs = RECORD_LEN + MARGIN, SAMPLE_RATE
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


def impair(s: np.ndarray, imp: Impairments, rng: np.random.Generator) -> np.ndarray:
    """Carrier offset and drift, initial phase, phase noise, clock offset; returns RECORD_LEN."""
    n = np.arange(RECORD_LEN)
    t = n * (1 + rng.uniform(-imp.clock, imp.clock))  # resample at the drifted clock
    src = np.arange(len(s))
    s = np.interp(t, src, s.real) + 1j * np.interp(t, src, s.imag)
    f0 = rng.uniform(-imp.cfo, imp.cfo)
    df = rng.uniform(-imp.drift, imp.drift)
    sigma = rng.uniform(0, imp.phase_noise)
    phase = (
        2 * np.pi * (f0 * n + df * n**2 / 2)
        + rng.uniform(0, 2 * np.pi)
        + np.cumsum(rng.normal(0, sigma, RECORD_LEN))
    )
    return s * np.exp(1j * phase)


def add_noise(s: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Normalize power to 1 and add complex AWGN: SNR = 10 lg(P_s / P_w) (thesis, (3.3))."""
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    p_w = 10 ** (-snr_db / 10)
    w = rng.normal(0, np.sqrt(p_w / 2), (2, len(s)))
    return (s + w[0] + 1j * w[1]).astype(np.complex64)


def generate_class(label: int, names: tuple[str, ...], n_records: int, cfg: GenConfig) -> dict:
    """All records of one class: n_records per SNR, TorchSig names used in turn."""
    seed = cfg.seed * 1000 + label
    ds = _torchsig_generators(names, cfg, seed)
    gens = {g.class_name: g for g in ds.signal_generators}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("x", "snr", "bandwidth", "tsig_class")}
    for snr in cfg.snrs:
        for i in range(n_records):
            name = names[i % len(names)]
            sig = gens[name]()
            x = impair(np.asarray(sig.data, dtype=np.complex128), cfg.impairments, rng)
            out["x"].append(add_noise(x, snr, rng))
            out["snr"].append(snr)
            out["bandwidth"].append(sig["bandwidth"] / SAMPLE_RATE)
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


def load(part: str) -> Records:
    """part: train, val, test or ood_test."""
    path = dataset_dir() / f"{part}.npz"
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
    parser.add_argument("--out", type=Path, default=None, help="default: <data_dir>/torchsig")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    cfg = GenConfig(records=args.records, ood_records=args.ood_records)
    generate(cfg, args.out or dataset_dir(), args.workers)


if __name__ == "__main__":
    main()

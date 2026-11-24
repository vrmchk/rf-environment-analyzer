"""RadioML 2016.10A: download, safe conversion and loading.

The original file is a Python 2 pickle: {(modulation, snr): array(1000, 2, 128)}.
Unpickling arbitrary files can execute code, so the pickle is read once with a
restricted unpickler that only allows numpy array reconstruction, and saved as
a plain .npz file that is loaded with allow_pickle=False afterwards.

Usage: uv run python -m rfea_ml.radioml
"""

import hashlib
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rfea_ml.paths import data_dir

URL = "https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content"
MD5 = "ffedb2c24e33b826d3d0df4aa8f32c1b"
ARCHIVE = "RML2016.10a.tar.bz2"
# The Zenodo mirror ships a re-pickled file (protocol 4) named RML2016.10a_dict_optimized.pkl;
# the original DeepSig archive has RML2016.10a_dict.pkl. Both have the same structure.
PICKLE = "RML2016.10a_dict.pkl"
NPZ = "rml2016.10a.npz"

# Fixed class order (index = label) used across all experiments.
CLASSES = (
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
)  # fmt: skip


def dataset_dir() -> Path:
    return data_dir() / "radioml2016.10a"


class _NumpyOnlyUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) in self._ALLOWED:
            module = module.replace("numpy.core.", "numpy._core.")
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden global in pickle: {module}.{name}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    """Download the archive, verify md5, extract the pickle, convert to .npz."""
    root = dataset_dir()
    root.mkdir(parents=True, exist_ok=True)
    npz = root / NPZ
    if npz.exists():
        print(f"Already prepared: {npz}")
        return npz

    archive = root / ARCHIVE
    if not archive.exists() or _md5(archive) != MD5:
        print(f"Downloading {URL}\n  -> {archive}")
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(archive)
    if (got := _md5(archive)) != MD5:
        raise RuntimeError(f"md5 mismatch for {archive}: {got} != {MD5}")
    print("md5 OK")

    pkl = root / PICKLE
    if not pkl.exists():
        with tarfile.open(archive, "r:bz2") as tar:
            members = tar.getmembers()
            member = next(m for m in members if Path(m.name).name.startswith("RML2016.10a_dict"))
            if not member.isfile():
                raise RuntimeError(f"Unexpected archive member type: {member.name}")
            member.name = PICKLE  # drop any directory prefix
            tar.extract(member, root, filter="data")

    with pkl.open("rb") as f:
        raw = _NumpyOnlyUnpickler(f, encoding="latin1").load()

    xs, ys, snrs = [], [], []
    for (mod, snr), arr in sorted(raw.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mod = mod.decode() if isinstance(mod, bytes) else mod
        xs.append(np.asarray(arr, dtype=np.float32))
        ys.append(np.full(len(arr), CLASSES.index(mod), dtype=np.int64))
        snrs.append(np.full(len(arr), int(snr), dtype=np.int64))
    x, y, snr = np.concatenate(xs), np.concatenate(ys), np.concatenate(snrs)
    np.savez(npz, x=x, y=y, snr=snr, classes=np.array(CLASSES))
    print(f"Saved {npz}: x{x.shape}, {len(CLASSES)} classes, SNR {snr.min()}..{snr.max()} dB")
    return npz


@dataclass
class RadioML:
    x: np.ndarray  # (N, 2, 128) float32
    y: np.ndarray  # (N,) int64 class index
    snr: np.ndarray  # (N,) int64 dB
    classes: tuple[str, ...]


def load() -> RadioML:
    path = dataset_dir() / NPZ
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: uv run python -m rfea_ml.radioml")
    with np.load(path, allow_pickle=False) as d:
        return RadioML(d["x"], d["y"], d["snr"], tuple(str(c) for c in d["classes"]))


def split_indices(
    y: np.ndarray, snr: np.ndarray, fractions=(0.6, 0.2, 0.2), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test split stratified by (class, SNR): every pair keeps the same proportions."""
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    for c in np.unique(y):
        for s in np.unique(snr):
            idx = np.flatnonzero((y == c) & (snr == s))
            rng.shuffle(idx)
            n_train = round(len(idx) * fractions[0])
            n_val = round(len(idx) * fractions[1])
            parts[0].append(idx[:n_train])
            parts[1].append(idx[n_train : n_train + n_val])
            parts[2].append(idx[n_train + n_val :])
    train, val, test = (np.sort(np.concatenate(p)) for p in parts)
    return train, val, test


if __name__ == "__main__":
    download()

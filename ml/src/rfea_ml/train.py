"""Training loop and evaluation shared by experiments.

The whole dataset fits in GPU memory (220k x 2 x 128 float32 = 225 MB), so batches
are sliced from device tensors directly, and the input transform (e.g. the STFT)
runs on the GPU per batch.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

Transform = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 60
    patience: int = 8  # early stopping on validation loss
    seed: int = 0


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


@torch.no_grad()
def predict_logits(
    model: nn.Module, x: torch.Tensor, transform: Transform = _identity, batch_size: int = 2048
) -> torch.Tensor:
    model.eval()
    out = [model(transform(x[i : i + batch_size])) for i in range(0, len(x), batch_size)]
    return torch.cat(out)


def fit(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: TrainConfig,
    transform: Transform = _identity,
) -> list[dict]:
    """Adam + cross-entropy; keeps the weights with the lowest validation loss."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=x_train.device).manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(x_train), device=x_train.device, generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            logits = model(transform(x_train[idx]))
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)

        val_logits = predict_logits(model, x_val, transform)
        val_loss = loss_fn(val_logits, y_val).item()
        val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        rec = {
            "epoch": epoch,
            "train_loss": total / n,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.perf_counter() - t0,
        }
        history.append(rec)
        print(
            f"  epoch {epoch:3d}  train {rec['train_loss']:.4f}  val {val_loss:.4f}  "
            f"acc {val_acc:.4f}  {rec['seconds']:.1f}s",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return history


def evaluate(logits: np.ndarray, y: np.ndarray, snr: np.ndarray, classes: tuple[str, ...]) -> dict:
    """Accuracy per SNR, confusion matrices per SNR, per-class P/R/F1 (all SNR and SNR >= 10)."""
    pred = logits.argmax(1)
    labels = list(range(len(classes)))
    snrs = sorted(int(s) for s in np.unique(snr))

    def prf(mask: np.ndarray) -> dict:
        p, r, f, _ = precision_recall_fscore_support(
            y[mask], pred[mask], labels=labels, zero_division=0
        )
        return {c: {"precision": p[i], "recall": r[i], "f1": f[i]} for i, c in enumerate(classes)}

    high = snr >= 10
    return {
        "classes": list(classes),
        "accuracy_overall": float((pred == y).mean()),
        "accuracy_snr_ge_0": float((pred == y)[snr >= 0].mean()),
        "accuracy_snr_ge_10": float((pred == y)[high].mean()),
        "accuracy_by_snr": {s: float((pred == y)[snr == s].mean()) for s in snrs},
        "confusion_by_snr": {
            s: confusion_matrix(y[snr == s], pred[snr == s], labels=labels).tolist() for s in snrs
        },
        "per_class_all": prf(np.ones_like(high)),
        "per_class_snr_ge_10": prf(high),
    }

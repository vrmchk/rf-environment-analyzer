"""Temperature scaling (thesis, subsection 3.6).

p_c(T) = softmax(z / T)_c. The temperature is fitted on a validation set by minimizing
the negative log-likelihood (3.19) with the network weights fixed. Dividing all logits by
one positive number keeps their order, so the predicted class does not change.
"""

import torch
import torch.nn.functional as F


def calibrated_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """log p_c(T) for (..., M) logits."""
    return F.log_softmax(logits / temperature, dim=-1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """T* = argmin_T mean NLL of softmax(logits / T) (3.19).

    Optimized over log T (so T stays positive) with L-BFGS in float64 on the CPU.
    """
    z = logits.detach().to("cpu", torch.float64)
    y = labels.detach().to("cpu", torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())

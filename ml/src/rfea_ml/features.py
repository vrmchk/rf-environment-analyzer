"""Hand-crafted features for the SVM baseline (thesis, subsection 3.5).

For one example x[n], n = 1..N (complex IQ samples), moments are estimated by the
sample mean over its N samples:  M_pq = mean( x^(p-q) * conj(x)^q ).

Feature vector (7 values):
  1  |C40| / C21^2,  C40 = M40 - 3 M20^2
  2  |C41| / C21^2,  C41 = M41 - 3 M20 M21
  3   C42  / C21^2,  C42 = M42 - |M20|^2 - 2 M21^2   (real by definition)
  4  |C20| / C21,    C20 = M20, C21 = M21 (mean power)
  5  sigma_a   - std of the normalized centered amplitude a_cn = a / mean(a) - 1, a = |x|
  6  gamma_max - max_k |DFT(a_cn)[k]|^2 / N  (Azzouz and Nandi)
  7  sigma_f   - std of the phase increment dphi[n] = arg(x[n] * conj(x[n-1]))

Magnitudes are used for C40, C41 and C20 because the carrier frequency offset in
RadioML makes their phase random. The phase increment is used instead of the
instantaneous phase: a constant phase offset cancels in it, and a frequency offset
only shifts its mean, so no carrier recovery or phase unwrapping is needed.
"""

import numpy as np

FEATURE_NAMES = (
    "abs_c40",
    "abs_c41",
    "c42",
    "abs_c20",
    "sigma_a",
    "gamma_max",
    "sigma_f",
)


def _moment(x: np.ndarray, p: int, q: int) -> np.ndarray:
    """M_pq = E[x^(p-q) conj(x)^q], estimated per example over the last axis."""
    return np.mean(x ** (p - q) * np.conj(x) ** q, axis=-1)


def svm_features(iq: np.ndarray, batch_size: int = 20000) -> np.ndarray:
    """(B, 2, N) float IQ examples -> (B, 7) float64 feature matrix."""
    out = [_features(iq[i : i + batch_size]) for i in range(0, len(iq), batch_size)]
    return np.concatenate(out)


def _features(iq: np.ndarray) -> np.ndarray:
    x = iq[:, 0, :].astype(np.float64) + 1j * iq[:, 1, :].astype(np.float64)
    n = x.shape[-1]

    m20 = _moment(x, 2, 0)
    m21 = _moment(x, 2, 1).real  # E|x|^2, real
    m40 = _moment(x, 4, 0)
    m41 = _moment(x, 4, 1)
    m42 = _moment(x, 4, 2).real  # E|x|^4, real

    c20 = m20
    c21 = m21
    c40 = m40 - 3 * m20**2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21**2

    a = np.abs(x)
    a_cn = a / a.mean(axis=-1, keepdims=True) - 1
    sigma_a = a_cn.std(axis=-1)
    gamma_max = np.max(np.abs(np.fft.fft(a_cn, axis=-1)) ** 2, axis=-1) / n

    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1]))
    sigma_f = dphi.std(axis=-1)

    return np.stack(
        [
            np.abs(c40) / c21**2,
            np.abs(c41) / c21**2,
            c42 / c21**2,
            np.abs(c20) / c21,
            sigma_a,
            gamma_max,
            sigma_f,
        ],
        axis=1,
    )

"""Learning-rate schedules of the TaskSet optimizer families (from google-research/task_set, optimizers/adam8p.py,
nadamw.py, adam.py; read 2026-11-20), as multiples of the base learning rate, at given steps (T_TOTAL = 10k).

    adam1p, adam4p : constant
    adam6p, adam8p : max(1 - linear_decay * s, 0) * exp(-exponential_decay * s)
    nadamw         : linear warm-up over W = min(warmup_fraction, constant_fraction) * T, constant until
                     C = constant_fraction * T, then cosine decay to min_learning_rate_mult at T

Also the two "areas" of the scaling law with learning-rate annealing (Tissue et al. 2024):
    S1(s) = sum of the learning rate up to step s                    (forward area: how far training got)
    S2(s) = sum over steps of the momentum-smoothed learning-rate decrease
            m_i = lambda * m_(i-1) + max(eta_(i-1) - eta_i, 0),  S2 = sum m_i  (annealing area; the increase
            during warm-up is not counted)
"""
from __future__ import annotations

import numpy as np

T_TOTAL = 10000


def lr_multiplier(opt: str, hp: dict, steps: np.ndarray) -> np.ndarray:
    s = np.asarray(steps, float)
    if opt in ("adam1p", "adam4p"):
        return np.ones_like(s)
    if opt in ("adam6p", "adam8p"):
        return np.maximum(1 - hp["linear_decay"] * s, 0.0) * np.exp(-hp["exponential_decay"] * s)
    if opt == "nadamw":
        T = T_TOTAL
        C = hp["constant_fraction"] * T
        W = min(hp["warmup_fraction"], hp["constant_fraction"]) * T
        lo = hp["min_learning_rate_mult"]
        t = np.minimum(s, T)
        warm = np.where(W > 0, t / max(W, 1e-12), 1.0)
        cos = lo + (1 - lo) * (np.cos(np.pi * (t - C) / max(T - C, 1e-12)) / 2 + 0.5)
        return np.where(t < W, warm, np.where(t <= C, 1.0, cos))
    raise KeyError(opt)


def areas(opt: str, hp: dict, points: np.ndarray, lam: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    """S1, S2 (in units of base-learning-rate steps) at the recorded points (steps)."""
    steps = np.arange(0, int(points.max()) + 1)
    eta = lr_multiplier(opt, hp, steps)
    s1 = np.cumsum(eta)
    from scipy.signal import lfilter
    d = np.maximum(np.concatenate([[0.0], eta[:-1] - eta[1:]]), 0.0)   # decreases only (warm-up excluded)
    m = lfilter([1.0], [1.0, -lam], d)                                  # m_i = lam * m_(i-1) + d_i
    s2 = np.cumsum(m)
    idx = np.asarray(points, int)
    return s1[idx], s2[idx]

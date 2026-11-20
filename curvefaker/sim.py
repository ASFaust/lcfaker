"""Shape-invariant model of one task's curves (analysis S5): one template in log-time, warped per curve.

    y_i(u) = alpha_i + a_i * f((u - beta_i) * exp(gamma_i)),      u = log(t), t = (k + 1) / T

f is piecewise linear on KNOTS (linear extrapolation beyond), with a second-difference smoothness penalty.
Fitted by alternation: exact penalised least squares for f given the warps; grid search over (beta, gamma) per
curve with alpha and a > 0 in closed form.
"""
from __future__ import annotations

import numpy as np

KNOTS = np.linspace(-9, 4, 48)
BETAS = np.linspace(-3, 3, 41)
GAMMAS = np.linspace(-1, 1, 9)


def weights(x: np.ndarray) -> np.ndarray:
    """Linear-interpolation weights of points x [M] on KNOTS -> [M, K] (linear extrapolation)."""
    k = KNOTS
    h = k[1] - k[0]
    pos = (x - k[0]) / h
    i = np.clip(np.floor(pos).astype(int), 0, len(k) - 2)
    w = pos - i
    W = np.zeros((len(x), len(k)))
    W[np.arange(len(x)), i] = 1 - w
    W[np.arange(len(x)), i + 1] = w
    return W


def evaluate(f, alpha, amp, beta, gamma, u, x_range=None):
    """Curves [n, T] for template f and warps (arrays of length n) at log-times u. x_range = (lo, hi): hold the
    template flat outside the range of template time actually covered by fitted curves (no extrapolation)."""
    out = np.empty((len(alpha), len(u)))
    for n in range(len(alpha)):
        x = (u - beta[n]) * np.exp(gamma[n])
        if x_range is not None:
            x = np.clip(x, x_range[0], x_range[1])
        out[n] = alpha[n] + amp[n] * (weights(x) @ f)
    return out


def covered_range(beta, gamma, u):
    """Range of template time x = (u - beta) e^gamma covered by curves with these warps."""
    lo = ((u[0] - beta) * np.exp(gamma)).min()
    hi = ((u[-1] - beta) * np.exp(gamma)).max()
    return float(lo), float(hi)


def fit_task(Y: np.ndarray, mask=None, iters=8, lam=1e-2, betas=BETAS, gammas=GAMMAS):
    """Y [N, T] curves of one task. Returns dict(f, alpha, amp, beta, gamma, fitted)."""
    N, T = Y.shape
    u = np.log(np.arange(1, T + 1) / T)
    mask = np.ones(T, bool) if mask is None else mask
    y, um = Y[:, mask], u[mask]
    K = len(KNOTS)
    D = np.diff(np.eye(K), 2, axis=0)
    beta, gamma = np.zeros(N), np.zeros(N)
    alpha, amp = np.zeros(N), np.ones(N)
    f = np.interp(KNOTS, u, Y.mean(0))
    for it in range(iters):
        best = np.full(N, np.inf)
        for bb in betas:
            for gg in gammas:
                fx = weights((um - bb) * np.exp(gg)) @ f
                A = np.stack([np.ones_like(fx), fx], 1)
                coef, *_ = np.linalg.lstsq(A, y.T, rcond=None)
                coef[1] = np.maximum(coef[1], 1e-3)
                sse = ((A @ coef - y.T) ** 2).sum(0)
                better = sse < best
                best[better] = sse[better]
                beta[better], gamma[better] = bb, gg
                alpha[better], amp[better] = coef[0, better], coef[1, better]
        rows, rhs, wts = [], [], []
        for n in range(N):
            rows.append(weights((um - beta[n]) * np.exp(gamma[n])))
            rhs.append((y[n] - alpha[n]) / amp[n])
            wts.append(np.full(len(um), amp[n] ** 2))
        Wm, r, w = np.concatenate(rows), np.concatenate(rhs), np.concatenate(wts)
        f = np.linalg.solve(Wm.T @ (Wm * w[:, None]) + lam * N * D.T @ D, Wm.T @ (w * r))
    # normalise: template spans 1 -> 0 over the observed window at the median warp (amplitude / level absorb it)
    fitted = evaluate(f, alpha, amp, beta, gamma, u)
    return {"f": f, "alpha": alpha, "amp": amp, "beta": beta, "gamma": gamma, "fitted": fitted}

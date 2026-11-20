"""Analysis S5 / T1: are the curves of a task one template, moved per config? (shape-invariant model, as SITAR in
growth-curve research)

Per task, live curves (gain >= 0.1, one seed, smoothed lightly only for the gain filter), u = log-time:
    y_i(u) = alpha_i + exp(delta_i) * f((u - beta_i) * exp(gamma_i))
f: a shared template, piecewise linear on 48 knots over u in [-9, 4] with a second-difference penalty. Fitted by
alternation (exact least squares for the template, grid search for shifts / stretches with closed-form level and
amplitude); a first version fitted everything by gradient descent and got stuck (nested variants came out worse).
Variants (per-curve parameters):
    vert     alpha, amplitude                  (level and amplitude only)                         2
    shift    + beta                        (shift in log-time: speed)                         3
    sitar    + gamma                       (stretch in log-time)                              4
Reference: independent per-curve two-stage S-shape fits (logsig2, 7 parameters each; curvefaker/families.py).
Score: RMSE on held-out points (every 5th) relative to the curve's noise sd, median over curves; and the share of
the within-task variance (around the task's mean curve) explained on held-out points.

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/shape_invariant.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from curvefaker import curves, data, families
from curvefaker.paths import REPO

RNG = np.random.default_rng(14)
DEV = "cuda"
KNOTS = torch.linspace(-9, 4, 48)
PLAN = {"taskset": ("fit", 150, 30), "nasbench201": ("fit", 3, 300), "quicktune": ("fit", 86, 25),
        "jahs": ("fit", 81, 30), "fcnet": ("select", 4, 300), "bbomix": ("select", 35, 30)}


def collect(src, split, n_tasks, per):
    ids = data.ids(split, [src])
    if len(ids) > n_tasks:
        ids = list(RNG.choice(ids, n_tasks, replace=False))
    tasks = []
    for tid in ids:
        t = data.load(tid)
        y = t.y[:, 0]
        if src == "quicktune":
            y = y[np.isfinite(y[:, -1])]
        z, _ = curves.normalise(y)
        ok = np.isfinite(z).all(1)
        zc = z[ok]
        c = curves.classify(zc)
        s = curves.smooth(zc)
        live = np.where(((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1))[0]
        if len(live) < 10:
            continue
        pick = RNG.choice(live, min(per, len(live)), replace=False)
        tasks.append(zc[pick])
    n = min(len(x) for x in tasks)
    return np.stack([x[:n] for x in tasks]).astype(np.float32)        # [tasks, curves, T]


def interp(f, x):
    """Piecewise-linear template f [B, K] on KNOTS evaluated at x [B, N, T] (linear extrapolation)."""
    k = KNOTS.to(x.device)
    h = k[1] - k[0]
    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)
    pos = ((x - k[0]) / h).clamp(0, len(k) - 1 - 1e-3)
    i = pos.floor().long()
    w = pos - i
    B, N, T = x.shape
    fi = torch.gather(f, 1, i.reshape(B, -1)).reshape(B, N, T)
    fj = torch.gather(f, 1, (i + 1).reshape(B, -1)).reshape(B, N, T)
    inside = fi * (1 - w) + fj * w
    # linear extrapolation beyond the knot range
    lo = f[:, :1, None] + (f[:, 1:2, None] - f[:, :1, None]) * (x - k[0]) / h
    hi = f[:, -1:, None] + (f[:, -1:, None] - f[:, -2:-1, None]) * (x - k[-1]) / h
    return torch.where(x < k[0], lo, torch.where(x > k[-1], hi, inside))


def _weights(x):
    """Linear-interpolation weights of points x [M] on KNOTS -> dense [M, K] (linear extrapolation)."""
    k = KNOTS.numpy().astype(float)
    h = k[1] - k[0]
    pos = (x - k[0]) / h
    i = np.clip(np.floor(pos).astype(int), 0, len(k) - 2)
    w = pos - i
    W = np.zeros((len(x), len(k)))
    W[np.arange(len(x)), i] = 1 - w
    W[np.arange(len(x)), i + 1] = w
    return W


def fit_sim(Y, mask, variant, iters=8, lam=1e-2):
    """Alternating fit per task. Template step: penalised least squares for the knot values given the warps.
    Warp step: per curve, grid search over shift beta (and stretch gamma), with level alpha and amplitude a > 0 in
    closed form. Starts from beta = gamma = 0, so each variant is at least as good in-sample as the one it nests."""
    B, N, T = Y.shape
    u = np.log(np.arange(1, T + 1) / T)
    K = len(KNOTS)
    D = np.diff(np.eye(K), 2, axis=0)
    betas = np.linspace(-3, 3, 41) if variant in ("shift", "sitar") else np.array([0.0])
    gammas = np.linspace(-1, 1, 9) if variant == "sitar" else np.array([0.0])
    out = np.zeros_like(Y)
    for b in range(B):
        y = Y[b][:, mask]
        um = u[mask]
        beta, gamma = np.zeros(N), np.zeros(N)
        alpha, amp = np.zeros(N), np.ones(N)
        f = np.interp(KNOTS.numpy(), u, Y[b].mean(0))
        for it in range(iters):
            # warp step (alpha, amp closed form per (beta, gamma))
            if it > 0 or variant == "vert":
                best = np.full(N, np.inf)
                for bb in betas:
                    for gg in gammas:
                        fx = _weights((um - bb) * np.exp(gg)) @ f                    # [M]
                        A = np.stack([np.ones_like(fx), fx], 1)
                        coef, *_ = np.linalg.lstsq(A, y.T, rcond=None)               # [2, N]
                        coef[1] = np.maximum(coef[1], 1e-3)
                        sse = ((A @ coef - y.T) ** 2).sum(0)
                        better = sse < best
                        best[better] = sse[better]
                        beta[better], gamma[better] = bb, gg
                        alpha[better], amp[better] = coef[0, better], coef[1, better]
            # template step: (y - alpha) / amp = W_i f, weighted by amp^2
            rows, rhs, wts = [], [], []
            for n in range(N):
                rows.append(_weights((um - beta[n]) * np.exp(gamma[n])))
                rhs.append((y[n] - alpha[n]) / amp[n])
                wts.append(np.full(len(um), amp[n] ** 2))
            Wm, r, w = np.concatenate(rows), np.concatenate(rhs), np.concatenate(wts)
            f = np.linalg.solve(Wm.T @ (Wm * w[:, None]) + lam * N * D.T @ D, Wm.T @ (w * r))
        for n in range(N):
            out[b, n] = alpha[n] + amp[n] * (_weights((u - beta[n]) * np.exp(gamma[n])) @ f)
    return out


def main():
    rows = []
    for src, (split, n_tasks, per) in PLAN.items():
        Y = collect(src, split, n_tasks, per)
        B, N, T = Y.shape
        mask = np.arange(T) % 5 != 2
        sd = np.maximum(families.noise_sd(Y.reshape(-1, T)), 1e-4).reshape(B, N)
        within = ((Y - Y.mean(1, keepdims=True))[..., ~mask] ** 2).sum()
        res = {}
        for v in ("vert", "shift", "sitar"):
            res[v] = fit_sim(Y, mask, v)
        f_ref, _ = families.fit("logsig2", Y.reshape(-1, T), mask)
        res["logsig2 (7/curve)"] = f_ref.reshape(B, N, T)
        for name, P in res.items():
            r = np.sqrt(((P - Y)[..., ~mask] ** 2).mean(-1)) / sd
            expl = 1 - ((P - Y)[..., ~mask] ** 2).sum() / within
            rows.append({"source": src, "tasks": B, "curves_per_task": N, "model": name,
                         "rel_rmse_med": float(np.median(r)), "within_var_explained": float(expl)})
        print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "shape_invariant.csv", index=False)


if __name__ == "__main__":
    main()

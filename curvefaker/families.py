"""Parametric curve families fitted per curve, batched (analysis S1 of docs/analysis_plan.md).

Time: point k of a curve with T points is at t = (k + 1) / T in (0, 1]; u = log t. Curves are normalised z (lower is
better). All families are fitted by least squares on a subset of points (`mask`).

Nonlinear families (Adam, several initialisations per curve, best kept):
    exp3     c + a exp(-b t)
    pow3     c + a t^(-alpha)
    weibull  c + a exp(-(t / s)^delta)                    (stretched / compressed exponential, 4 params)
    logsig   c + a sigmoid(-k (u - m))                    (S-shape in log-time: plateau, descent, floor; ifBO's
                                                           "hill" basis curve is this shape)
    logsig2  c + a1 sigmoid(-k1 (u - m1)) + a2 sigmoid(-k2 (u - m2))      (two stages, 7 params)
Linear-in-parameters families (exact least squares; knots by grid search):
    loglin   c + a u                                       (pure power law without floor, in log-time: a line)
    hinge1   continuous piecewise-linear in u with 1 knee  (4 params incl. the knot)
    hinge2   ... with 2 knees                              (6 params)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

NONLIN = {"exp3": 3, "pow3": 3, "weibull": 4, "logsig": 4, "logsig2": 7}
LIN = {"loglin": 2, "hinge1": 4, "hinge2": 6}
FAMILIES = {**NONLIN, **LIN}


def _curve(kind, p, t, u):
    sg = torch.sigmoid
    if kind == "exp3":
        c, a, lb = p.unbind(-1)
        return c[:, None] + a[:, None] * torch.exp(-torch.exp(lb)[:, None] * t)
    if kind == "pow3":
        c, a, la = p.unbind(-1)
        return c[:, None] + a[:, None] * t ** (-torch.exp(la)[:, None])
    if kind == "weibull":
        c, a, ls, ld = p.unbind(-1)
        return c[:, None] + a[:, None] * torch.exp(-(t / torch.exp(ls)[:, None]) ** torch.exp(ld)[:, None])
    if kind == "logsig":
        c, a, lk, m = p.unbind(-1)
        return c[:, None] + a[:, None] * sg(-torch.exp(lk)[:, None] * (u - m[:, None]))
    if kind == "logsig2":
        c, a1, lk1, m1, a2, lk2, m2 = p.unbind(-1)
        return (c[:, None] + a1[:, None] * sg(-torch.exp(lk1)[:, None] * (u - m1[:, None]))
                + a2[:, None] * sg(-torch.exp(lk2)[:, None] * (u - m2[:, None])))
    raise KeyError(kind)


def _inits(kind, y, u):
    """List of initial parameter tensors [n, p] (several starts)."""
    n = len(y)
    first, last = y[:, 0], y[:, -1]
    a0 = first - last
    umin = float(u[0])
    full = lambda *vals: torch.stack([v if torch.is_tensor(v) else torch.full((n,), float(v), device=y.device)
                                      for v in vals], -1)
    if kind == "exp3":
        return [full(last, a0, math.log(b)) for b in (1, 5, 25)]
    if kind == "pow3":
        return [full(last, a0 * (1 / len(u)) ** al, math.log(al)) for al in (0.2, 0.5, 1.0)]
    if kind == "weibull":
        return [full(last, a0, math.log(s), math.log(d)) for s in (0.05, 0.3) for d in (0.5, 1.5)]
    if kind == "logsig":
        return [full(last, a0, math.log(k), umin * q) for k in (1, 4) for q in (0.75, 0.5, 0.25)]
    if kind == "logsig2":
        return [full(last, a0 / 2, math.log(3), umin * q1, a0 / 2, math.log(3), umin * q2)
                for q1, q2 in ((0.8, 0.4), (0.6, 0.2), (0.9, 0.1))]
    raise KeyError(kind)


def fit_nonlin(kind, Y, mask, steps=1500, lr=0.03, device="cuda"):
    """Y [n, T] (numpy), mask [T] bool (points used for fitting). Returns fitted curves [n, T] and train SSE [n]."""
    y = torch.tensor(Y, dtype=torch.float32, device=device)
    T = y.shape[1]
    t = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
    u = torch.log(t)
    m = torch.tensor(mask, device=device)
    ym = y[:, m]
    best_sse = torch.full((len(y),), float("inf"), device=device)
    best_fit = torch.zeros_like(y)
    for p0 in _inits(kind, y, u):
        p = torch.nn.Parameter(p0.clone())
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            f = _curve(kind, p, t[None], u[None])
            loss = ((f[:, m] - ym) ** 2).sum(1)
            loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.data = torch.nan_to_num(p.data).clamp(-30, 30)
        with torch.no_grad():
            f = _curve(kind, p, t[None], u[None])
            sse = ((f[:, m] - ym) ** 2).sum(1)
            sse = torch.where(torch.isfinite(sse), sse, torch.full_like(sse, float("inf")))
            better = sse < best_sse
            best_sse = torch.where(better, sse, best_sse)
            best_fit[better] = f[better]
    return best_fit.cpu().numpy(), best_sse.cpu().numpy()


def fit_lin(kind, Y, mask, n_knots=25):
    """Exact least squares; knots on a grid of u within the fitted range. Returns fitted curves, SSE, knots."""
    T = Y.shape[1]
    u = np.log(np.arange(1, T + 1) / T)
    um = u[mask]
    grid = np.linspace(um.min(), um.max(), n_knots + 2)[1:-1]
    combos = {"loglin": [()], "hinge1": [(k,) for k in grid],
              "hinge2": list(itertools.combinations(grid, 2))}[kind]
    best_sse = np.full(len(Y), np.inf)
    best_fit = np.zeros_like(Y)
    best_k = np.full((len(Y), max(1, len(combos[0]))), np.nan)
    for ks in combos:
        cols = [np.ones(T), u] + [np.maximum(0, u - k) for k in ks]
        A = np.stack(cols, 1)                                  # [T, p]
        coef, *_ = np.linalg.lstsq(A[mask], Y[:, mask].T, rcond=None)
        f = (A @ coef).T
        sse = ((f[:, mask] - Y[:, mask]) ** 2).sum(1)
        better = sse < best_sse
        best_sse[better] = sse[better]
        best_fit[better] = f[better]
        if ks:
            best_k[better] = np.array(ks)[None]
    return best_fit, best_sse, best_k


def fit(kind, Y, mask, **kw):
    if kind in LIN:
        f, s, _ = fit_lin(kind, Y, mask)
        return f, s
    return fit_nonlin(kind, Y, mask, **kw)


def noise_sd(Y):
    """Robust per-curve noise sd: MAD of first differences / sqrt 2."""
    d = np.diff(Y, axis=1)
    return np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1) * 1.4826 / math.sqrt(2)

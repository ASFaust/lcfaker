"""Config -> performance maps ("landscapes") of single tasks, for question A.

Encoding: every hyperparameter is one input group. Numeric: one column in [0, 1] (log first where flagged).
Categorical: integer codes, compared only for equality.

Gaussian-process models on a target y (e.g. final normalised loss), fitted by type-II maximum likelihood:
    full      : s^2 * prod_j k_j(x_j, x'_j)                       (every interaction allowed)
    additive  : sum_j s_j^2 k_j(x_j, x'_j)                        (no interactions)
    add+full  : additive + full                                    (interaction share = s_full^2 / total)
with k_j = Matern-5/2 with its own length-scale for a numeric hyperparameter, and exp(-theta_j [a != b]) for a
categorical one. Plus a constant mean and white noise.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch


def encode(hps: pd.DataFrame, space: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> X [n, g] (numeric in [0, 1], categorical as codes), is_cat [g]. Constant hyperparameters dropped."""
    cols, cat = [], []
    for name, s in space.items():
        c = hps[name]
        if s["type"] == "cat":
            codes, uniq = pd.factorize(c.astype(str), sort=True)
            if len(uniq) < 2:
                continue
            cols.append(codes.astype(np.float64))
            cat.append(True)
        else:
            v = c.to_numpy(np.float64)
            if s.get("log") and np.nanmin(v) > 0:
                v = np.log(v)
            lo, hi = np.nanmin(v), np.nanmax(v)
            if not hi > lo:
                continue
            cols.append(np.nan_to_num((v - lo) / (hi - lo), nan=0.5))
            cat.append(False)
    return np.stack(cols, 1), np.array(cat)


def _group_kernels(X1, X2, is_cat, log_ell, log_theta):
    """[g, n1, n2] per-group kernel matrices."""
    d = X1.T[:, :, None] - X2.T[:, None, :]                     # [g, n1, n2]
    ell = torch.exp(log_ell)[:, None, None]
    r = math.sqrt(5.0) * d.abs() / ell
    mat = (1 + r + r * r / 3) * torch.exp(-r)
    catk = torch.exp(-torch.exp(log_theta)[:, None, None] * (d != 0).double())
    return torch.where(is_cat[:, None, None], catk, mat)


class LandscapeGP(torch.nn.Module):
    def __init__(self, g: int, is_cat: np.ndarray, kind: str, device="cpu"):
        super().__init__()
        assert kind in ("full", "additive", "add+full")
        self.kind = kind
        P = lambda v, n=None: torch.nn.Parameter(torch.full((n,) if n else (), float(v), dtype=torch.float64,
                                                            device=device))
        self.is_cat = torch.tensor(is_cat, device=device)
        self.log_ell_f, self.log_th_f = P(math.log(0.5), g), P(0.0, g)
        self.log_ell_a, self.log_th_a = P(math.log(0.5), g), P(0.0, g)
        self.log_s_add = P(math.log(0.3 / math.sqrt(g)), g)
        self.log_s_full = P(math.log(0.3))
        self.log_noise = P(math.log(0.05))
        self.mean = P(0.0)

    def kernel(self, X1, X2):
        K = 0.0
        if self.kind in ("full", "add+full"):
            Kg = _group_kernels(X1, X2, self.is_cat, self.log_ell_f, self.log_th_f)
            K = K + torch.exp(2 * self.log_s_full) * Kg.prod(0)
        if self.kind in ("additive", "add+full"):
            Kg = _group_kernels(X1, X2, self.is_cat, self.log_ell_a, self.log_th_a)
            K = K + (torch.exp(2 * self.log_s_add)[:, None, None] * Kg).sum(0)
        return K

    def nll(self, X, y):
        n = len(y)
        K = self.kernel(X, X) + (torch.exp(2 * self.log_noise) + 1e-6) * torch.eye(n, dtype=X.dtype, device=X.device)
        L = torch.linalg.cholesky(K)
        r = (y - self.mean)[:, None]
        a = torch.cholesky_solve(r, L)
        return 0.5 * (r * a).sum() + torch.log(torch.diagonal(L)).sum() + 0.5 * n * math.log(2 * math.pi)

    def fit(self, X, y, steps=300, lr=0.05):
        with torch.no_grad():
            self.mean.fill_(float(y.mean()))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = self.nll(X, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.log_ell_f.clamp_(math.log(0.01), math.log(20))
                self.log_ell_a.clamp_(math.log(0.01), math.log(20))
                self.log_noise.clamp_(math.log(0.03), math.log(2))
        self.X, self.y = X, y
        return self

    @torch.no_grad()
    def predict(self, Xs):
        n = len(self.y)
        K = self.kernel(self.X, self.X) + (torch.exp(2 * self.log_noise) + 1e-6) * torch.eye(
            n, dtype=self.X.dtype, device=self.X.device)
        a = torch.linalg.solve(K, (self.y - self.mean)[:, None])
        return (self.kernel(Xs, self.X) @ a)[:, 0] + self.mean

    @torch.no_grad()
    def describe(self) -> dict:
        """Interaction share (add+full) and the effective number of hyperparameters that matter (additive part:
        exp of the entropy of the variance shares s_j^2 / sum s^2)."""
        out = {}
        if self.kind in ("additive", "add+full"):
            v = torch.exp(2 * self.log_s_add)
            p = (v / v.sum()).cpu().numpy()
            out["eff_hps"] = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
            out["top_hp_share"] = float(p.max())
            if self.kind == "add+full":
                f = float(torch.exp(2 * self.log_s_full))
                out["interaction_share"] = f / (f + float(v.sum()))
        out["noise_share"] = float(torch.exp(2 * self.log_noise)) / float(self.y.var())
        return out

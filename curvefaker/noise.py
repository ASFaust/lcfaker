"""Gaussian-process noise models for single learning curves (question C).

A curve y[t], t = 1..T (normalised z, one seed) is modelled as

    y = c + f(t) + e(t),   f ~ GP(0, k_trend),   e ~ GP(0, k_noise)

The trend kernel is the same for every noise model: Matern-5/2 in log-time plus Matern-5/2 in linear time (fast
early drops and late, schedule-driven drops can both be smooth in one of the two). Its parameters (two variances,
two length-scales), the offset c and the noise scale sigma are fitted per curve.

Noise models (shape parameters are global: one value per source):
    white      e_t iid N(0, sigma^2)
    ar1        AR(1): Cov(e_s, e_t) = sigma^2 rho^|s-t|, rho in (-1, 1)
    phase      white, sd sigma * exp(beta * (u_t - 0.5)), u_t = log t / log T  (noise grows / shrinks in training)
    ar1+phase  both

Comparison: choose the global shape parameters on one set of curves (grid search: for every grid value the per-curve
parameters are fitted and the summed likelihood compared; joint gradient fitting of global and per-curve parameters
was tried first and ended in different local optima for equivalent models), freeze them, re-fit the per-curve
parameters on held-out curves and compare log marginal likelihoods there (nats per point). All models have the same number of
per-curve parameters, so the held-out comparison is fair.
"""
from __future__ import annotations

import math

import torch

MODELS = ("white", "ar1", "phase", "ar1+phase")


def _matern52(d, ell):
    r = math.sqrt(5.0) * d / ell
    return (1 + r + r * r / 3) * torch.exp(-r)


class CurveGP(torch.nn.Module):
    def __init__(self, n_curves: int, T: int, model: str, device):
        super().__init__()
        assert model in MODELS
        self.model, self.T = model, T
        z = lambda v: torch.nn.Parameter(torch.full((n_curves,), float(v), device=device))
        self.log_s1, self.log_l1 = z(math.log(0.3)), z(math.log(0.3))
        self.log_s2, self.log_l2 = z(math.log(0.1)), z(math.log(0.3))
        self.log_sig, self.c = z(math.log(0.03)), z(0.0)
        self.a_rho = torch.nn.Parameter(torch.zeros((), device=device))      # rho = 0.99 tanh(a)
        self.beta = torch.nn.Parameter(torch.zeros((), device=device))
        t = torch.arange(1, T + 1, device=device, dtype=torch.float64)
        u = torch.log(t) / math.log(T)
        v = (t - 1) / (T - 1)
        self.register_buffer("du", (u[:, None] - u[None]).abs())
        self.register_buffer("dv", (v[:, None] - v[None]).abs())
        self.register_buffer("lag", (t[:, None] - t[None]).abs())
        self.register_buffer("u", u)

    def global_params(self):
        return [self.a_rho, self.beta]

    def local_params(self):
        return [self.log_s1, self.log_l1, self.log_s2, self.log_l2, self.log_sig, self.c]

    def rho(self):
        return 0.99 * torch.tanh(self.a_rho) if "ar1" in self.model else torch.zeros((), device=self.a_rho.device)

    def cov(self):
        e = lambda p: torch.exp(p.double())[:, None, None]
        K = e(2 * self.log_s1) * _matern52(self.du[None], e(self.log_l1)) \
            + e(2 * self.log_s2) * _matern52(self.dv[None], e(self.log_l2))
        sd = torch.exp(self.log_sig.double())[:, None].expand(-1, self.T)
        if "phase" in self.model:
            sd = sd * torch.exp(self.beta.double() * (self.u - 0.5))[None]
        R = self.rho().double() ** self.lag if "ar1" in self.model else torch.eye(self.T, device=sd.device,
                                                                                    dtype=sd.dtype)
        N = sd[:, :, None] * R[None] * sd[:, None, :]
        return K + N, N

    def loglik(self, y):
        """Log marginal likelihood per curve, y [n, T] float64."""
        K, _ = self.cov()
        K = K + 1e-8 * torch.eye(self.T, device=K.device, dtype=K.dtype)
        L = torch.linalg.cholesky(K)
        r = (y - self.c.double()[:, None])[..., None]
        a = torch.cholesky_solve(r, L)
        return (-0.5 * (r * a).sum((1, 2)) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
                - 0.5 * self.T * math.log(2 * math.pi))

    def residuals(self, y):
        """Posterior mean of the noise e given y (y minus the smoothed trend), [n, T]."""
        K, N = self.cov()
        K = K + 1e-8 * torch.eye(self.T, device=K.device, dtype=K.dtype)
        r = (y - self.c.double()[:, None])[..., None]
        return (N @ torch.linalg.solve(K, r))[..., 0]


def fit(y: torch.Tensor, model: str, steps=400, lr=0.05, frozen: dict | None = None):
    """Fit per-curve parameters (and global ones unless `frozen` gives them). Returns (gp, loglik per curve)."""
    gp = CurveGP(y.shape[0], y.shape[1], model, y.device)
    with torch.no_grad():
        gp.c.copy_(y.mean(1).float())
        if frozen:
            gp.a_rho.fill_(frozen["a_rho"])
            gp.beta.fill_(frozen["beta"])
    params = gp.local_params() + ([] if frozen else gp.global_params())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -gp.loglik(y).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            gp.log_l1.clamp_(math.log(0.02), math.log(5.0))
            gp.log_l2.clamp_(math.log(0.02), math.log(5.0))
            gp.log_sig.clamp_(math.log(1e-4), math.log(2.0))
    with torch.no_grad():
        ll = gp.loglik(y)
    return gp, ll


RHO_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9)
BETA_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _a(rho):
    return math.atanh(rho / 0.99)


def select_global(y: torch.Tensor, model: str, **kw) -> tuple[dict, float]:
    """Grid search of the global shape parameters on curves y; returns (frozen params, mean loglik per point).
    ar1+phase: coordinate search (best rho at beta = 0, then best beta at that rho, then rho again)."""
    T = y.shape[1]

    def score(rho, beta):
        _, ll = fit(y, model, frozen={"a_rho": _a(rho), "beta": beta}, **kw)
        return float(ll.mean()) / T

    if model == "white":
        return {"a_rho": 0.0, "beta": 0.0}, score(0.0, 0.0)
    if model == "ar1":
        res = {r: score(r, 0.0) for r in RHO_GRID}
        r = max(res, key=res.get)
        return {"a_rho": _a(r), "beta": 0.0}, res[r]
    if model == "phase":
        res = {b: score(0.0, b) for b in BETA_GRID}
        b = max(res, key=res.get)
        return {"a_rho": 0.0, "beta": b}, res[b]
    rr = {r: score(r, 0.0) for r in RHO_GRID}
    r = max(rr, key=rr.get)
    bb = {b: score(r, b) for b in BETA_GRID}
    b = max(bb, key=bb.get)
    rr = {r2: score(r2, b) for r2 in RHO_GRID}
    r = max(rr, key=rr.get)
    return {"a_rho": _a(r), "beta": b}, rr[r]


def frozen_of(gp: CurveGP) -> dict:
    return {"a_rho": float(gp.a_rho.detach()), "beta": float(gp.beta.detach())}

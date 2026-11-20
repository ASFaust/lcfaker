"""swt (shape-warp-template): a data-driven prior of learning curves for freeze-thaw HPO.

Each generated task is a search space ([0, 1]^d, d <= 10), a set of configurations and, for every configuration, a
learning curve of a normalised loss (lower is better; 0 ~ the task's best, 1 ~ a typical starting loss, 2 = diverged).

How a task is generated (details in the paper, paper/main.tex):
  1. A record is drawn: the fitted description of one real HPO task (a template curve in log-time, the joint
     distribution of per-curve warps, dead / diverged shares, noise level, profile and tail weight).
  2. Smooth random Gaussian-process fields over the configurations decide each configuration's final value
     (with a spatially rough part matched to the real task's predictability) and its speed.
  3. Each configuration's curve is the record's template, warped:  y(u) = alpha + a * f((u - beta) * exp(gamma)),
     u = log(t / T): beta shifts the descent in log-time, gamma stretches it, a and alpha set amplitude and level.
  4. Dead configurations (flat) sit at the low end of a learning-rate-like first dimension, diverged ones at its
     high end; Student-t noise shrinking over the budget is added.

Only numpy and scipy are needed. Everything learned from real data is in assets/swt_v11.npz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ASSET = Path(__file__).resolve().parent / "assets" / "swt_v11.npz"
CAP = 2.0                                   # value of diverged curves
KNOTS = np.linspace(-9, 4, 48)              # template knots in log-time
BETA_STEP = 6.0 / 40                        # grid step of the fitted log-time shifts
ROUGH_SCALE = 0.15
LR_AXIS_WEIGHT = 0.7
N_FIELDS = 6

_RECORDS: dict | None = None


@dataclass
class Task:
    """One generated HPO task.

    y      : [n_configs, T] float32, normalised loss curves (lower is better)
    X      : [n_configs, d] configurations in [0, 1]^d (dimension 0 acts like a learning rate)
    source : benchmark family of the record the task was generated from (template and parameter distributions)
    """
    y: np.ndarray
    X: np.ndarray
    source: str
    info: dict = field(default_factory=dict)


def _records() -> dict:
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = dict(np.load(ASSET, allow_pickle=False))
    return _RECORDS


def _weights(x):
    """Linear-interpolation weights of points x [M] on KNOTS -> [M, K]."""
    h = KNOTS[1] - KNOTS[0]
    pos = (x - KNOTS[0]) / h
    i = np.clip(np.floor(pos).astype(int), 0, len(KNOTS) - 2)
    w = pos - i
    W = np.zeros((len(x), len(KNOTS)))
    W[np.arange(len(x)), i] = 1 - w
    W[np.arange(len(x)), i + 1] = w
    return W


def _evaluate(f, alpha, amp, beta, gamma, u, x_range):
    """Warped template curves [n, len(u)]; the template is held flat outside x_range (never extrapolated)."""
    out = np.empty((len(alpha), len(u)))
    for k in range(len(alpha)):
        x = np.clip((u - beta[k]) * np.exp(gamma[k]), x_range[0], x_range[1])
        out[k] = alpha[k] + amp[k] * (_weights(x) @ f)
    return out


def _matern52(d, ell):
    r = math.sqrt(5) * np.abs(d) / ell
    return (1 + r + r * r / 3) * np.exp(-r)


def _fields(rng, X, n_fields):
    """Smooth GP fields with a random Matern-5/2 kernel (random relevances, random additive / product mix) and
    rough companions (length-scales x ROUGH_SCALE). Each standardised."""
    n, d = X.shape
    r = rng.dirichlet(np.full(d, 0.7))
    ell = math.exp(rng.uniform(math.log(0.2), 0.0)) / np.sqrt(r / r.max())
    w = rng.random()
    out = []
    for scale in (1.0, ROUGH_SCALE):
        Kg = np.stack([_matern52(X[:, None, j] - X[None, :, j], ell[j] * scale) for j in range(d)])
        Kx = w * np.tensordot(r, Kg, 1) + (1 - w) * Kg.prod(0)
        f = np.linalg.cholesky(Kx + 1e-6 * np.eye(n)) @ rng.standard_normal((n, n_fields))
        out.append((f - f.mean(0)) / f.std(0))
    return out


def _noise_profile(thirds, T):
    return np.interp(np.linspace(0, 1, T), np.array([1 / 6, 1 / 2, 5 / 6]), thirds)


def sample_task(rng: np.random.Generator, R: dict | None = None) -> Task:
    """Generate one task."""
    R = R or _records()
    i = int(rng.choice(len(R["weight"]), p=R["weight"]))
    T, n = int(R["T"][i]), int(R["n"][i])
    d = int(np.clip(R["n_hps"][i], 1, 10))
    f, xr = R["f"][i], R["x_range"][i]
    X = rng.random((n, d))

    # landscape fields: 0 = final value (with the rough part), 1 = speed, 4 / 5 = dead / diverged regions
    eta = float(R["eta"][i])
    sm, ro = _fields(rng, X, N_FIELDS)
    g = sm.copy()
    g[:, :1] = math.sqrt(1 - eta) * sm[:, :1] + math.sqrt(eta) * ro[:, :1]

    # warps
    Wi = R["warps"][R["warp_offs"][i]:R["warp_offs"][i + 1]]
    u = np.log(np.arange(1, T + 1) / T)
    rng.integers(len(Wi), size=n)   # (kept for random-stream compatibility with the development code)
    fin_all = _evaluate(f, Wi[:, 0], np.exp(Wi[:, 1]), Wi[:, 2], Wi[:, 3], u[-2:], xr)[:, -1]
    zs = norm.ppf(np.stack([rankdata(fin_all), rankdata(Wi[:, 2])], 1) / (len(Wi) + 1))
    rho = float(np.corrcoef(zs.T)[0, 1]) if len(Wi) > 2 else 0.0
    tgt = np.stack([g[:, 0], rho * g[:, 0] + math.sqrt(max(1 - rho ** 2, 0)) * g[:, 1]], 1)
    tgt = (tgt - tgt.mean(0)) / tgt.std(0)
    d2 = 4.0 * (tgt[:, None, 0] - zs[None, :, 0]) ** 2 + (tgt[:, None, 1] - zs[None, :, 1]) ** 2
    warps = Wi[d2.argmin(1)].copy()          # nearest real vector: amplitude and stretch
    q = norm.cdf(tgt)
    fin_t = np.quantile(fin_all, q[:, 0])    # final value and speed: quantiles of the real marginals
    warps[:, 2] = np.quantile(Wi[:, 2], q[:, 1]) + rng.uniform(-BETA_STEP / 2, BETA_STEP / 2, n)
    end = _evaluate(f, np.zeros(n), np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u[-2:], xr)[:, -1]
    matched = warps.copy()
    warps[:, 0] = fin_t - end                # level solved: the noise-free curve ends at the target final value
    lo = _evaluate(f, Wi[:, 0], np.exp(Wi[:, 1]), Wi[:, 2], Wi[:, 3], u, xr).min()
    yc = _evaluate(f, warps[:, 0], np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u, xr)
    bad = yc.min(1) < lo - 0.02              # never below the lowest fitted real curve
    warps[bad] = matched[bad]
    y = _evaluate(f, warps[:, 0], np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u, xr)

    # classes along the learning-rate-like axis
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]

    # noise; dead level in the task's own normalised units
    sd = R["noise_sd_diff"][i] * _noise_profile(R["noise_thirds_diff"][i], T)
    noise = rng.standard_t(R["df"][i], (n, T)) * sd
    y[dead] = R["dead_level"][i]
    y = y + noise
    if len(dead):
        live_rows = np.setdiff1d(np.arange(n), dead)
        for _ in range(2):
            b = min(y[live_rows].min(), 0.0) if len(live_rows) else 0.0
            r = np.quantile(y[:, 0], 0.9)
            y[dead] = b + R["dead_level"][i] * (r - b) + noise[dead]
    dt = R["div_times"][R["div_offs"][i]:R["div_offs"][i + 1]]
    for a in div:
        t0 = rng.choice(dt) if len(dt) else rng.random()
        y[a, int(round(t0 * (T - 1))):] = CAP
    y = np.clip(y, -0.5, CAP).astype(np.float32)
    return Task(y=y, X=X, source=str(R["source"][i]),
                info={"dead": np.sort(dead), "diverged": np.sort(div), "record": i})


def generate(n_tasks: int, seed: int | np.random.Generator = 0, source_weights: dict | None = None) -> list[Task]:
    """Generate n_tasks tasks. Deterministic for a given integer seed.

    source_weights: optional {source: weight} over the record families ("taskset", "jahs", "quicktune",
    "nasbench201"). Default: in proportion to the number of real fit tasks per family (96% TaskSet), which is the
    setting evaluated in the paper.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    R = _records()
    if source_weights is not None:
        R = dict(R)
        src = R["source"]
        unknown = set(source_weights) - set(np.unique(src))
        if unknown:
            raise ValueError(f"unknown sources {unknown}; available: {sorted(set(src))}")
        w = np.zeros(len(src))
        for s, v in source_weights.items():
            m = (src == s) & (R["weight"] > 0)
            w[m] = v / m.sum()
        R["weight"] = w / w.sum()
    return [sample_task(rng, R) for _ in range(n_tasks)]

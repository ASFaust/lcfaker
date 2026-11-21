"""bpl: a data-driven prior of learning curves for freeze-thaw HPO (version bpl-v16).

Each generated task is a search space ([0, 1]^d, d <= 10), a set of configurations and, for every configuration, a
validation learning curve in raw units (a loss, or an error rate for records from error-rate benchmarks).

How a task is generated (details in the paper, paper/main.pdf):
  1. A record is drawn: the fitted description of one real HPO task (floor and start level, the joint distribution
     of per-curve broken-power-law parameters, their relation to the learning rate, schedule hyperparameters,
     shares of dead / diverged / rising curves, noise level, profile, growth with the loss and tail weight).
  2. Configurations are uniform in [0, 1]^d. Dimension 0 is the log learning rate; for TaskSet records the next
     dimensions are the schedule hyperparameters of the record's optimiser family.
  3. Each configuration's time axis is optimiser progress, x = (lr / lr_ref)^kappa * S1(t) / T_total, with S1 the
     cumulative learning-rate multiplier of its schedule.
  4. Smooth Gaussian-process fields give each configuration the four parameters of
         y = L + d + A / (1 + (x / tau)^a)
     through a Gaussian copula with the record's real marginals, correlations and learning-rate relation.
  5. Rising curves (overfitting / instability), dead configurations (low learning rates), diverged ones (high
     learning rates) and heavy-tailed noise whose size scales with the excess loss are added.

Only numpy and scipy are needed. Everything learned from real data is in assets/bpl_v16.npz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

ASSET = Path(__file__).resolve().parent / "assets" / "bpl_v16.npz"
VERSION = "bpl-v16"
N_FIELDS = 7
ROUGH_SCALE = 0.15
LR_AXIS_WEIGHT = 0.7
T_TOTAL = 10000
SCHED_HPS = {"adam6p": ["linear_decay", "exponential_decay"], "adam8p": ["linear_decay", "exponential_decay"],
             "nadamw": ["warmup_fraction", "constant_fraction", "min_learning_rate_mult"]}
CAP = 2.0

_RECORDS: dict | None = None


@dataclass
class Task:
    """One generated HPO task.

    y       : [n_configs, T] float32, validation curves in raw units, lower is better. TaskSet-like tasks include the
              value at step 0 (before training) as the first point.
    X       : [n_configs, d] configurations in [0, 1]^d. X[:, 0] maps to the log learning rate (info["log_lr"]).
    metric  : "loss" (unbounded) or "error" (bounded error rate: in % for JAHS / QuickTune records, a fraction for
              NAS-Bench-201 records)
    source  : benchmark family of the record the task was generated from
    info    : dead / diverged / rising configuration indices, log learning rates, schedule family, record id
    """
    y: np.ndarray
    X: np.ndarray
    metric: str
    source: str
    info: dict = field(default_factory=dict)

    def normalised(self) -> np.ndarray:
        """Curves mapped to z = (y - best) / (ref - best), ref = 90th percentile of first values: 0 ~ the task's
        best, 1 ~ a typical start, capped at 2 (diverged). This is the scale used for the statistics in the paper."""
        y = self.y.astype(float)
        best = np.nanmin(y)
        ref = np.nanquantile(y[:, 0], 0.9)
        z = (y - best) / max(ref - best, 1e-12)
        return np.clip(np.nan_to_num(z, nan=CAP), None, CAP)


def _records() -> dict:
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = dict(np.load(ASSET, allow_pickle=False))
    return _RECORDS


def _seg(R, k, i):
    return R[k][R[k + "_offs"][i]:R[k + "_offs"][i + 1]]


def lr_multiplier(opt: str, hp: dict, steps: np.ndarray) -> np.ndarray:
    """Learning-rate multiplier of the TaskSet optimiser families' schedules (T_TOTAL steps)."""
    s = np.asarray(steps, float)
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
    return np.ones_like(s)


def progress(opt_id: int, sched: dict, frac: np.ndarray) -> np.ndarray:
    """Cumulative learning-rate multiplier S1 / T_TOTAL at budget fractions frac. opt_id: 0 constant, 1 linear /
    exponential decay, 2 warm-up / constant / cosine, 3 cosine to 0."""
    steps = np.arange(0, T_TOTAL + 1)
    if opt_id == 0:
        m = np.ones_like(steps, float)
    elif opt_id == 1:
        m = lr_multiplier("adam6p", sched, steps)
    elif opt_id == 2:
        m = lr_multiplier("nadamw", sched, steps)
    else:
        m = 0.5 * (1 + np.cos(np.pi * steps / T_TOTAL))
    s1 = np.cumsum(m) / T_TOTAL
    idx = np.clip(np.round(np.asarray(frac) * T_TOTAL).astype(int), 1, T_TOTAL)
    return np.maximum(s1[idx], 1e-9)


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


def sample_task(rng: np.random.Generator, R: dict | None = None) -> Task:
    """Generate one task."""
    R = R or _records()
    i = int(rng.choice(len(R["weight"]), p=R["weight"]))
    T, n = int(R["T"][i]), int(R["n"][i])
    d = int(np.clip(R["n_hps"][i], 1, 10))
    frac = np.arange(1, T + 1) / T
    X = rng.random((n, d))
    L, y0, kappa = R["L"][i], R["y0"][i], R["kappa"][i]

    # learning rate (dimension 0) and schedule (TaskSet: next dimensions) -> optimiser progress x [n, T]
    lo, hi = R["lr_q_live"][i]
    log_eta = lo + X[:, 0] * (hi - lo)
    oid, opt = int(R["opt_id"][i]), str(R["opt"][i])
    if oid in (1, 2):
        names = SCHED_HPS[opt]
        cols = [1 + j for j in range(len(names)) if 1 + j < d]
        grid = np.linspace(0, 1, 101)
        xs = []
        for a in range(n):
            hp = {h: float(np.interp(X[a, c], grid, R["sched_q"][i][j])) for j, (h, c) in enumerate(zip(names, cols))}
            for h in names:
                hp.setdefault(h, float(R["sched_q"][i][names.index(h)][50]))
            xs.append(progress(oid, hp, frac))
        xs = np.stack(xs)
    else:
        xs = np.tile(progress(oid, {}, frac), (n, 1))
    lrfac = np.exp(kappa * (log_eta - math.log(R["eta_ref"][i]))) if kappa > 0 else np.ones(n)
    x = lrfac[:, None] * xs

    # parameter scores: field 0 (with its rough part) -> floor offset; the others through the copula
    eta_mix = float(R["eta"][i])
    sm, ro = _fields(rng, X, N_FIELDS)
    g = sm.copy()
    g[:, 0] = math.sqrt(1 - eta_mix) * sm[:, 0] + math.sqrt(eta_mix) * ro[:, 0]
    C = R["copula"][i]
    c0 = C[1:, 0]
    cond = C[1:, 1:] - np.outer(c0, c0)
    Lc = np.linalg.cholesky(cond + 1e-6 * np.eye(3))
    zsc = np.empty((n, 4))
    zsc[:, 0] = g[:, 0]
    zsc[:, 1:] = g[:, :1] * c0[None, :] + g[:, 1:4] @ Lc.T
    P = _seg(R, "params", i)
    if R["lr_rel"][i]:          # the record's real learning-rate relation as conditional mean of each score
        for j in range(4):
            zsc[:, j] = np.interp(log_eta, R["lr_mids"][i], R["lr_mb"][i, j]) + math.sqrt(1 - R["lr_r2"][i, j]) * zsc[:, j]
    q = norm.cdf(zsc)
    par = np.stack([np.quantile(P[:, j], q[:, j]) for j in range(4)], 1)    # log d, log A, log tau, log a
    if R["has_start"][i] > 0:   # TaskSet: no live curve above the shared initial loss
        f1 = 1 / (1 + (x[:, 0] / np.exp(par[:, 2])) ** np.exp(par[:, 3]))
        par[:, 1] = np.minimum(par[:, 1], np.log(0.999 * (y0 - L) / np.maximum(f1, 1e-12)))
        dmax = np.maximum(y0 - L - np.exp(par[:, 1]) * f1, 1e-4 * (y0 - L))
        par[:, 0] = np.minimum(par[:, 0], np.log(dmax))
    y = L + np.exp(par[:, 0:1]) + np.exp(par[:, 1:2]) / (1 + (x / np.exp(par[:, 2:3])) ** np.exp(par[:, 3:4]))
    clean_ex = y - L

    # dead (low learning rates) and diverged (high learning rates)
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]

    # rising curves (overfitting / instability): rise after an onset by a relative amount from the record
    on, ri = _seg(R, "over_onset", i), _seg(R, "over_rise", i)
    over = np.array([], int)
    no = int(round(R["share_over"][i] * n))
    if no and len(on):
        reg = X[:, 1] if d > 1 else rng.random(n)
        score = 0.5 * g[:, 6] + 0.5 * (reg - 0.5) / math.sqrt(1 / 12)
        over = np.argsort(-score)[:no]
        j = rng.integers(len(on), size=no)
        o, r = on[j], ri[j]
        ramp = np.clip((frac[None, :] - o[:, None]) / np.maximum(1 - o[:, None], 1e-3), 0, None)
        y[over] += r[:, None] * (y0 - L) * ramp

    # noise: sd ~ (excess / median excess)^k, time profile, Student-t
    prof = np.interp(np.linspace(0, 1, T), [1 / 6, 1 / 2, 5 / 6], R["noise_thirds"][i])
    scale = (np.maximum(clean_ex, 1e-12) / R["noise_med_ex"][i]) ** R["noise_k"][i]
    sd = R["noise_sd"][i] * prof[None, :] * np.minimum(scale, 20.0)
    y = y + rng.standard_t(R["df"][i], (n, T)) * sd
    y[dead] = R["dead_level"][i] + rng.standard_t(R["df"][i], (len(dead), T)) * R["noise_sd"][i] * prof
    dt = _seg(R, "div_times", i)
    hi_val = y0 + 10 * (y0 - L)
    for a in div:
        t0 = rng.choice(dt) if len(dt) else rng.random()
        y[a, int(round(t0 * (T - 1))):] = hi_val
    if R["has_start"][i] > 0:   # TaskSet: the step-0 point (before training), all configs near the start level
        st = L + (y0 - L) * np.exp(R["start_sd"][i] * rng.standard_normal(n))
        st[div] = np.maximum(st[div], y0)
        y = np.concatenate([st[:, None], y], 1)
    src = str(R["source"][i])
    error = int(R["kind"][i]) == 1
    if error:
        y = np.clip(y, 0, 100 if src in ("jahs", "quicktune") else 1)
    return Task(y=y.astype(np.float32), X=X, metric="error" if error else "loss", source=src,
                info={"dead": np.sort(dead), "diverged": np.sort(div), "rising": np.sort(over),
                      "log_lr": log_eta, "schedule": {0: "constant", 1: opt, 2: opt, 3: "cosine"}[oid], "record": i})


def generate(n_tasks: int, seed: int | np.random.Generator = 0, source_weights: dict | None = None) -> list[Task]:
    """Generate n_tasks tasks. Deterministic for a given integer seed.

    source_weights: optional {source: weight} over the record families ("taskset", "jahs", "quicktune",
    "nasbench201"). Default: in proportion to the number of real fit tasks per family (96% TaskSet), the setting
    evaluated in the paper.
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

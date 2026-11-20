"""Faker v0: hierarchical task bootstrap + Gaussian-process landscapes + a learned linear curve basis.

Fitted only on the *fit* split (`fit()`, saved to results/hgp_fit.npz):
    basis   : mean curve + K = 8 principal components of normalised replica-mean curves (50 points over the budget;
              diverged curves left out). Earlier result (journal, question B): 4-8 linear components represent real
              curves as well as a VAE.
    records : one per real fit task: mean and covariance of its live curves' PCA scores, fraction dead (smoothed
              curve never below z = 0.9) and their mean level, fraction diverged, noise level (robust sd of
              step-to-step changes of one seed), number of hyperparameters.

Generating one task (`sample_task`):
    1. Draw a real fit-task record (the task "type"); d = its number of hps (at most 10), n configs uniform in [0,1]^d.
    2. Landscape kernel: relevance r ~ Dirichlet(0.7), length-scales l_j = L / sqrt(r_j / max r), L ~ logU(0.2, 1);
       K = w * additive + (1 - w) * product (Matern-5/2), w ~ U(0, 1).
    3. Ten independent GP fields with kernel K: 8 drive the curve scores, s(x) = mu + chol(Sigma) g(x), and two
       mark regions: the top `frac_dead` of one field are dead configs (flat at the record's dead level), the top
       `frac_diverged` of the other diverge (curve jumps to the cap at a uniform random time).
    4. White noise with the record's noise level.
Output: y [n, 1, 50] in normalised units (lower is better), and X [n, d].
"""
from __future__ import annotations

import math

import numpy as np

from curvefaker import curves, data
from curvefaker.evaluate import resample
from curvefaker.paths import REPO

K = 8
G = 50
PATH = REPO / "results" / "hgp_fit.npz"


def fit(max_tasks_per_source=400, max_cfg=1000, seed=0):
    rng = np.random.default_rng(seed)
    recs, live_curves = [], []
    for src in data.SOURCES:
        ids = data.ids("fit", [src])
        if len(ids) > max_tasks_per_source:
            ids = list(rng.choice(ids, max_tasks_per_source, replace=False))
        for tid in ids:
            t = data.load(tid)
            y = t.y
            if src == "quicktune":
                y = y[np.isfinite(y[:, 0, -1])]
            if len(y) > max_cfg:
                y = y[rng.choice(len(y), max_cfg, replace=False)]
            z, _ = curves.normalise(y)
            m = resample(curves.mean_curve(z), G)
            fin = np.isfinite(m).all(1)
            cls = np.full(len(m), 3)
            cls[fin] = curves.classify(m[fin])
            dead, div = cls == 1, cls == 3
            live = ~dead & ~div
            r0 = z[:, 0]
            ok = np.isfinite(r0).all(1) & (r0 < curves.CAP).all(1)
            dif = np.diff(r0[ok], axis=1)
            mad = np.median(np.abs(dif - np.median(dif, 1, keepdims=True)), 1) * 1.4826 / math.sqrt(2)
            recs.append({"task": tid, "n_hps": len(t.space), "frac_dead": dead.mean(), "frac_div": div.mean(),
                         "dead_level": float(np.median(m[dead])) if dead.any() else 1.0,
                         "noise": float(np.median(mad)) if len(mad) else 0.01, "live": m[live]})
            live_curves.append(m[live][rng.choice(live.sum(), min(200, live.sum()), replace=False)])
    Y = np.concatenate(live_curves)
    mu = Y.mean(0)
    _, S, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    W = Vt[:K]
    out = {"mu": mu, "W": W}
    keep = [r for r in recs if len(r["live"]) >= 20]
    scores = [(r["live"] - mu) @ W.T for r in keep]
    out["s_mean"] = np.stack([s.mean(0) for s in scores])
    out["s_cov"] = np.stack([np.cov(s.T) + 1e-6 * np.eye(K) for s in scores])
    for k in ("n_hps", "frac_dead", "frac_div", "dead_level", "noise"):
        out[k] = np.array([r[k] for r in keep], float)
    out["task"] = np.array([r["task"] for r in keep])
    np.savez(PATH, **out)
    return out


def load():
    return dict(np.load(PATH))


def _matern52(d, ell):
    r = math.sqrt(5) * np.abs(d) / ell
    return (1 + r + r * r / 3) * np.exp(-r)


def sample_task(P, rng, n=1000, max_d=10):
    i = rng.integers(len(P["task"]))
    d = int(np.clip(P["n_hps"][i], 1, max_d))
    X = rng.random((n, d))
    r = rng.dirichlet(np.full(d, 0.7))
    ell = math.exp(rng.uniform(math.log(0.2), 0.0)) / np.sqrt(r / r.max())
    Kg = np.stack([_matern52(X[:, None, j] - X[None, :, j], ell[j]) for j in range(d)])   # [d, n, n]
    w = rng.random()
    Kx = w * np.tensordot(r, Kg, 1) + (1 - w) * Kg.prod(0)
    L = np.linalg.cholesky(Kx + 1e-6 * np.eye(n))
    g = L @ rng.standard_normal((n, K + 2))                                                # [n, K + 2]
    s = P["s_mean"][i] + g[:, :K] @ np.linalg.cholesky(P["s_cov"][i]).T
    m = P["mu"] + s @ P["W"]                                                               # [n, G]
    nd = int(round(P["frac_dead"][i] * n))
    if nd:
        dead = np.argsort(-g[:, K])[:nd]
        m[dead] = P["dead_level"][i]
    nv = int(round(P["frac_div"][i] * n))
    if nv:
        div = np.argsort(-g[:, K + 1])[:nv]
        t0 = rng.integers(1, G, nv)
        for a, t in zip(div, t0):
            m[a, t:] = curves.CAP
    y = m + rng.standard_normal(m.shape) * P["noise"][i]
    return y[:, None, :].astype(np.float32), X


def faker(n_tasks, rng):
    P = load()
    return [sample_task(P, rng) for _ in range(n_tasks)]


# ---- v1: the real task's own curves, assigned to configs by a GP landscape ----------------------------------------
# Change from v0 (one change): the per-task Gaussian over PCA scores is replaced by the real task's own curves
# (one seed, resampled to G points, dead and diverged ones included), so the set of curves of a generated task is
# that of a real fit task. A GP field (same random kernel as v0) decides which config gets which curve: configs
# are ranked by the field and receive the curves in order of final value (the highest field gets the worst
# curve). v0's dead / diverged regions and added noise are no longer needed: both come with the real curves.
PATH_V1 = REPO / "results" / "hgp_v1_fit.npz"


def fit_v1(max_tasks_per_source=400, max_cfg=1000, seed=0):
    rng = np.random.default_rng(seed)
    blocks, offs, n_hps, tasks = [], [0], [], []
    for src in data.SOURCES:
        ids = data.ids("fit", [src])
        if len(ids) > max_tasks_per_source:
            ids = list(rng.choice(ids, max_tasks_per_source, replace=False))
        for tid in ids:
            t = data.load(tid)
            y = t.y[:, 0]
            if src == "quicktune":
                y = y[np.isfinite(y[:, -1])]
            if len(y) > max_cfg:
                y = y[rng.choice(len(y), max_cfg, replace=False)]
            z, _ = curves.normalise(y)
            blocks.append(resample(z, G).astype(np.float32))
            offs.append(offs[-1] + len(z))
            n_hps.append(len(t.space))
            tasks.append(tid)
    np.savez(PATH_V1, curves=np.concatenate(blocks), offs=np.array(offs), n_hps=np.array(n_hps),
             task=np.array(tasks))


def _kernel_field(rng, X, n_fields):
    n, d = X.shape
    r = rng.dirichlet(np.full(d, 0.7))
    ell = math.exp(rng.uniform(math.log(0.2), 0.0)) / np.sqrt(r / r.max())
    Kg = np.stack([_matern52(X[:, None, j] - X[None, :, j], ell[j]) for j in range(d)])
    w = rng.random()
    Kx = w * np.tensordot(r, Kg, 1) + (1 - w) * Kg.prod(0)
    return np.linalg.cholesky(Kx + 1e-6 * np.eye(n)) @ rng.standard_normal((n, n_fields))


def sample_task_v1(P, rng, max_d=10, i=None):
    i = rng.integers(len(P["task"])) if i is None else i
    C = P["curves"][P["offs"][i]:P["offs"][i + 1]]
    n = len(C)
    d = int(np.clip(P["n_hps"][i], 1, max_d))
    X = rng.random((n, d))
    f = _kernel_field(rng, X, 1)[:, 0]
    fin = np.where(np.isfinite(C).all(1), curves.final_value(np.nan_to_num(C, nan=curves.CAP)), np.inf)
    y = np.empty_like(C)
    y[np.argsort(f)] = C[np.argsort(fin, kind="stable")]
    return y[:, None, :], X


def faker_v1(n_tasks, rng):
    P = dict(np.load(PATH_V1))
    return [sample_task_v1(P, rng) for _ in range(n_tasks)]

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


def sample_task_v1(P, rng, max_d=10, i=None, warp_level=False, warp_time=False, eta=0.0):
    """v1; v2 adds warp_level, v3 adds warp_time (task-level transformations, see below)."""
    i = rng.integers(len(P["task"])) if i is None else i
    C = P["curves"][P["offs"][i]:P["offs"][i + 1]].copy()
    if warp_level:   # v2: z -> z^gamma, gamma ~ logU(1/2, 2): easier / harder task, every order kept
        g = math.exp(rng.uniform(-math.log(2), math.log(2)))
        C = np.where(np.isfinite(C), np.clip(C, 0, None) ** g, C)
    if warp_time:    # v3: curve(u) -> curve(u^a), a ~ logU(1/2, 2): faster / slower learning
        a = math.exp(rng.uniform(-math.log(2), math.log(2)))
        u = np.linspace(0, 1, C.shape[1])
        C = np.stack([np.interp(u ** a, u, c) if np.isfinite(c).all() else c for c in C])
    n = len(C)
    d = int(np.clip(P["n_hps"][i], 1, max_d))
    X = rng.random((n, d))
    f = _kernel_field(rng, X, 1)[:, 0]
    if eta > 0:      # v5: nugget, part of the final-value order not predictable from the config
        f = math.sqrt(1 - eta) * f / f.std() + math.sqrt(eta) * rng.standard_normal(n)
    fin = np.where(np.isfinite(C).all(1), curves.final_value(np.nan_to_num(C, nan=curves.CAP)), np.inf)
    y = np.empty_like(C)
    y[np.argsort(f)] = C[np.argsort(fin, kind="stable")]
    return y[:, None, :], X


def faker_v1(n_tasks, rng, **kw):
    P = dict(np.load(PATH_V1))
    return [sample_task_v1(P, rng, **kw) for _ in range(n_tasks)]


# ---- v4: independent class fractions -------------------------------------------------------------------------------
# Change from v2 (one change): in a real task the shares of dead, diverged and turning curves come as a package. v4
# draws each share from a *different* random fit task, together with that task's curves of that class; the
# improving curves come from the main record. So combinations no single fit task has (e.g. a predictable task with a
# few overfitting configs) can occur. The level warp of v2 is kept.
def _classes(P):
    if "cls" not in P:
        C = P["curves"]
        fin = np.isfinite(C).all(1)
        cls = np.full(len(C), 3)
        cls[fin] = curves.classify(C[fin])
        P["cls"] = cls
    return P["cls"]


def sample_task_v4(P, rng, max_d=10):
    cls = _classes(P)
    nrec = len(P["task"])
    i = rng.integers(nrec)
    sl = slice(P["offs"][i], P["offs"][i + 1])
    n = sl.stop - sl.start
    parts = []
    for c in (1, 2, 3):                                   # dead, turns, diverged: share and curves from another task
        j = rng.integers(nrec)
        cj = cls[P["offs"][j]:P["offs"][j + 1]]
        k = int(round((cj == c).mean() * n))
        pool = P["offs"][j] + np.where(cj == c)[0]
        if k and len(pool):
            parts.append(rng.choice(pool, k, replace=len(pool) < k))
    k_imp = max(n - sum(len(p) for p in parts), max(1, n // 10))
    ci = cls[sl]
    pool = sl.start + np.where(ci == 0)[0]
    if not len(pool):
        pool = sl.start + np.arange(n)
    parts.append(rng.choice(pool, k_imp, replace=len(pool) < k_imp))
    C = P["curves"][np.concatenate(parts)].copy()
    g = math.exp(rng.uniform(-math.log(2), math.log(2)))
    C = np.where(np.isfinite(C), np.clip(C, 0, None) ** g, C)
    d = int(np.clip(P["n_hps"][i], 1, max_d))
    X = rng.random((len(C), d))
    f = _kernel_field(rng, X, 1)[:, 0]
    fin = np.where(np.isfinite(C).all(1), curves.final_value(np.nan_to_num(C, nan=curves.CAP)), np.inf)
    y = np.empty_like(C)
    y[np.argsort(f)] = C[np.argsort(fin, kind="stable")]
    return y[:, None, :], X


def faker_v4(n_tasks, rng):
    P = dict(np.load(PATH_V1))
    return [sample_task_v4(P, rng) for _ in range(n_tasks)]


# ---- v1w: v1 with records weighted like the fit split (diagnostic) ------------------------------------------------
def faker_v1w(n_tasks, rng):
    """v1, but records are drawn in proportion to how common their source is among all fit tasks (TaskSet ~96%),
    instead of uniformly over records (TaskSet capped at 400 of 566). Separates the effect of the task mixture
    from that of the landscape model."""
    P = dict(np.load(PATH_V1))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src])
    w /= w.sum()
    return [sample_task_v1(P, rng, i=int(rng.choice(len(w), p=w))) for _ in range(n_tasks)]


# ---- v5: landscape predictability matched per record ---------------------------------------------------------------
# Change from v1w (one change): generated landscapes were too predictable (median landscape_rho 0.89 vs 0.77 on real
# select tasks), because the final value was an exact function of a smooth GP field. v5 mixes in a nugget: field =
# sqrt(1 - eta) * GP + sqrt(eta) * white noise. eta is chosen per record so that the expected landscape_rho of the
# generated task equals that of the real fit task it came from (measured on the fit split only, `fit_v5`), via a
# table of expected landscape_rho over (d, eta) simulated once (`calibrate_v5`).
PATH_V5 = REPO / "results" / "hgp_v5_fit.npz"
ETAS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def fit_v5(max_cfg=1000, seed=0, reps=8):
    from curvefaker.evaluate import landscape_rho
    rng = np.random.default_rng(seed)
    P = dict(np.load(PATH_V1))
    rho = []
    for k, tid in enumerate(P["task"]):
        t = data.load(str(tid))
        y = t.y[:, 0]
        X = t.X()
        keep = np.arange(len(y))
        if t.source == "quicktune":
            keep = keep[np.isfinite(y[:, -1])]
        if len(keep) > max_cfg:
            keep = rng.choice(keep, max_cfg, replace=False)
        z, _ = curves.normalise(y[keep])
        m = resample(z, G)
        ok = np.isfinite(m).all(1)
        rho.append(landscape_rho(X[keep][ok], curves.final_value(m[ok])) if ok.sum() > 20 else np.nan)
    table = np.zeros((11, len(ETAS)))                      # expected landscape_rho for d = 0..10 (row 0 unused)
    for d in range(1, 11):
        for e, eta in enumerate(ETAS):
            vals = []
            for _ in range(reps):
                X = rng.random((1000, d))
                f = _kernel_field(rng, X, 1)[:, 0]
                f = math.sqrt(1 - eta) * f / f.std() + math.sqrt(eta) * rng.standard_normal(len(f))
                vals.append(landscape_rho(X, f))
            table[d, e] = np.mean(vals)
    np.savez(PATH_V5, rho=np.array(rho), table=table)


def _eta_for(table_d, target):
    """Smallest eta whose expected rho is at most the target; 0 if the target is higher. The simulated table is
    noisy, so it is made monotone (running minimum along eta) first."""
    table_d = np.minimum.accumulate(table_d)
    if not np.isfinite(target) or target >= table_d[0]:
        return 0.0
    if target <= table_d[-1]:
        return float(ETAS[-1])
    return float(np.interp(-target, -table_d, ETAS))


def faker_v5(n_tasks, rng):
    P = dict(np.load(PATH_V1))
    C5 = dict(np.load(PATH_V5))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src])
    w /= w.sum()
    out = []
    for _ in range(n_tasks):
        i = int(rng.choice(len(w), p=w))
        d = int(np.clip(P["n_hps"][i], 1, 10))
        eta = _eta_for(C5["table"][d], C5["rho"][i])
        out.append(sample_task_v1(P, rng, i=i, eta=eta))
    return out

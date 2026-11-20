"""Quality measures, real vs fake (README "Quality measures"): M1 curve statistics and M4 coverage.

Statistics start from rogue-dagger's scripts/curve_stats.py (same definitions, so numbers are comparable), plus
statistics that need replicas (seed noise) and are computed on real tasks only.

Pipeline:
    curves(task)        -> Y [n, T] normalised loss in [0, 1] (ifBO conventions), X [n, d] in [0, 1]
    stats(Y, X)         -> dict of scalars per task
    ifbo_twins(...)     -> fake tasks with the same n, d, T from the ifBO prior (rogue-dagger's port)
    compare(real, fake) -> per statistic: medians and KS distance
    coverage(real, fake)-> k-NN precision / recall of task-level statistic vectors (Kynkäänniemi et al. 2019)
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from curvefaker.data import Task
from curvefaker.paths import REPO

ROGUE = REPO.parent / "rogue-dagger"

# --------------------------------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------------------------------
BOUNDED = {"metric_valid_error": 1.0, "100-top1": 100.0}   # error rates: divide by the maximum, clip to [0, 1]


def curves(task: Task, replica="first", max_configs=1000, rng=None, full_only=True, cat="onehot"):
    """Normalised curves as an HPO method on this task would see them.

    replica: "first" (one training run per config, as on LCBench / PD1) or "mean" (mean over replicas, as
    rogue-dagger processed TaskSet). Unbounded losses: drop the step-0 point (ifBO's drop_epoch_0), divide by the
    median initial loss (a practitioner's upper bound), NaN/inf -> that bound, clip to [0, 1]. Error rates: divide
    by their maximum. full_only drops configs whose curve stops early (QuickTune). At most max_configs configs,
    a fixed random subset."""
    rng = rng or np.random.default_rng(0)
    y = task.y[:, 0] if replica == "first" else task.mean_curve()
    X = task.X(cat)
    if full_only:
        keep = np.isfinite(y[:, -1]) | (np.isfinite(y).sum(1) == y.shape[1])
        if task.source == "quicktune":   # NaN past the end of a cut run, not divergence
            keep = np.isfinite(y).all(1)
        y, X = y[keep], X[keep]
    if len(y) > max_configs:
        idx = np.sort(rng.choice(len(y), max_configs, replace=False))
        y, X = y[idx], X[idx]
    y = y.astype(np.float64)
    if task.metric in BOUNDED:
        y = np.where(np.isfinite(y), y, BOUNDED[task.metric]) / BOUNDED[task.metric]
        Y = y.clip(0, 1)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ub = np.nanmedian(np.where(np.isfinite(y[:, 0]), y[:, 0], np.nan))
        if not np.isfinite(ub) or ub <= 0:
            ub = np.nanmax(np.where(np.isfinite(y), y, np.nan))
        y = y[:, 1:]
        Y = np.where(np.isfinite(y), y, ub).clip(0, ub) / ub
    return Y, X


# --------------------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------------------
def _rank(a):
    return np.argsort(np.argsort(a, axis=0), axis=0).astype(np.float64)


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def stats(Y, X, rng=None, n_pairs=4000):
    """Y [n, T] loss in [0, 1] (lower better), X [n, d]. Same definitions as rogue-dagger's curve_stats.py."""
    rng = rng or np.random.default_rng(0)
    n, T = Y.shape
    cm = np.minimum.accumulate(Y, axis=1)
    lo, hi = np.quantile(Y, 0.01), np.quantile(Y, 0.99)
    rngv = max(hi - lo, 1e-9)
    final = cm[:, -1]
    s = {}
    for f in (0.1, 0.25, 0.5):
        k = max(0, int(round(f * T)) - 1)
        s[f"rank_corr@{f}"] = spearman(cm[:, k], final)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    k = max(0, int(round(0.1 * T)) - 1)
    d0, d1 = cm[i, k] - cm[j, k], final[i] - final[j]
    ok = (np.abs(d0) > 1e-9) & (np.abs(d1) > 1e-9)
    s["flip@0.1"] = float((np.sign(d0[ok]) != np.sign(d1[ok])).mean()) if ok.any() else np.nan
    s["final_spread"] = float((np.quantile(final, 0.9) - np.quantile(final, 0.1)) / rngv)
    s["frac_near_best"] = float((final <= final.min() + 0.05 * rngv).mean())
    own = Y.max(1) - Y.min(1)
    flat = own < 0.02 * rngv
    s["frac_flat"] = float(flat.mean())
    s["frac_diverge"] = float((Y[:, -1] > Y[:, 0] + 0.05 * rngv).mean())
    s["improve_late"] = float(np.median((cm[:, T // 2] - final) / rngv))
    Z, own = Y[~flat], own[~flat]
    if len(Z) >= 5:
        Zn = (Z - Z.min(1, keepdims=True)) / own[:, None]
        am = Z.argmin(1)
        s["frac_overfit"] = float(((am < 0.8 * (T - 1)) & (Zn[:, -1] > 0.1)).mean())
        if T >= 7:
            ker = np.ones(5) / 5
            sm = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="valid"), 1, Zn)
            s["noise"] = float(np.median((Zn[:, 2:-2] - sm).std(1)))
            dY = np.diff(Zn, axis=1)
            s["frac_up_steps"] = float((dY > 1e-6).mean())
            a, b = dY[:, :-1], dY[:, 1:]
            a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
            with np.errstate(all="ignore"):
                s["incr_autocorr"] = float(np.nanmedian(np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-12),
                                                                 np.nan)))
    if X.shape[1] > 0:
        dx = np.linalg.norm(X[i] - X[j], axis=1)
        s["cfg_smooth"] = spearman(dx, np.abs(final[i] - final[j]))
    return s


def replica_stats(task: Task, max_configs=1000, rng=None):
    """Real-only statistics from replicas (seeds), on the same normalisation as curves(..., replica="first").

    luck_ratio: across-seed variance of the final running best / across-config variance of its seed mean. The
    share of the final-loss spread that the config does not determine (avenue V2's luck term)."""
    if task.y.shape[1] < 2:
        return {}
    rng = rng or np.random.default_rng(0)
    reps = []
    for r in range(task.y.shape[1]):
        t2 = Task(task.id, task.source, task.family, task.hps, task.space, task.steps, task.y[:, r:r + 1],
                  task.metric)
        Y, _ = curves(t2, max_configs=max_configs, rng=np.random.default_rng(0), full_only=False)
        reps.append(np.minimum.accumulate(Y, axis=1)[:, -1])
    F = np.stack(reps, 1)                     # [n, R]
    within = F.var(1, ddof=1).mean()
    between = F.mean(1).var(ddof=1)
    rho = np.mean([spearman(F[:, a], F[:, b]) for a in range(F.shape[1]) for b in range(a + 1, F.shape[1])])
    return {"luck_ratio": float(within / max(between, 1e-12)), "seed_rank_corr": float(rho)}


# --------------------------------------------------------------------------------------------------------------
# ifBO twins
# --------------------------------------------------------------------------------------------------------------
def ifbo_twins(shapes, k, device="cuda", seed=0, noise_scale=1.0, noise_smooth=0.0, d_max=10):
    """For each (n, d, T) in shapes, k ifBO prior tasks with uniform random configs. Yields (index, X, Y)."""
    import torch
    if str(ROGUE) not in sys.path:
        sys.path.insert(0, str(ROGUE))
    from roguedagger.prior import PriorTasks

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    with torch.no_grad():
        for idx, (n, d, T) in enumerate(shapes):
            d = max(1, min(d, d_max))
            pr = PriorTasks(k, d_max, device, generator=g)
            pr.noise_scale, pr.noise_smooth = noise_scale, noise_smooth
            pr.resample(torch.arange(k, device=device), torch.full((k,), d, device=device))
            x = torch.rand(k, n, d_max, device=device, generator=g)
            x[:, :, d:] = 0
            out = []
            for c in range(0, n, 250):
                out.append(pr.curves(x[:, c:c + 250], torch.full((k,), T, device=device), T).cpu().numpy())
            Y = np.concatenate(out, 1)
            for r in range(k):
                yield idx, x[r, :, :d].cpu().numpy(), Y[r].astype(np.float64)


# --------------------------------------------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------------------------------------------
def ks(a, b):
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    z = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, z, "right") / len(a) - np.searchsorted(b, z, "right") / len(b)).max())


def compare(R: list[dict], S: list[dict]):
    """Per statistic: (KS, real median, fake median), sorted by KS descending."""
    keys = [k for k in dict.fromkeys(k for x in R + S for k in x) if k != "family"]
    rows = []
    for key in keys:
        r = np.array([x.get(key, np.nan) for x in R], float)
        s = np.array([x.get(key, np.nan) for x in S], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append((key, ks(r, s), float(np.nanmedian(r)), float(np.nanmedian(s))))
    return sorted(rows, key=lambda z: -np.nan_to_num(z[1]))


def coverage(R: list[dict], S: list[dict], keys, k=5):
    """k-NN precision / recall (Kynkäänniemi et al. 2019) on standardised statistic vectors.

    recall: fraction of real tasks inside the fake manifold (some fake task's k-NN ball contains it) -- what M4
    calls coverage. precision: fraction of fake tasks inside the real manifold."""
    A = np.array([[x.get(q, np.nan) for q in keys] for x in R], float)
    B = np.array([[x.get(q, np.nan) for q in keys] for x in S], float)
    ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
    A, B = A[:, ok], B[:, ok]
    A, B = A[np.isfinite(A).all(1)], B[np.isfinite(B).all(1)]
    mu, sd = np.concatenate([A, B]).mean(0), np.concatenate([A, B]).std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd

    def radii(P):
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        return np.sort(D, 1)[:, min(k, len(P) - 1)]

    rA, rB = radii(A), radii(B)
    DAB = np.linalg.norm(A[:, None] - B[None], axis=-1)       # [nA, nB]
    recall = float((DAB <= rB[None]).any(1).mean())
    precision = float((DAB <= rA[:, None]).any(0).mean())
    return {"recall": recall, "precision": precision, "n_real": len(A), "n_fake": len(B),
            "keys": [q for q, o in zip(keys, ok) if o]}

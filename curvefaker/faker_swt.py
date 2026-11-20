"""Faker "swt" (shape-warp-template): the model class motivated by the analysis (docs/analysis_plan.md, section 6).

Fitted on the *fit* split only (`fit_records`, one record per fit task of curvefaker.faker_hgp's v1 list):
    template : the task's shape-invariant template f in log-time (curvefaker/sim.py), fitted on up to 200 live
               curves (one seed, native length)
    warps    : the fitted per-curve warps (level alpha, log amplitude, log-time shift beta, stretch gamma); their
               empirical distributions and the correlation of their normal scores (Gaussian copula)
    noise    : robust sd of the residuals, its profile over the budget (thirds), excess kurtosis -> Student-t degrees
               of freedom
    classes  : shares of dead / diverged curves, dead level, divergence times (budget fraction)

Generating one task (v1):
    1. Record i drawn in proportion to its source in the fit split; d = its number of hps (<= 10); configs uniform.
    2. GP fields over the configs (random kernel of faker_hgp, nugget eta of faker v5 on field 0). Warps are joint
       vectors of the task's real fitted curves (resampled): field 0 ranks configs, which get vectors in order of
       their implied final value; within strata of 10 similar-final vectors, field 1 assigns the speeds. So the
       config shapes the curve beyond its final value (analysis H4), and every warp combination is a real one.
    3. Dead / diverged: the top shares of two scores that mix the first config dimension (an "LR-like" axis) with
       a field: dead = low end, diverged = high end of that axis (analysis H5).
    4. Live curves = template warped per config (template held flat outside the range real curves covered); dead = flat at the dead level; diverged = live curve until a
       divergence time drawn from the record, then the cap. Noise: Student-t with the record's df, scaled by the
       record's sd and its time profile.
The curves are generated (template + warps + noise), not replayed; the template itself is re-used from a real task.
"""
from __future__ import annotations

import math
from multiprocessing import get_context

import numpy as np
from scipy.stats import kurtosis, norm, rankdata

from curvefaker import curves, data, sim
from curvefaker.paths import REPO

PATH = REPO / "results" / "swt_records.npz"
MAX_LIVE = 200
LR_AXIS_WEIGHT = 0.7
STRATUM = 10


def _fit_one(args):
    tid, seed = args
    rng = np.random.default_rng(seed)
    t = data.load(tid)
    y = t.y[:, 0]
    if t.source == "quicktune":
        y = y[np.isfinite(y[:, -1])]
    if len(y) > 1000:
        y = y[rng.choice(len(y), 1000, replace=False)]
    z, _ = curves.normalise(y)
    ok = np.isfinite(z).all(1)
    c = np.full(len(z), 3)
    c[ok] = curves.classify(z[ok])
    T = z.shape[1]
    live = np.where((c == 0) | (c == 2))[0]
    out = {"task": tid, "T": T, "n": len(z), "n_hps": len(t.space),
           "share_dead": float((c == 1).mean()), "share_div": float((c == 3).mean())}
    out["dead_level"] = float(np.median(curves.final_value(z[c == 1]))) if (c == 1).any() else 1.0
    div = np.where(c == 3)[0]
    zc = np.nan_to_num(z, nan=curves.CAP, posinf=curves.CAP)
    out["div_times"] = np.array([(np.argmax(zc[i] >= curves.CAP) if (zc[i] >= curves.CAP).any() else T - 1) / (T - 1)
                                 for i in div], float)
    if len(live) < 10:
        out["ok"] = False
        return out
    pick = rng.choice(live, min(MAX_LIVE, len(live)), replace=False)
    fit = sim.fit_task(z[pick])
    r = z[pick] - fit["fitted"]
    sd = np.median(np.abs(r - np.median(r))) * 1.4826
    thirds = [np.median(np.abs(r[:, th] - np.median(r[:, th]))) * 1.4826 / max(sd, 1e-9)
              for th in np.array_split(np.arange(T), 3)]
    k = float(np.median(kurtosis(r / max(sd, 1e-9), axis=1)))
    u = np.log(np.arange(1, T + 1) / T)
    out["x_range"] = np.array(sim.covered_range(fit["beta"], fit["gamma"], u))
    out.update({"ok": True, "f": fit["f"], "alpha": fit["alpha"], "log_amp": np.log(fit["amp"]),
                "beta": fit["beta"], "gamma": fit["gamma"], "noise_sd": float(sd), "noise_thirds": np.array(thirds),
                "df": float(np.clip(4 + 6 / max(k, 1e-3), 4.5, 50)) if k > 0 else 50.0})
    W = np.stack([fit["alpha"], np.log(fit["amp"]), fit["beta"], fit["gamma"]], 1)
    S = norm.ppf(np.apply_along_axis(rankdata, 0, W) / (len(W) + 1))
    C = np.corrcoef(S.T)
    out["copula"] = np.nan_to_num(C, nan=0.0) + np.diag(np.where(np.isnan(np.diag(C)), 1.0, 0.0))
    return out


def fit_records(processes=64):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker: no
        os.environ[v] = "1"                                                  # oversubscription (spawned workers
    from curvefaker import faker_hgp as F                                     # inherit the environment)
    P = dict(np.load(F.PATH_V1))
    jobs = [(str(t), i) for i, t in enumerate(P["task"])]
    with get_context("spawn").Pool(processes) as pool:
        recs = pool.map(_fit_one, jobs, chunksize=2)
    out = {"task": np.array([r["task"] for r in recs]), "ok": np.array([r["ok"] for r in recs])}
    for k in ("T", "n", "n_hps", "share_dead", "share_div", "dead_level"):
        out[k] = np.array([r[k] for r in recs], float)
    K = len(sim.KNOTS)
    nan4 = np.full(4, np.nan)
    out["f"] = np.stack([r.get("f", np.zeros(K)) for r in recs])
    out["noise_sd"] = np.array([r.get("noise_sd", np.nan) for r in recs])
    out["noise_thirds"] = np.stack([r.get("noise_thirds", np.ones(3)) for r in recs])
    out["df"] = np.array([r.get("df", 50.0) for r in recs])
    out["x_range"] = np.stack([r.get("x_range", np.array([-9.0, 4.0])) for r in recs])
    out["copula"] = np.stack([r.get("copula", np.eye(4)) for r in recs])
    # ragged arrays: warps of live curves and divergence times, concatenated with offsets
    W = [np.stack([r["alpha"], r["log_amp"], r["beta"], r["gamma"]], 1) if r["ok"] else np.zeros((0, 4)) for r in recs]
    out["warps"] = np.concatenate(W)
    out["warp_offs"] = np.concatenate([[0], np.cumsum([len(w) for w in W])])
    D = [r["div_times"] for r in recs]
    out["div_times"] = np.concatenate(D) if sum(len(d) for d in D) else np.zeros(0)
    out["div_offs"] = np.concatenate([[0], np.cumsum([len(d) for d in D])])
    np.savez(PATH, **out)
    return out


def _noise_profile(thirds, T):
    centers = np.array([1 / 6, 1 / 2, 5 / 6])
    return np.interp(np.linspace(0, 1, T), centers, thirds)


def sample_task(R, P5, w, rng, version=1):
    from curvefaker import faker_hgp as F
    i = int(rng.choice(len(w), p=w))
    T = int(R["T"][i])
    n = int(R["n"][i])
    d = int(np.clip(R["n_hps"][i], 1, 10))
    eta = F._eta_for(P5["table"][d], P5["rho"][i])
    X = rng.random((n, d))
    g = F._kernel_field(rng, X, 6)
    g = (g - g.mean(0)) / g.std(0)
    if eta > 0:
        g[:, :4] = math.sqrt(1 - eta) * g[:, :4] + math.sqrt(eta) * rng.standard_normal((n, 4))
    # warps: joint vectors of real fitted curves (a Gaussian copula over the four parameters was tried first and
    # produced combinations no real curve has: up to 18% of live final values below the task's best). Field 0 (with
    # the nugget) ranks configs, which receive warp vectors in order of their implied final value; within strata of
    # STRATUM vectors with similar final value, field 1 decides which config gets which speed (beta).
    Wi = R["warps"][R["warp_offs"][i]:R["warp_offs"][i + 1]]
    u = np.log(np.arange(1, T + 1) / T)
    rows = Wi[rng.integers(len(Wi), size=n)]
    fin_rows = sim.evaluate(R["f"][i], rows[:, 0], np.exp(rows[:, 1]), rows[:, 2], rows[:, 3], u[-2:],
                            R["x_range"][i])[:, -1]
    if version == 1:
        rows = rows[np.argsort(fin_rows, kind="stable")]
        order = np.argsort(g[:, 0])
        warps = np.empty_like(rows)
        for s0 in range(0, n, STRATUM):
            cfg = order[s0:s0 + STRATUM]
            blk = rows[s0:s0 + STRATUM]
            warps[cfg[np.argsort(g[cfg, 1])]] = blk[np.argsort(blk[:, 2])]
    else:
        # v2: speed is its own smooth function of the config. Each config gets a target (normal scores of final
        # value and speed) from fields 0 and 1, correlated like the task's real (final, speed) pairs, and takes the
        # real warp vector nearest to it. (v1's strata of 10 left speed almost unpredictable from the config:
        # R^2 0.13 vs 0.73 on real tasks.)
        fin_all = sim.evaluate(R["f"][i], Wi[:, 0], np.exp(Wi[:, 1]), Wi[:, 2], Wi[:, 3], u[-2:],
                               R["x_range"][i])[:, -1]
        zs = norm.ppf(np.stack([rankdata(fin_all), rankdata(Wi[:, 2])], 1) / (len(Wi) + 1))
        rho = float(np.corrcoef(zs.T)[0, 1]) if len(Wi) > 2 else 0.0
        g1 = g[:, 1] if eta == 0 else math.sqrt(1 - eta) * g[:, 1] + math.sqrt(eta) * rng.standard_normal(n)
        tgt = np.stack([g[:, 0], rho * g[:, 0] + math.sqrt(max(1 - rho ** 2, 0)) * g1], 1)
        tgt = (tgt - tgt.mean(0)) / tgt.std(0)
        d2 = ((tgt[:, None, :] - zs[None]) ** 2).sum(-1)
        warps = Wi[d2.argmin(1)]
    y = sim.evaluate(R["f"][i], warps[:, 0], np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u, R["x_range"][i])
    # classes along an LR-like axis (first config dimension)
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]
    sd = R["noise_sd"][i] * _noise_profile(R["noise_thirds"][i], T)
    noise = rng.standard_t(R["df"][i], (n, T)) * sd
    y[dead] = R["dead_level"][i]
    y = y + noise
    dt = R["div_times"][R["div_offs"][i]:R["div_offs"][i + 1]]
    for a in div:
        t0 = rng.choice(dt) if len(dt) else rng.random()
        y[a, int(round(t0 * (T - 1))):] = curves.CAP
    return np.clip(y, -0.5, curves.CAP)[:, None, :].astype(np.float32), X


def faker(n_tasks, rng, version=1):
    from curvefaker import faker_hgp as F
    R = dict(np.load(PATH))
    P5 = dict(np.load(F.PATH_V5))
    src = np.array([t.split("/")[0] for t in R["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    return [sample_task(R, P5, w, rng, version) for _ in range(n_tasks)]

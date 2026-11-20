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

PATH = REPO / "results" / "swt_records.npz"          # v1-v4: up to 200 live curves fitted per task
PATH_ALL = REPO / "results" / "swt_records_all.npz"  # v5: all live curves (up to 1000)
ETA_ROUGH = REPO / "results" / "swt_eta_rough.npz"
ROUGH_SCALE = 0.15
MAX_LIVE = 200
LR_AXIS_WEIGHT = 0.7
STRATUM = 10


def _fit_one(args):
    tid, seed = args[:2]
    max_live = args[2] if len(args) > 2 else MAX_LIVE
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
    pick = rng.choice(live, min(max_live, len(live)), replace=False)
    fit = sim.fit_task(z[pick])
    r = z[pick] - fit["fitted"]
    sd = np.median(np.abs(r - np.median(r))) * 1.4826
    thirds = [np.median(np.abs(r[:, th] - np.median(r[:, th]))) * 1.4826 / max(sd, 1e-9)
              for th in np.array_split(np.arange(T), 3)]
    k = float(np.median(kurtosis(r / max(sd, 1e-9), axis=1)))
    # v4: noise level from step-to-step differences of the real curves (robust; template misfit is smooth, so it
    # barely enters differences, unlike the residuals above, which include it)
    dz = np.diff(z[pick], axis=1)
    mad = lambda a: np.median(np.abs(a - np.median(a))) * 1.4826 / math.sqrt(2)
    sd_d = mad(dz)
    thirds_d = [mad(dz[:, th]) / max(sd_d, 1e-9) for th in np.array_split(np.arange(T - 1), 3)]
    out["noise_sd_diff"] = float(sd_d)
    out["noise_thirds_diff"] = np.array(thirds_d)
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


def fit_records(processes=64, max_live=MAX_LIVE, path=None):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker: no
        os.environ[v] = "1"                                                  # oversubscription (spawned workers
    from curvefaker import faker_hgp as F                                     # inherit the environment)
    P = dict(np.load(F.PATH_V1))
    jobs = [(str(t), i, max_live) for i, t in enumerate(P["task"])]
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
    out["noise_sd_diff"] = np.array([r.get("noise_sd_diff", np.nan) for r in recs])
    out["noise_thirds_diff"] = np.stack([r.get("noise_thirds_diff", np.ones(3)) for r in recs])
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
    np.savez(path or PATH, **out)
    return out


def _fields(rng, X, n_fields):
    """Smooth GP fields (the random kernel of faker_hgp) and rough companions: same relevances and additive /
    product mix, length-scales x ROUGH_SCALE. Returns (smooth [n, k], rough [n, k]), each standardised."""
    from curvefaker.faker_hgp import _matern52
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


def calibrate_rough(reps=8, seed=0):
    """Expected landscape_rho for (d, eta) when the unpredictable part is the rough field (v3), as faker v5's table
    did for white noise."""
    from curvefaker.evaluate import landscape_rho
    from curvefaker.faker_hgp import ETAS
    rng = np.random.default_rng(seed)
    table = np.zeros((11, len(ETAS)))
    for d in range(1, 11):
        for e, eta in enumerate(ETAS):
            vals = []
            for _ in range(reps):
                X = rng.random((1000, d))
                sm, ro = _fields(rng, X, 1)
                f = math.sqrt(1 - eta) * sm[:, 0] + math.sqrt(eta) * ro[:, 0]
                vals.append(landscape_rho(X, f))
            table[d, e] = np.mean(vals)
    np.savez(ETA_ROUGH, table=table)


def _noise_profile(thirds, T):
    centers = np.array([1 / 6, 1 / 2, 5 / 6])
    return np.interp(np.linspace(0, 1, T), centers, thirds)


def sample_task(R, P5, w, rng, version=1):
    from curvefaker import faker_hgp as F
    i = int(rng.choice(len(w), p=w))
    T = int(R["T"][i])
    n = int(R["n"][i])
    d = int(np.clip(R["n_hps"][i], 1, 10))
    X = rng.random((n, d))
    if version >= 3:
        # v3: the unpredictable part of the landscape is a rough but spatially coherent field (length-scales x0.15)
        # instead of white noise; white noise made landscapes far too rugged (local-optima ratio 0.77 vs 0.46).
        eta = F._eta_for(P5["table_rough"][d], P5["rho"][i])
        sm, ro = _fields(rng, X, 6)
        g = sm.copy()
        # v8: only the final-value field (0) gets the rough part. It is calibrated to the final value's
        # predictability; mixing it into the speed field too made speed far less predictable than on real tasks
        # (diagnostic: without the rough part speed-from-config 0.46 -> 0.63, real 0.73).
        k = 1 if version >= 8 else 2
        g[:, :k] = math.sqrt(1 - eta) * sm[:, :k] + math.sqrt(eta) * ro[:, :k]
        eta_extra = 0.0
    else:
        eta = F._eta_for(P5["table"][d], P5["rho"][i])
        g = F._kernel_field(rng, X, 6)
        g = (g - g.mean(0)) / g.std(0)
        if eta > 0:
            g[:, :4] = math.sqrt(1 - eta) * g[:, :4] + math.sqrt(eta) * rng.standard_normal((n, 4))
        eta_extra = eta
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
        g1 = g[:, 1] if eta_extra == 0 else math.sqrt(1 - eta_extra) * g[:, 1] + math.sqrt(eta_extra) * rng.standard_normal(n)
        tgt = np.stack([g[:, 0], rho * g[:, 0] + math.sqrt(max(1 - rho ** 2, 0)) * g1], 1)
        tgt = (tgt - tgt.mean(0)) / tgt.std(0)
        wf = 4.0 if version >= 6 else 1.0   # v6: the final-value coordinate weighs 4x, so the nearest-vector match
        # does not trade final-value fidelity for speed (v5's landscapes were less predictable than calibrated)
        d2 = wf * (tgt[:, None, 0] - zs[None, :, 0]) ** 2 + (tgt[:, None, 1] - zs[None, :, 1]) ** 2
        warps = Wi[d2.argmin(1)].copy()
        if version >= 7:
            # v7: construct instead of match. The nearest-vector match kept only ~0.75 rank correlation between the
            # target field and the realised final value, and beta sits on the fit grid (41 values), so speed was
            # quantised. Now final value and beta are the target's quantiles of the task's real marginals (beta
            # jittered within its grid cell); amplitude and gamma come from the nearest real vector; alpha is solved
            # so that the curve ends exactly at the target final value.
            q = norm.cdf(tgt)
            fin_t = np.quantile(fin_all, q[:, 0])
            step = sim.BETAS[1] - sim.BETAS[0]
            warps[:, 2] = np.quantile(Wi[:, 2], q[:, 1]) + rng.uniform(-step / 2, step / 2, n)
            end = sim.evaluate(R["f"][i], np.zeros(n), np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u[-2:],
                               R["x_range"][i])[:, -1]
            matched = warps.copy()
            warps[:, 0] = fin_t - end
            if version >= 9:
                # v9: a constructed vector whose curve dips below the lowest fitted real curve of the task falls
                # back to its matched real vector. Solving alpha after moving beta made 23% of tasks undershoot the
                # real best by > 0.1 (fitted records: never by > 0.3; v6: 4%), which created fake optima and shifted
                # the per-task normalisation (dead curves no longer classified dead).
                lo = sim.evaluate(R["f"][i], Wi[:, 0], np.exp(Wi[:, 1]), Wi[:, 2], Wi[:, 3], u, R["x_range"][i]).min()
                yc = sim.evaluate(R["f"][i], warps[:, 0], np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u,
                                  R["x_range"][i])
                bad = yc.min(1) < lo - 0.02
                warps[bad] = matched[bad]
    y = sim.evaluate(R["f"][i], warps[:, 0], np.exp(warps[:, 1]), warps[:, 2], warps[:, 3], u, R["x_range"][i])
    # classes along an LR-like axis (first config dimension)
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]
    if version >= 4:   # v4: noise level from step-to-step differences (residual-based sd included template misfit)
        sd = R["noise_sd_diff"][i] * _noise_profile(R["noise_thirds_diff"][i], T)
    else:
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
    # v5: records fitted on all live curves of a task, so 1000 generated configs re-use each real warp vector about
    # once instead of ~5 times (near-duplicate curves differ only in noise and cross at random, which lowered the
    # early-final rank consistency: top-10% kept 0.56 vs 0.72 real)
    R = dict(np.load(PATH_ALL if version >= 5 else PATH))
    P5 = dict(np.load(F.PATH_V5))
    if version >= 3:   # (v4 builds on v3)
        P5["table_rough"] = np.load(ETA_ROUGH)["table"]
    src = np.array([t.split("/")[0] for t in R["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    return [sample_task(R, P5, w, rng, version) for _ in range(n_tasks)]

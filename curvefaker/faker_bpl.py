"""Faker "bpl": broken power laws in schedule-aware progress time, on the log-excess scale.

Motivated by docs/data_semantics.md and scripts/{representation,fpca,schedule}_study.py (journal 2026-11-20, part 2):
  * the log-excess scale v = log(y - L) beats linear loss for curve structure (one component explains 84-94% of
    within-task variance, vs 81-91%) and for predicting finals from partial curves;
  * a logistic in log-time with a floor is a broken power law, y = L + d + A / (1 + (x / tau)^a): plateau, then
    (y - L - d) ~ x^-a, with a config-specific floor offset d >= 0;
  * time is optimiser progress: x = (eta / eta_ref)^kappa * S1(t) / T_total, S1 = cumulative learning-rate
    multiplier of the config's schedule (decayed configs: partial-curve prediction 0.90 -> 0.93 rank corr.);
    the learning rate is a partial time warp (kappa ~ 0.33 on TaskSet);
  * noise size ~ excess^k: k ~ 0 (additive) on TaskSet losses / QuickTune, ~ 1.3 on error-rate sources;
  * validation curves rise after their minimum in a config-dependent subset (classic overfitting).

Record (one per fit task; `fit_records`): metric kind, floor L, start level, the joint distribution of fitted
per-curve parameters (log d, log A, log tau, log a; tau in progress units, so learning rate and schedule are
removed), kappa, learning-rate range, schedule family and the distributions of its parameters, overfitting share /
onset / rise, dead and diverged shares, divergence times, noise (sd, excess exponent k, time profile, t dof).

Generated task (`sample_task`), in the record's raw units:
  1. Configs uniform in [0, 1]^d. Dimension 0 = log learning rate over the record's range. For TaskSet records
     the next dimensions are the schedule parameters of the record's optimiser family (quantile-mapped to the
     real marginals); the rest are generic.
  2. Seven GP fields (random Matern-5/2 kernel, as swt). Field 0, mixed with a rough companion calibrated to the
     record's predictability, gives the floor-offset score; fields 1-3 give the remaining parameter scores,
     correlated with it like the record's real parameters (Gaussian copula); marginals are the real ones.
  3. Curve: y = L + d + A / (1 + (x / tau)^a), x from the config's learning rate and schedule.
  4. Overfitting: a subset (score mixing field 6 and low "regularisation" = high dimension-1 value) rises after
     an onset by a relative amount drawn from the record.
  5. Dead (low learning rate end) / diverged (high end) as in swt; Student-t noise with sd ~ excess^k.
"""
from __future__ import annotations

import math
from multiprocessing import get_context

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import kurtosis, norm, rankdata, theilslopes

from curvefaker import curves, data, schedules
from curvefaker.paths import REPO

PATH = REPO / "results" / "bpl_records.npz"
PATH5 = REPO / "results" / "bpl_records5.npz"    # v5+: floor offset decaying slowly, d * (x / x_end)^-b
ERROR_SOURCES = ("jahs", "nasbench201", "quicktune")
MAX_LIVE = 300
N_FIELDS = 7
LR_AXIS_WEIGHT = 0.7
SCHED_HPS = {"adam6p": ["linear_decay", "exponential_decay"], "adam8p": ["linear_decay", "exponential_decay"],
             "nadamw": ["warmup_fraction", "constant_fraction", "min_learning_rate_mult"]}
OPT_ID = {"const": 0, "adam6p": 1, "adam8p": 1, "nadamw": 2, "cosine": 3}


# ----------------------------------------------------------------------------------------------- schedules
def progress(opt_id, sched, frac, T_total=10000):
    """Cumulative learning-rate multiplier S1(t) / T_total at budget fractions frac [T] for schedule params
    sched (dict). opt_id: 0 constant, 1 linear/exponential decay, 2 warm-up / constant / cosine, 3 cosine to 0."""
    steps = np.arange(0, T_total + 1)
    if opt_id == 0:
        m = np.ones_like(steps, float)
    elif opt_id == 1:
        m = schedules.lr_multiplier("adam6p", sched, steps)
    elif opt_id == 2:
        m = schedules.lr_multiplier("nadamw", sched, steps)
    else:
        m = 0.5 * (1 + np.cos(np.pi * steps / T_total))
    s1 = np.cumsum(m) / T_total
    idx = np.clip(np.round(np.asarray(frac) * T_total).astype(int), 1, T_total)
    return np.maximum(s1[idx], 1e-9)


def _bpl(p, x, L):
    logd, logA, logtau, loga = p[:4]
    floor = np.exp(logd) * ((x / x[-1]) ** (-p[4]) if len(p) > 4 else 1.0)
    return L + floor + np.exp(logA) / (1 + (x / np.exp(logtau)) ** np.exp(loga))


def _fit_curve(y, x, L, n_params=4):
    v = np.log(np.maximum(y - L, 1e-12))

    def res(p):
        return np.log(np.maximum(_bpl(p, x, L) - L, 1e-12)) - v

    tail = max(np.min(y) - L, 1e-6)
    p0 = [np.log(0.5 * tail), np.log(max(y[0] - L, 1e-6)), np.log(np.median(x)), 0.0]
    lo, hi = [-30, -30, -25, -3], [10, 10, 15, 2.5]
    if n_params == 5:   # second, slow stage: the floor offset itself decays as x^-b (two-stage power law)
        p0, lo, hi = p0 + [0.1], lo + [0.0], hi + [1.0]
    try:
        r = least_squares(res, p0, method="trf", max_nfev=300, bounds=(lo, hi))
        return r.x, float(np.sqrt(np.mean(r.fun ** 2)))
    except Exception:
        return None, np.nan


# ----------------------------------------------------------------------------------------------- records
def _schedule_of(t, tid):
    """(opt_id, per-config schedule dicts, eta [n] or None) for a task."""
    n = len(t.hps)
    if t.source == "taskset":
        opt = tid.split("/")[-1]
        oid = OPT_ID.get(opt, 0)
        sch = [dict(r) for _, r in t.hps.iterrows()] if oid else [{}] * n
        return oid, sch, t.hps["learning_rate"].to_numpy(float), opt
    if t.source in ("jahs", "nasbench201"):
        eta = t.hps["LearningRate"].to_numpy(float) if "LearningRate" in t.hps else None
        return OPT_ID["cosine"], [{}] * n, eta, "cosine"
    if t.source == "quicktune":
        return 0, [{}] * n, t.hps["lr"].to_numpy(float), "const"
    return 0, [{}] * n, None, "const"


def _fit_one(args):
    tid, seed = args[:2]
    n_params = args[2] if len(args) > 2 else 4
    rng = np.random.default_rng(seed)
    t = data.load(tid)
    y = t.y[:, 0].astype(float)
    keep = np.arange(len(y))
    if t.source == "quicktune":
        keep = keep[np.isfinite(y[:, -1])]
    if len(keep) > 1000:
        keep = rng.choice(keep, 1000, replace=False)
    y = y[keep]
    oid, sch, eta, opt = _schedule_of(t, tid)
    sch = [sch[k] for k in keep]
    eta = eta[keep] if eta is not None else None
    start = None
    if t.source == "taskset":          # step 0 = before training: the start level; time must be > 0
        start = y[:, 0]
        y = y[:, 1:]
    T = y.shape[1]
    frac = np.arange(1, T + 1) / T
    z, _ = curves.normalise(y)
    ok = np.isfinite(z).all(1)
    c = np.full(len(z), 3)
    c[ok] = curves.classify(z[ok])
    live = np.where((c == 0) | (c == 2))[0]
    out = {"task": tid, "source": t.source, "kind": 1 if t.source in ERROR_SOURCES else 0, "T": T, "n": len(y),
           "n_hps": len(t.space), "opt_id": oid, "opt": opt,
           "share_dead": float((c == 1).mean()), "share_div": float((c == 3).mean()), "ok": False}
    if len(live) < 20:
        return out
    yl = y[live]
    rngt = np.quantile(yl[:, 0], 0.9) - np.nanmin(yl)
    L = float(np.nanmin(yl) - 0.02 * rngt)
    has_start = start is not None and np.isfinite(start[live]).all()
    y0 = float(np.median(start[live])) if has_start else float(np.median(yl[:, 0]))
    start_sd = float(np.std(np.log(np.maximum(start[live] - L, 1e-12)))) if has_start else 0.0
    # kappa: learning rate as a partial time warp
    x_sched = np.stack([progress(oid, sch[i], frac) for i in live]) if oid in (1, 2) else \
        np.tile(progress(oid, {}, frac), (len(live), 1))
    pick = rng.choice(len(live), min(MAX_LIVE, len(live)), replace=False)
    kappa, eta_ref, lr_q, lr_q_live = 0.0, 1.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    if eta is not None and np.all(eta > 0) and np.std(np.log(eta[live])) > 0.3:
        le = np.log(eta)
        eta_ref = float(np.exp(np.median(le[live])))
        lr_q = np.quantile(le, [0.0, 1.0])
        lr_q_live = np.quantile(le[live], [0.02, 0.98])
        sub = pick[:80]
        taus = np.array([(_fit_curve(yl[k], x_sched[k], L)[0] if True else None) for k in sub], dtype=object)
        good = [j for j, p in enumerate(taus) if p is not None]
        if len(good) >= 20:
            lt = np.array([taus[j][2] for j in good])
            lv = le[live][sub][good]
            q = (lt > np.quantile(lt, 0.1)) & (lt < np.quantile(lt, 0.9))
            if q.sum() >= 10 and lv[q].std() > 0:
                kappa = float(np.clip(-theilslopes(lt[q], lv[q])[0], 0.0, 1.0))
    lrfac = (eta[live] / eta_ref) ** kappa if eta is not None and kappa > 0 else np.ones(len(live))
    params, rmses, xT, leta = [], [], [], []
    for k in pick:
        p, r = _fit_curve(yl[k], lrfac[k] * x_sched[k], L, n_params)
        if p is not None and np.isfinite(r):
            params.append(p)
            rmses.append(r)
            xT.append(float(lrfac[k] * x_sched[k][-1]))   # the curve's own progress at the end of the budget
            leta.append(float(np.log(eta[live][k])) if eta is not None and np.all(eta > 0) else 0.0)
    if len(params) < 20:
        return out
    P = np.array(params)
    S = norm.ppf(np.apply_along_axis(rankdata, 0, P[:, :4]) / (len(P) + 1))
    C = np.nan_to_num(np.corrcoef(S.T), nan=0.0)
    np.fill_diagonal(C, 1.0)
    # overfitting (audit criterion) among live curves
    s = curves.smooth(yl)
    m = s.argmin(1)
    drop = s[:, 0] - s[np.arange(len(s)), m]
    dd = np.diff(yl, axis=1)
    sdn = np.median(np.abs(dd - np.median(dd, 1, keepdims=True)), 1) * 1.4826 / math.sqrt(2)
    rise = s[:, -1] - s[np.arange(len(s)), m]
    over = (m < 0.9 * T) & (rise >= np.maximum(0.1 * drop, 3 * sdn))
    # noise: robust sd of differences, excess exponent k, time profile, tail weight
    resid = (yl - s)[:, 2:-2]
    ex = np.maximum((s - L)[:, 2:-2], 1e-12)
    qs = np.quantile(ex, np.linspace(0, 1, 9))
    xs, ys = [], []
    for a, b in zip(qs[:-1], qs[1:]):
        mm = (ex >= a) & (ex <= b)
        if mm.sum() > 50:
            rr = resid[mm]
            xs.append(np.log(np.median(ex[mm])))
            ys.append(np.log(np.median(np.abs(rr - np.median(rr))) * 1.4826 + 1e-12))
    k_ex = float(np.clip(np.polyfit(xs, ys, 1)[0], 0.0, 1.5)) if len(xs) >= 4 else 0.0
    mad = lambda a: np.median(np.abs(a - np.median(a))) * 1.4826 / math.sqrt(2)
    med_ex0 = float(np.median(ex))
    # differences with the excess scaling removed, so the time profile does not count the decline of the excess
    # a second time (the excess falls over training)
    ex_d = np.maximum(0.5 * (s[:, 1:] + s[:, :-1]) - L, 1e-12)
    dd_n = dd / np.minimum((ex_d / med_ex0) ** k_ex, 20.0)
    sd_d = mad(dd_n)
    thirds = [mad(dd_n[:, th]) / max(sd_d, 1e-12) for th in np.array_split(np.arange(T - 1), 3)]
    # v6: second differences cancel a locally linear trend, so the steep early drop does not count as noise
    # (first differences inflated early noise ~2x on TaskSet; late/early ratio 0.04 vs 0.08 on the same records)
    d2n = np.diff(yl, 2, axis=1) / np.minimum((ex_d[:, 1:] / med_ex0) ** k_ex, 20.0)
    mad2 = lambda a: np.median(np.abs(a - np.median(a))) * 1.4826 / math.sqrt(6)
    sd_2 = mad2(d2n)
    thirds2 = [mad2(d2n[:, th]) / max(sd_2, 1e-12) for th in np.array_split(np.arange(T - 2), 3)]
    kk = float(np.median(kurtosis(resid / max(np.median(np.abs(resid)) * 1.4826, 1e-12), axis=1)))
    med_ex = float(np.median(ex))
    # dead level and divergence times
    dead_level = float(np.median(curves.final_value(y[c == 1]))) if (c == 1).any() else y0
    zc = np.nan_to_num(z, nan=curves.CAP, posinf=curves.CAP)
    div = np.where(c == 3)[0]
    div_times = np.array([(np.argmax(zc[i] >= curves.CAP) if (zc[i] >= curves.CAP).any() else T - 1) / (T - 1)
                          for i in div], float)
    sched_q = np.zeros((3, 101))
    if opt in SCHED_HPS:
        for j, h in enumerate(SCHED_HPS[opt]):
            sched_q[j] = np.quantile(t.hps[h].to_numpy(float), np.linspace(0, 1, 101))
    out.update({"ok": True, "L": L, "y0": y0, "has_start": float(has_start), "start_sd": start_sd, "range": float(rngt), "kappa": kappa, "eta_ref": eta_ref,
                "lr_q": lr_q, "lr_q_live": lr_q_live, "params": P, "xT": np.log(np.array(xT)), "leta": np.array(leta), "copula": C, "fit_rmse": float(np.median(rmses)),
                "share_over": float(over.mean()), "over_onset": (m[over] / T).astype(float),
                "over_rise": (rise[over] / max(y0 - L, 1e-12)).astype(float),
                "noise_sd": float(sd_d), "noise_sd2": float(sd_2), "noise_thirds2": np.array(thirds2), "noise_k": k_ex, "noise_med_ex": med_ex, "noise_thirds": np.array(thirds),
                "df": float(np.clip(4 + 6 / max(kk, 1e-3), 4.5, 50)) if kk > 0 else 50.0,
                "dead_level": dead_level, "div_times": div_times, "sched_q": sched_q})
    return out


def fit_records(processes=64, path=None, n_params=4):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from curvefaker import faker_hgp as F
    P = dict(np.load(F.PATH_V1))
    jobs = [(str(t), i, n_params) for i, t in enumerate(P["task"])]
    with get_context("spawn").Pool(processes) as pool:
        recs = pool.map(_fit_one, jobs, chunksize=1)
    out = {"task": np.array([r["task"] for r in recs]), "source": np.array([r["source"] for r in recs]),
           "opt": np.array([r["opt"] for r in recs]), "ok": np.array([r["ok"] for r in recs])}
    for k in ("kind", "T", "n", "n_hps", "opt_id", "share_dead", "share_div"):
        out[k] = np.array([r[k] for r in recs], float)
    scal = {"L": 0.0, "y0": 1.0, "has_start": 0.0, "start_sd": 0.0, "range": 1.0, "kappa": 0.0, "eta_ref": 1.0, "fit_rmse": np.nan, "share_over": 0.0,
            "noise_sd": 0.0, "noise_sd2": 0.0, "noise_k": 0.0, "noise_med_ex": 1.0, "df": 50.0, "dead_level": 1.0}
    for k, dflt in scal.items():
        out[k] = np.array([r.get(k, dflt) for r in recs], float)
    out["lr_q"] = np.stack([r.get("lr_q", np.array([0.0, 1.0])) for r in recs])
    out["lr_q_live"] = np.stack([r.get("lr_q_live", np.array([0.0, 1.0])) for r in recs])
    out["copula"] = np.stack([r.get("copula", np.eye(4)) for r in recs])
    out["noise_thirds"] = np.stack([r.get("noise_thirds", np.ones(3)) for r in recs])
    out["noise_thirds2"] = np.stack([r.get("noise_thirds2", np.ones(3)) for r in recs])
    out["sched_q"] = np.stack([r.get("sched_q", np.zeros((3, 101))) for r in recs])
    for k, width in (("params", n_params), ("xT", 0), ("leta", 0), ("over_onset", 0), ("over_rise", 0), ("div_times", 0)):
        A = [r.get(k, np.zeros((0, width)) if width else np.zeros(0)) for r in recs]
        out[k] = np.concatenate(A) if sum(len(a) for a in A) else (np.zeros((0, width)) if width else np.zeros(0))
        out[k + "_offs"] = np.concatenate([[0], np.cumsum([len(a) for a in A])])
    np.savez(path or (PATH5 if n_params == 5 else PATH), **out)
    return out


# ----------------------------------------------------------------------------------------------- generation
def _fields(rng, X, n_fields, rough_scale=0.15):
    from curvefaker.faker_hgp import _matern52
    n, d = X.shape
    r = rng.dirichlet(np.full(d, 0.7))
    ell = math.exp(rng.uniform(math.log(0.2), 0.0)) / np.sqrt(r / r.max())
    w = rng.random()
    out = []
    for scale in (1.0, rough_scale):
        Kg = np.stack([_matern52(X[:, None, j] - X[None, :, j], ell[j] * scale) for j in range(d)])
        Kx = w * np.tensordot(r, Kg, 1) + (1 - w) * Kg.prod(0)
        f = np.linalg.cholesky(Kx + 1e-6 * np.eye(n)) @ rng.standard_normal((n, n_fields))
        out.append((f - f.mean(0)) / f.std(0))
    return out


def _h50(logtau, loga, x, frac):
    """Log budget fraction at which A / (1 + (x / tau)^a) has made 50% of its drop over the window, per curve.
    logtau, loga [m]; x [m, T]."""
    f = 1 / (1 + (x / np.exp(logtau)[:, None]) ** np.exp(loga)[:, None])
    p = (f[:, :1] - f) / np.maximum(f[:, :1] - f[:, -1:], 1e-12)
    k = np.argmax(p >= 0.5, axis=1)
    return np.log(frac[k])


def _seg(R, k, i):
    return R[k][R[k + "_offs"][i]:R[k + "_offs"][i + 1]]


def sample_task(R, P5, w, rng, version=1, return_meta=False):
    from curvefaker import faker_hgp as F
    i = int(rng.choice(len(w), p=w))
    T, n = int(R["T"][i]), int(R["n"][i])
    d = int(np.clip(R["n_hps"][i], 1, 10))
    frac = np.arange(1, T + 1) / T
    X = rng.random((n, d))
    L, y0, kappa = R["L"][i], R["y0"][i], R["kappa"][i]
    # learning rate and schedule from the config
    # v3: dimension 0 spans the learning-rate range of the record's *live* configs. Over the full range (TaskSet:
    # ~1e-8 to 10) generated live configs reached learning rates at which real configs diverge, and the time warp
    # (lr^kappa) made their curves converge far too early (time to 90% progress -1.6 vs -1.2 on the same records).
    # Dead / diverged configs still take the two ends of the axis.
    lo, hi = R["lr_q_live"][i] if version >= 3 else R["lr_q"][i]
    log_eta = lo + X[:, 0] * (hi - lo)
    oid, opt = int(R["opt_id"][i]), str(R["opt"][i])
    if oid in (1, 2):
        names = SCHED_HPS[opt]
        cols = [1 + j for j in range(len(names)) if 1 + j < d]
        sched = []
        for a in range(n):
            hp = {h: float(np.interp(X[a, c], np.linspace(0, 1, 101), R["sched_q"][i][j]))
                  for j, (h, c) in enumerate(zip(names, cols))}
            for h in names:
                hp.setdefault(h, float(R["sched_q"][i][names.index(h)][50]))
            sched.append(hp)
        xs = np.stack([progress(oid, s, frac) for s in sched])
    else:
        xs = np.tile(progress(oid, {}, frac), (n, 1))
    lrfac = np.exp(kappa * (log_eta - math.log(R["eta_ref"][i]))) if kappa > 0 else np.ones(n)
    x = lrfac[:, None] * xs
    # parameter scores: field 0 (with rough part) -> floor offset; others via the copula
    eta_mix = F._eta_for(P5["table_rough"][d], P5["rho"][i])
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
    if version in (13, 14) and "leta" in R and np.ptp(_seg(R, "leta", i)) > 0.3:
        # v13: v1's copula (smooth parameter fields: best HPO behaviour over three generation seeds, freeze-thaw
        # gain 0.113 vs real 0.112) plus v10's working idea: each parameter's normal score gets the record's real
        # learning-rate relation as conditional mean (10 quantile bins), the fields scaled to the remaining variance
        le_r = _seg(R, "leta", i)
        Sr = norm.ppf(np.apply_along_axis(rankdata, 0, P[:, :4]) / (len(P) + 1))
        edges = np.quantile(le_r, np.linspace(0, 1, 11))
        mids = 0.5 * (edges[1:] + edges[:-1])
        for j in range(4):
            mb = np.array([Sr[(le_r >= a) & (le_r <= b), j].mean() if ((le_r >= a) & (le_r <= b)).any() else 0.0
                           for a, b in zip(edges[:-1], edges[1:])])
            r2 = float(np.clip(1 - np.var(Sr[:, j] - np.interp(le_r, mids, mb)) / max(np.var(Sr[:, j]), 1e-12),
                               0, 0.95))
            zsc[:, j] = np.interp(log_eta, mids, mb) + math.sqrt(1 - r2) * zsc[:, j]
    if version >= 14 and "leta" in R and np.ptp(_seg(R, "leta", i)) > 0.3:
        # v14: v13's smooth shape parameters, but the final value targets the record's real final values: the
        # copula preserves each parameter's marginal, not the distribution of the finals they produce together
        # (v13: median final 0.24 vs 0.48 real, near-best share 0.20 vs 0.14, freeze-thaw gain too large). Target =
        # field 0 (with the rough part) + the real learning-rate relation of the finals; d solved to hit it.
        q = norm.cdf(zsc)
        par = np.stack([np.quantile(P[:, j], q[:, j]) for j in range(4)], 1)
        xref = progress(oid, {h: float(R["sched_q"][i][j][50]) for j, h in enumerate(SCHED_HPS.get(opt, []))},
                        frac) if oid in (1, 2) else progress(oid, {}, frac)
        fin_r = np.exp(P[:, 0]) + np.exp(P[:, 1]) / (1 + (np.exp(_seg(R, "xT", i)) / np.exp(P[:, 2])) **
                                                     np.exp(P[:, 3]))
        zf = norm.ppf(rankdata(fin_r) / (len(P) + 1))
        mb = np.array([zf[(le_r >= a) & (le_r <= b)].mean() if ((le_r >= a) & (le_r <= b)).any() else 0.0
                       for a, b in zip(edges[:-1], edges[1:])])
        r2 = float(np.clip(1 - np.var(zf - np.interp(le_r, mids, mb)) / max(np.var(zf), 1e-12), 0, 0.95))
        tf = np.interp(log_eta, mids, mb) + math.sqrt(1 - r2) * g[:, 0]
        fe_t = np.quantile(fin_r, norm.cdf(np.clip(tf, -3.5, 3.5)))
        cT = np.exp(par[:, 1]) / (1 + (x[:, -1] / np.exp(par[:, 2])) ** np.exp(par[:, 3]))
        par[:, 0] = np.log(np.maximum(fe_t - cT, 1e-4 * (y0 - L)))
    elif 2 <= version < 13 or version > 14:
        # v2: joint real parameter vectors. tau and the exponent trade off in the fits (curves without a visible
        # plateau), so independent quantile maps (v1) created combinations no real curve has and curves converged
        # too early (time to 90% progress -1.7 vs -1.1 log budget fraction). Each config takes the real vector
        # nearest to its targets (final level from field 0, speed from field 1, correlated as in the record); the
        # floor offset d is then solved so that the curve ends at the target final value for the config's own
        # learning rate and schedule (d >= 0: a too small learning rate cannot reach it).
        xref = progress(oid, {h: float(R["sched_q"][i][j][50]) for j, h in enumerate(SCHED_HPS.get(opt, []))},
                        frac) if oid in (1, 2) else progress(oid, {}, frac)
        xT = xref[-1]
        fin_r = np.exp(P[:, 0]) + np.exp(P[:, 1]) / (1 + (xT / np.exp(P[:, 2])) ** np.exp(P[:, 3]))
        zs = norm.ppf(np.stack([rankdata(fin_r), rankdata(P[:, 2])], 1) / (len(P) + 1))
        rho = float(np.corrcoef(zs.T)[0, 1])
        tgt = np.stack([g[:, 0], rho * g[:, 0] + math.sqrt(max(1 - rho ** 2, 0)) * g[:, 1]], 1)
        tgt = (tgt - tgt.mean(0)) / tgt.std(0)
        d2 = 4.0 * (tgt[:, None, 0] - zs[None, :, 0]) ** 2 + (tgt[:, None, 1] - zs[None, :, 1]) ** 2
        if 4 <= version < 7 or version in (8, 9):
            # v4: a real vector's time scale is relative to the progress its own config made (schedule, learning
            # rate); match it to configs with a similar progress budget x_T, so a vector fitted under a constant
            # learning rate is not paired with a schedule that stops progress early (curves froze too early)
            lxr = _seg(R, "xT", i)
            lxg = np.log(x[:, -1])
            sdx = max(lxr.std(), 1e-3)
            d2 = d2 + 4.0 * ((lxg[:, None] - lxr[None, :]) / sdx) ** 2
        if version == 7:
            # v7: speed constructed. The nearest-vector match (v2-v6) left curve speed hardly predictable from the
            # config (R^2 0.17-0.25 vs 0.73 real; v1 0.48). Relative speed rs = log tau - log x_T (where the curve
            # sits in its S-shape at the end of its own budget) becomes the speed target's quantile of the record's
            # real values, and tau follows from the config's own progress budget (learning rate, schedule).
            # Matching uses (final, relative speed); amplitude and exponent come from the nearest real vector.
            rs = P[:, 2] - _seg(R, "xT", i)
            zs[:, 1] = norm.ppf(rankdata(rs) / (len(P) + 1))
            rho = float(np.corrcoef(zs.T)[0, 1])
            tgt = np.stack([g[:, 0], rho * g[:, 0] + math.sqrt(max(1 - rho ** 2, 0)) * g[:, 1]], 1)
            tgt = (tgt - tgt.mean(0)) / tgt.std(0)
            d2 = 4.0 * (tgt[:, None, 0] - zs[None, :, 0]) ** 2 + (tgt[:, None, 1] - zs[None, :, 1]) ** 2
        if version >= 10 and "leta" in R and kappa >= 0 and np.ptp(_seg(R, "leta", i)) > 0.3:
            # v10: the record's real learning-rate relation. On real TaskSet tasks speed is predictable from the
            # config (R^2 0.90) and from the learning rate alone (0.58), not without it; generated speed was 0.44
            # (0.37 with the speed field off): shape parameters came from vectors chosen without regard to the
            # learning rate. Targets = conditional mean of the real scores given log learning rate (10 quantile
            # bins) + the fields scaled to the remaining variance; matching also on log learning rate.
            le_r = _seg(R, "leta", i)
            edges = np.quantile(le_r, np.linspace(0, 1, 11))
            mids = 0.5 * (edges[1:] + edges[:-1])
            tgt_lr = np.zeros((n, 2))
            for j in range(2):
                mb = np.array([zs[(le_r >= a) & (le_r <= b), j].mean() if ((le_r >= a) & (le_r <= b)).any() else 0.0
                               for a, b in zip(edges[:-1], edges[1:])])
                fitted = np.interp(le_r, mids, mb)
                r2 = float(np.clip(1 - np.var(zs[:, j] - fitted) / max(np.var(zs[:, j]), 1e-12), 0, 0.95))
                tgt_lr[:, j] = np.interp(log_eta, mids, mb)
                tgt[:, j] = tgt_lr[:, j] + math.sqrt(1 - r2) * tgt[:, j]
            sdl = max(le_r.std(), 1e-3)
            d2 = 4.0 * (tgt[:, None, 0] - zs[None, :, 0]) ** 2 + (tgt[:, None, 1] - zs[None, :, 1]) ** 2 + \
                4.0 * ((log_eta[:, None] - le_r[None, :]) / sdl) ** 2
            lxr = _seg(R, "xT", i)
            d2 = d2 + 4.0 * ((np.log(x[:, -1])[:, None] - lxr[None, :]) / max(lxr.std(), 1e-3)) ** 2
            q0 = norm.cdf(np.clip(tgt[:, 0], -3.5, 3.5))
        else:
            q0 = norm.cdf(tgt[:, 0])
        par = P[d2.argmin(1)].copy()
        if version == 7:
            par[:, 2] = np.quantile(rs, norm.cdf(tgt[:, 1])) + np.log(x[:, -1])
        if version in (8, 9):
            # v8: speed targeted directly. "Speed" = log budget fraction at which a curve has made 50% of its
            # (template) progress, h50. Real vectors' own h50 is computed on their own progress path (the record's
            # reference path scaled to their x_T); each config's target is the speed field's quantile of these,
            # and tau is solved (grid over log tau) so that the config's curve, on its own path, reaches 50% then.
            lxr = _seg(R, "xT", i)
            own = np.exp(lxr)[:, None] * (xref / xref[-1])[None, :]
            h_r = _h50(P[:, 2], P[:, 3], own, frac)
            h_t = np.quantile(h_r, norm.cdf(tgt[:, 1]))
            grid = np.linspace(-12, 12, 97)
            best = np.full(n, np.inf)
            lt_best = par[:, 2].copy()
            for dl in grid:
                lt = np.log(x[:, -1]) + dl
                err = np.abs(_h50(lt, par[:, 3], x, frac) - h_t)
                b = err < best
                best[b], lt_best[b] = err[b], lt[b]
            par[:, 2] = lt_best
        fe_t = np.quantile(fin_r, q0)
        cT = np.exp(par[:, 1]) / (1 + (x[:, -1] / np.exp(par[:, 2])) ** np.exp(par[:, 3]))
        par[:, 0] = np.log(np.maximum(fe_t - cT, 1e-4 * (y0 - L)))
        if version in (9, 12) and R["has_start"][i] > 0:
            # v9: start-consistent curves (TaskSet, where step 0 = the shared initial loss). Amplitude A and floor
            # offset d are solved together so the curve starts at the start level s0 (with the real spread) and
            # ends at its target final value: d * c1 + A * f1 = s0 - L, d + A * fT = fe_t, with f the template
            # term at the first observed progress / at the end and c1 the floor-decay factor at the first point.
            # (v1-v8 drew the step-0 value independently of the curve's own plateau; that random jump dominated
            # the measured speed: speed-from-config R^2 ~0.2.)
            s0 = L + (y0 - L) * np.exp(R["start_sd"][i] * rng.standard_normal(n))
            f1 = 1 / (1 + (x[:, 0] / np.exp(par[:, 2])) ** np.exp(par[:, 3]))
            fT = 1 / (1 + (x[:, -1] / np.exp(par[:, 2])) ** np.exp(par[:, 3]))
            c1 = (x[:, 0] / x[:, -1]) ** (-par[:, 4]) if par.shape[1] > 4 else np.ones(n)
            det = c1 * fT - f1
            A_ = np.where(np.abs(det) > 1e-9, (c1 * fe_t - (s0 - L)) / np.where(np.abs(det) > 1e-9, det, 1), np.nan)
            d_ = fe_t - A_ * fT
            okk = np.isfinite(A_) & (A_ > 0) & (d_ > 0)
            par[okk, 1] = np.log(A_[okk])
            par[okk, 0] = np.log(d_[okk])
            start_val = s0
    else:
        q = norm.cdf(zsc)
        par = np.stack([np.quantile(P[:, j], q[:, j]) for j in range(4)], 1)
    floor = np.exp(par[:, 0:1]) * ((x / x[:, -1:]) ** (-par[:, 4:5]) if par.shape[1] > 4 else 1.0)
    y = L + floor + np.exp(par[:, 1:2]) / (1 + (x / np.exp(par[:, 2:3])) ** np.exp(par[:, 3:4]))
    clean_ex = y - L
    # classes along the learning-rate axis
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]
    # overfitting
    on, ri = _seg(R, "over_onset", i), _seg(R, "over_rise", i)
    over = np.array([], int)
    if version == 11:
        # v11: only configs that are neither dead nor diverged can overfit (27% of planted overfitters were
        # overwritten before), their number follows the record's share among live curves, and after the onset
        # the curve holds its value and rises from there, so its minimum sits at the onset as in real curves
        # (with the rise added to a still-descending curve the minimum often fell after 90% of the budget and the
        # rise was not detected: 1.1% of live curves rose vs 3.9% real)
        elig = np.setdiff1d(np.arange(n), np.concatenate([dead, div]))
        no = int(round(R["share_over"][i] * len(elig)))
    else:
        elig = np.arange(n)
        no = int(round(R["share_over"][i] * n))
    if no and len(on):
        reg = X[:, 1] if d > 1 else rng.random(n)
        score = 0.5 * g[:, 6] + 0.5 * (reg - 0.5) / math.sqrt(1 / 12)
        over = elig[np.argsort(-score[elig])[:no]]
        j = rng.integers(len(on), size=no)
        o, r = on[j], ri[j]
        ramp = np.clip((frac[None, :] - o[:, None]) / np.maximum(1 - o[:, None], 1e-3), 0, None)
        if version == 11:
            k_on = np.clip(np.round(o * T).astype(int), 0, T - 1)
            held = np.where(np.arange(T)[None, :] >= k_on[:, None], y[over, k_on][:, None], y[over])
            y[over] = held + r[:, None] * (y0 - L) * ramp
        else:
            y[over] += r[:, None] * (y0 - L) * ramp
    # noise: sd ~ (excess / median excess)^k, time profile, Student-t
    v6 = version == 6 and "noise_sd2" in R     # second-difference noise: tried in v6, worse; not kept
    prof = np.interp(np.linspace(0, 1, T), [1 / 6, 1 / 2, 5 / 6], R["noise_thirds2" if v6 else "noise_thirds"][i])
    scale = (np.maximum(clean_ex, 1e-12) / R["noise_med_ex"][i]) ** R["noise_k"][i]
    sd = R["noise_sd2" if v6 else "noise_sd"][i] * prof[None, :] * np.minimum(scale, 20.0)
    y = y + rng.standard_t(R["df"][i], (n, T)) * sd
    y[dead] = R["dead_level"][i] + rng.standard_t(R["df"][i], (len(dead), T)) * \
        R["noise_sd2" if v6 else "noise_sd"][i] * prof
    dt = _seg(R, "div_times", i)
    hi_val = y0 + 10 * (y0 - L)     # diverged: far above the start; normalisation caps it
    for a in div:
        t0 = rng.choice(dt) if len(dt) else rng.random()
        y[a, int(round(t0 * (T - 1))):] = hi_val
    if R["has_start"][i] > 0:      # TaskSet: the step-0 point (before training), all configs near the start level
        if version in (9, 12):   # the curve's own plateau (template term at its maximum, floor at the first point)
            c1 = (x[:, 0] / x[:, -1]) ** (-par[:, 4]) if par.shape[1] > 4 else np.ones(n)
            st = L + np.exp(par[:, 0]) * c1 + np.exp(par[:, 1])
        else:
            st = L + (y0 - L) * np.exp(R["start_sd"][i] * rng.standard_normal(n))
        st[div] = np.maximum(st[div], y0)
        y = np.concatenate([st[:, None], y], 1)
    if R["kind"][i] == 1 and R["source"][i] in ("jahs", "quicktune"):
        y = np.clip(y, 0, 100)
    elif R["kind"][i] == 1:
        y = np.clip(y, 0, 1)
    out = (y[:, None, :].astype(np.float32), X)
    if return_meta:
        return out + ({"record": i, "source": str(R["source"][i]), "kind": int(R["kind"][i]), "opt": opt,
                       "dead": dead, "div": div, "over": over},)
    return out


def faker(n_tasks, rng, version=1):
    from curvefaker import faker_hgp as F
    from curvefaker import faker_swt as S
    R = dict(np.load(PATH5 if version >= 5 and version not in (13, 14) else PATH))
    P5 = dict(np.load(F.PATH_V5))
    P5["table_rough"] = np.load(S.ETA_ROUGH)["table"]
    rl = np.load(S.RHO_LIVE)["rho"]
    P5["rho"] = np.where(np.isfinite(rl), rl, P5["rho"])
    src = R["source"]
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    return [sample_task(R, P5, w, rng, version) for _ in range(n_tasks)]

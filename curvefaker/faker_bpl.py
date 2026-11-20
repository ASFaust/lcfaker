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
    logd, logA, logtau, loga = p
    return L + np.exp(logd) + np.exp(logA) / (1 + (x / np.exp(logtau)) ** np.exp(loga))


def _fit_curve(y, x, L):
    v = np.log(np.maximum(y - L, 1e-12))

    def res(p):
        return np.log(np.maximum(_bpl(p, x, L) - L, 1e-12)) - v

    tail = max(np.min(y) - L, 1e-6)
    p0 = [np.log(0.5 * tail), np.log(max(y[0] - L, 1e-6)), np.log(np.median(x)), 0.0]
    try:
        r = least_squares(res, p0, method="trf", max_nfev=300,
                          bounds=([-30, -30, -25, -3], [10, 10, 15, 2.5]))
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
    tid, seed = args
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
    kappa, eta_ref, lr_q = 0.0, 1.0, np.array([0.0, 1.0])
    if eta is not None and np.all(eta > 0) and np.std(np.log(eta[live])) > 0.3:
        le = np.log(eta)
        eta_ref = float(np.exp(np.median(le[live])))
        lr_q = np.quantile(le, [0.0, 1.0])
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
    params, rmses = [], []
    for k in pick:
        p, r = _fit_curve(yl[k], lrfac[k] * x_sched[k], L)
        if p is not None and np.isfinite(r):
            params.append(p)
            rmses.append(r)
    if len(params) < 20:
        return out
    P = np.array(params)
    S = norm.ppf(np.apply_along_axis(rankdata, 0, P) / (len(P) + 1))
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
                "lr_q": lr_q, "params": P, "copula": C, "fit_rmse": float(np.median(rmses)),
                "share_over": float(over.mean()), "over_onset": (m[over] / T).astype(float),
                "over_rise": (rise[over] / max(y0 - L, 1e-12)).astype(float),
                "noise_sd": float(sd_d), "noise_k": k_ex, "noise_med_ex": med_ex, "noise_thirds": np.array(thirds),
                "df": float(np.clip(4 + 6 / max(kk, 1e-3), 4.5, 50)) if kk > 0 else 50.0,
                "dead_level": dead_level, "div_times": div_times, "sched_q": sched_q})
    return out


def fit_records(processes=64, path=None):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from curvefaker import faker_hgp as F
    P = dict(np.load(F.PATH_V1))
    jobs = [(str(t), i) for i, t in enumerate(P["task"])]
    with get_context("spawn").Pool(processes) as pool:
        recs = pool.map(_fit_one, jobs, chunksize=1)
    out = {"task": np.array([r["task"] for r in recs]), "source": np.array([r["source"] for r in recs]),
           "opt": np.array([r["opt"] for r in recs]), "ok": np.array([r["ok"] for r in recs])}
    for k in ("kind", "T", "n", "n_hps", "opt_id", "share_dead", "share_div"):
        out[k] = np.array([r[k] for r in recs], float)
    scal = {"L": 0.0, "y0": 1.0, "has_start": 0.0, "start_sd": 0.0, "range": 1.0, "kappa": 0.0, "eta_ref": 1.0, "fit_rmse": np.nan, "share_over": 0.0,
            "noise_sd": 0.0, "noise_k": 0.0, "noise_med_ex": 1.0, "df": 50.0, "dead_level": 1.0}
    for k, dflt in scal.items():
        out[k] = np.array([r.get(k, dflt) for r in recs], float)
    out["lr_q"] = np.stack([r.get("lr_q", np.array([0.0, 1.0])) for r in recs])
    out["copula"] = np.stack([r.get("copula", np.eye(4)) for r in recs])
    out["noise_thirds"] = np.stack([r.get("noise_thirds", np.ones(3)) for r in recs])
    out["sched_q"] = np.stack([r.get("sched_q", np.zeros((3, 101))) for r in recs])
    for k, width in (("params", 4), ("over_onset", 0), ("over_rise", 0), ("div_times", 0)):
        A = [r.get(k, np.zeros((0, width)) if width else np.zeros(0)) for r in recs]
        out[k] = np.concatenate(A) if sum(len(a) for a in A) else (np.zeros((0, width)) if width else np.zeros(0))
        out[k + "_offs"] = np.concatenate([[0], np.cumsum([len(a) for a in A])])
    np.savez(path or PATH, **out)
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
    lo, hi = R["lr_q"][i]
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
    q = norm.cdf(zsc)
    par = np.stack([np.quantile(P[:, j], q[:, j]) for j in range(4)], 1)
    y = L + np.exp(par[:, 0:1]) + np.exp(par[:, 1:2]) / (1 + (x / np.exp(par[:, 2:3])) ** np.exp(par[:, 3:4]))
    clean_ex = y - L
    # overfitting
    no = int(round(R["share_over"][i] * n))
    on, ri = _seg(R, "over_onset", i), _seg(R, "over_rise", i)
    over = np.array([], int)
    if no and len(on):
        reg = X[:, 1] if d > 1 else rng.random(n)
        score = 0.5 * g[:, 6] + 0.5 * (reg - 0.5) / math.sqrt(1 / 12)
        over = np.argsort(-score)[:no]
        j = rng.integers(len(on), size=no)
        o, r = on[j], ri[j]
        ramp = np.clip((frac[None, :] - o[:, None]) / np.maximum(1 - o[:, None], 1e-3), 0, None)
        y[over] += r[:, None] * (y0 - L) * ramp
    # classes along the learning-rate axis
    ax = (X[:, 0] - 0.5) / math.sqrt(1 / 12)
    dead_score = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[:, 4]
    div_score = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[:, 5]
    nd, nv = int(round(R["share_dead"][i] * n)), int(round(R["share_div"][i] * n))
    dead = np.argsort(-dead_score)[:nd]
    rest = np.setdiff1d(np.arange(n), dead)
    div = rest[np.argsort(-div_score[rest])[:nv]]
    # noise: sd ~ (excess / median excess)^k, time profile, Student-t
    prof = np.interp(np.linspace(0, 1, T), [1 / 6, 1 / 2, 5 / 6], R["noise_thirds"][i])
    scale = (np.maximum(clean_ex, 1e-12) / R["noise_med_ex"][i]) ** R["noise_k"][i]
    sd = R["noise_sd"][i] * prof[None, :] * np.minimum(scale, 20.0)
    y = y + rng.standard_t(R["df"][i], (n, T)) * sd
    y[dead] = R["dead_level"][i] + rng.standard_t(R["df"][i], (len(dead), T)) * R["noise_sd"][i] * prof
    dt = _seg(R, "div_times", i)
    hi_val = y0 + 10 * (y0 - L)     # diverged: far above the start; normalisation caps it
    for a in div:
        t0 = rng.choice(dt) if len(dt) else rng.random()
        y[a, int(round(t0 * (T - 1))):] = hi_val
    if R["has_start"][i] > 0:      # TaskSet: the step-0 point (before training), all configs near the start level
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
    R = dict(np.load(PATH))
    P5 = dict(np.load(F.PATH_V5))
    P5["table_rough"] = np.load(S.ETA_ROUGH)["table"]
    rl = np.load(S.RHO_LIVE)["rho"]
    P5["rho"] = np.where(np.isfinite(rl), rl, P5["rho"])
    src = R["source"]
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    return [sample_task(R, P5, w, rng, version) for _ in range(n_tasks)]

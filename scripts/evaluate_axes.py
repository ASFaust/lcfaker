"""Per-axis evaluation of fakers (docs/analysis_plan.md, section 1). Real reference: the select tasks of
scripts/evaluate_fakers.py; equal numbers of generated tasks; one seed of each.

Axis B (curves). Pooled curves (one seed, normalised per task, resampled to 40 points evenly in log-time over the
budget fraction 0.02-1). Live curves (gain >= 0.1) for shape features:
    u10 / u50 / u90  log-time at 10 / 50 / 90% of the curve's progress
    late             progress made in the last 20% of the budget
    rise_share       share of steps of the smoothed curve that go up (get worse)
    over             curve rises after its minimum (min before 90% of the budget, rise >= max(10% of the drop,
                     3 noise sd)); B_share_over = share among live curves (real: B_real_share_over)
    max_rise         largest rise of the smoothed curve after its minimum so far (z units)
    final            final value (all curves)
  gap per feature: 1-D Wasserstein distance / real sd.
  c2st_auc: cross-validated AUC of boosted trees telling real from generated curves (0.5 = indistinguishable).
  precision / recall: k-NN (k = 5) precision / recall of generated vs real curves in an 8-dim PCA embedding.
Axis C (noise). Residual = curve - its smoothed version (live curves):
    kurtosis (excess), spike rate (|r| > 5 robust sd), sd in the late third / first third.
Axis A (landscape), per task: landscape_rho at 200 and at 50 configs (curvefaker.evaluate.landscape_rho),
    near_best (share within 0.05 z of the best), shape_from_config (cross-validated R^2 of u50 from the config,
    live curves), local optima ratio on a 10-NN graph.
Output: results/axes_<faker>.json and a printed comparison.

    PYTHONPATH=. python scripts/evaluate_axes.py faker [faker ...]
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))
import evaluate_fakers as EF  # noqa: E402
from curvefaker import curves, evaluate  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

G = 40
GRID = np.exp(np.linspace(np.log(0.02), 0, G))
U = np.log(GRID)
RNG = np.random.default_rng(20)


def logres(z):
    T = z.shape[1]
    t = np.arange(1, T + 1) / T
    return np.stack([np.interp(GRID, t, r) for r in z])


def curve_table(tasks, per_task=60):
    """Pooled curve features and embeddings from a list of (y, X)."""
    feats, emb, noise = [], [], []
    for y, X in tasks:
        y0 = y[:, 0] if y.ndim == 3 else y
        z, _ = curves.normalise(y0)
        ok = np.isfinite(z).all(1)
        z = z[ok]
        if len(z) < 10:
            continue
        idx = RNG.choice(len(z), min(per_task, len(z)), replace=False)
        z = z[idx]
        zl = logres(z)
        emb.append(zl)
        c = curves.classify(z)
        s = curves.smooth(z)
        live = ((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1)
        for k in range(len(z)):
            f = {"final": float(curves.final_value(z[k:k + 1])[0])}
            if live[k]:
                p = (s[k, 0] - s[k]) / (s[k, 0] - s[k].min())
                pl = np.interp(GRID, np.arange(1, z.shape[1] + 1) / z.shape[1], p)
                f.update({"u10": U[np.argmax(pl >= 0.1)], "u50": U[np.argmax(pl >= 0.5)], "u90": U[np.argmax(pl >= 0.9)],
                          "late": float(1 - np.interp(0.8, GRID, pl)),
                          "rise_share": float((np.diff(s[k]) > 0).mean()),
                          "max_rise": float((s[k] - np.minimum.accumulate(s[k])).max())})
                m = int(np.argmin(s[k]))
                drop, rise = s[k, 0] - s[k, m], s[k, -1] - s[k, m]
                dd = np.diff(z[k])
                sdn = np.median(np.abs(dd - np.median(dd))) * 1.4826 / np.sqrt(2)
                f["over"] = float((m < 0.9 * z.shape[1]) and rise >= max(0.1 * drop, 3 * sdn))
                r = z[k] - s[k]
                sd = np.median(np.abs(r - np.median(r))) * 1.4826
                if sd > 0 and np.isfinite(r).all():
                    T = len(r)
                    noise.append({"kurtosis": float(kurtosis(r / sd)), "spikes": float((np.abs(r) > 5 * sd).mean()),
                                  "late_vs_early_sd": float(np.std(r[-T // 3:]) / max(np.std(r[:T // 3]), 1e-9))})
            feats.append(f)
    return pd.DataFrame(feats), np.concatenate(emb), pd.DataFrame(noise)


def landscape_table(tasks):
    rows = []
    for y, X in tasks:
        y0 = y[:, 0] if y.ndim == 3 else y
        z, _ = curves.normalise(y0)
        ok = np.isfinite(z).all(1)
        fin = curves.final_value(np.nan_to_num(z, nan=curves.CAP))
        if X.shape[1] == 0 or len(fin) < 100:
            continue
        r = {"rho200": evaluate.landscape_rho(X, fin), "rho50": evaluate.landscape_rho(X, fin, n_train=50),
             "near_best": float((fin <= fin.min() + 0.05).mean())}
        k = 10
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        idx = nn.kneighbors(X, return_distance=False)[:, 1:]
        f = fin + 1e-9 * RNG.standard_normal(len(fin))
        fs = RNG.permutation(f)
        lo, los = (f[:, None] < f[idx]).all(1).mean(), (fs[:, None] < fs[idx]).all(1).mean()
        r["optima_ratio"] = float(lo / max(los, 1e-9))
        c = np.full(len(z), 3)
        c[ok] = curves.classify(z[ok])
        s = curves.smooth(np.nan_to_num(z, nan=curves.CAP))
        live = ((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1)
        if live.sum() >= 80:
            p = (s[live, :1] - s[live]) / (s[live, :1] - s[live].min(1, keepdims=True))
            T = z.shape[1]
            pl = np.stack([np.interp(GRID, np.arange(1, T + 1) / T, q) for q in p])
            u50 = np.array([U[np.argmax(q >= 0.5)] for q in pl])
            if u50.std() > 0:
                m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                      verbose=-1, n_jobs=4)
                pr = cross_val_predict(m, X[live], u50, cv=5)
                r["shape_from_config"] = float(1 - ((u50 - pr) ** 2).mean() / u50.var())
        rows.append(r)
    return pd.DataFrame(rows)


def c2st(A, B):
    n = min(len(A), len(B), 6000)
    A, B = A[RNG.choice(len(A), n, replace=False)], B[RNG.choice(len(B), n, replace=False)]
    X = np.concatenate([A, B])
    y = np.r_[np.zeros(n), np.ones(n)]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8)
    p = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def prec_recall(A, B, k=5):
    """k-NN precision (generated B inside the real manifold of A) and recall (A inside B's manifold)."""
    n = min(len(A), len(B), 5000)
    A, B = A[RNG.choice(len(A), n, replace=False)], B[RNG.choice(len(B), n, replace=False)]
    pca = PCA(8).fit(A)
    a, b = pca.transform(A), pca.transform(B)

    def manifold(R, Q):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(R)
        rad = nn.kneighbors(R)[0][:, -1]
        d, i = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(Q)
        return float((d[:, 0] <= rad[i[:, 0]]).mean())

    return manifold(a, b), manifold(b, a)


def summarise(real, gen):
    fr, er, nr = real["curves"]
    fg, eg, ng = gen["curves"]
    out = {}
    for col in ["u10", "u50", "u90", "late", "rise_share", "max_rise", "final"]:
        a, b = fr[col].dropna().to_numpy(), fg[col].dropna().to_numpy() if col in fg else np.array([])
        out[f"B_gap_{col}"] = float(wasserstein_distance(a, b) / (a.std() + 1e-9)) if len(b) else np.nan
        out[f"B_median_{col}"] = float(np.median(b)) if len(b) else np.nan
    out["B_share_over"] = float(fg["over"].mean()) if "over" in fg else np.nan     # share of live curves that
    out["B_real_share_over"] = float(fr["over"].mean())                              # rise after their minimum
    out["B_c2st_auc"] = c2st(np.nan_to_num(er, nan=curves.CAP), np.nan_to_num(eg, nan=curves.CAP))
    out["B_precision"], out["B_recall"] = prec_recall(np.nan_to_num(er, nan=curves.CAP), np.nan_to_num(eg, nan=curves.CAP))
    for col in ["kurtosis", "spikes", "late_vs_early_sd"]:
        a = nr[col].to_numpy()
        b = ng[col].to_numpy() if col in ng else np.array([])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        out[f"C_median_{col}"] = float(np.nanmedian(b)) if len(b) else np.nan
        if col == "late_vs_early_sd":   # a ratio: compare logs, else single curves with near-zero early sd dominate
            a, b = np.log(np.maximum(a, 1e-3)), np.log(np.maximum(b, 1e-3))
        out[f"C_gap_{col}"] = float(wasserstein_distance(a, b) / (a.std() + 1e-9)) if len(b) else np.nan
    lr, lg = real["land"], gen["land"]
    for col in ["rho200", "rho50", "near_best", "optima_ratio", "shape_from_config"]:
        a, b = lr[col].dropna().to_numpy(), lg[col].dropna().to_numpy() if col in lg else np.array([])
        out[f"A_gap_{col}"] = float(wasserstein_distance(a, b) / (a.std() + 1e-9)) if len(b) else np.nan
        out[f"A_median_{col}"] = float(np.median(b)) if len(b) else np.nan
    return out


def main():
    warnings.filterwarnings("ignore")
    names = sys.argv[1:]
    real_tasks = [(y, X) for y, X, _ in EF.real_select()]
    real = {"curves": curve_table(real_tasks), "land": landscape_table(real_tasks)}
    rows = {"real (median)": {**{f"C_median_{c}": float(real["curves"][2][c].median()) for c in real["curves"][2]},
                              **{f"B_median_{c}": float(real["curves"][0][c].median()) for c in real["curves"][0]
                                 if c != "over"},
                              "B_share_over": float(real["curves"][0]["over"].mean()),
                              **{f"A_median_{c}": float(real["land"][c].median()) for c in real["land"]}}}
    for name in names:
        tasks = EF.FAKERS[name](len(real_tasks), np.random.default_rng(6))
        gen = {"curves": curve_table(tasks), "land": landscape_table(tasks)}
        rows[name] = summarise(real, gen)
        json.dump(rows[name], open(REPO / "results" / f"axes_{name}.json", "w"), indent=1)
        print(name, "done", flush=True)
    json.dump(rows["real (median)"], open(REPO / "results" / "axes_real.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()

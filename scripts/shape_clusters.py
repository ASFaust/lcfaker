"""Analyses S2 (are curve shapes multimodal?) and S3 (in which time axis are shapes simplest?).

Shape representation: live curves (improving / turning) of one seed with a gain (start - min) of at least 0.1 in z,
lightly smoothed (curvefaker.curves.smooth), turned into a *progress curve*
    p(t) = (s(0) - s(t)) / (s(0) - min s)      0 = still at the start, 1 = at the curve's best
which removes level and scale and keeps only the shape. Sampled on a common grid of the budget fraction from 0.02
to 1: G = 40 points evenly in log-time (S2) or in linear time (S3).

S2 views:
    (a) Gaussian mixtures on the first 6 PCA scores of the progress curves, k = 1..12, BIC; stability of the chosen
        clustering under bootstrap resampling (adjusted Rand index between refits on resamples, compared on all
        curves);
    (b) bimodality of interpretable features: u50 (log-time at which half the progress is made) and late share
        (progress made in the last 20% of the budget): bimodality coefficient (> 0.555 suggests bimodality) and
        BIC of 1 vs 2 Gaussian components;
    (c) a UMAP map, cluster-median curves and cluster shares per source (figures).
S3: the same mixture analysis in linear time; fewer, better-separated clusters = the simpler clock.
Analysis on the fit split; the select split is reported as replication.

    PYTHONPATH=. python scripts/shape_clusters.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from curvefaker import curves, data
from curvefaker.paths import REPO

RNG = np.random.default_rng(11)
G = 40
FIG = REPO / "docs" / "figures"
PLAN = {"taskset": [("fit", 400, 10), ("select", 150, 10)], "nasbench201": [("fit", 3, 800)],
        "quicktune": [("fit", 86, 20)], "jahs": [("fit", 81, 30)], "fcnet": [("select", 4, 600)],
        "bbomix": [("select", 35, 60)]}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a63d2", "#c0392b", "#7f8c8d",
          "#16a085", "#d35400", "#2c3e50"]


def progress_curves(z, clock):
    T = z.shape[1]
    s = curves.smooth(z)
    start, low = s[:, 0], s.min(1)
    p = (start[:, None] - s) / (start - low)[:, None]
    t = np.arange(1, T + 1) / T
    if clock == "log":
        grid = np.exp(np.linspace(np.log(0.02), 0, G))
    else:
        grid = np.linspace(0.02, 1, G)
    return np.stack([np.interp(grid, t, r) for r in p])


def collect():
    rows = []
    for src, parts in PLAN.items():
        for split, n_tasks, per in parts:
            ids = data.ids(split, [src])
            if len(ids) > n_tasks:
                ids = list(RNG.choice(ids, n_tasks, replace=False))
            for tid in ids:
                t = data.load(tid)
                y = t.y[:, 0]
                if src == "quicktune":
                    y = y[np.isfinite(y[:, -1])]
                z, _ = curves.normalise(y)
                ok = np.isfinite(z).all(1)
                zc = z[ok]
                c = curves.classify(zc)
                s = curves.smooth(zc)
                keep = ((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1)
                idx = np.where(keep)[0]
                idx = RNG.choice(idx, min(per, len(idx)), replace=False) if len(idx) else idx
                for i in idx:
                    rows.append((src, split, tid, zc[i]))
    return rows


def features(P_log):
    grid = np.linspace(np.log(0.02), 0, G)
    u50 = np.array([grid[np.argmax(r >= 0.5)] for r in P_log])
    lin = np.linspace(0.02, 1, G)
    late = np.array([1 - np.interp(0.8, np.exp(grid), r) for r in P_log])
    return {"u50": u50, "late_share": np.clip(late, -0.5, 1.5)}


def bimodality(x):
    x = x[np.isfinite(x)]
    n = len(x)
    bc = (skew(x) ** 2 + 1) / (kurtosis(x) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    b1 = GaussianMixture(1, random_state=0).fit(x[:, None]).bic(x[:, None])
    b2 = GaussianMixture(2, random_state=0, n_init=3).fit(x[:, None]).bic(x[:, None])
    return float(bc), float(b1 - b2)


def mixture_analysis(P, label):
    pca = PCA(6).fit(P)
    S = pca.transform(P)
    bics = {}
    for k in range(1, 13):
        gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(S)
        bics[k] = gm.bic(S)
    kbest = min(bics, key=bics.get)
    base = GaussianMixture(kbest, covariance_type="full", random_state=0, n_init=2).fit(S)
    lab = base.predict(S)
    aris = []
    for b in range(5):
        idx = RNG.integers(len(S), size=len(S))
        gm = GaussianMixture(kbest, covariance_type="full", random_state=b + 1, n_init=2).fit(S[idx])
        aris.append(adjusted_rand_score(lab, gm.predict(S)))
    # BIC curve relative to k = 1, per curve (to compare clocks with the same number of curves)
    rel = {k: (bics[1] - v) / len(S) for k, v in bics.items()}
    print(f"[{label}] PCA var explained (6): {pca.explained_variance_ratio_.sum():.3f}; best k by BIC: {kbest}; "
          f"BIC gain per curve vs k=1 at k=2,4,8,best: "
          f"{rel[2]:.2f}, {rel[4]:.2f}, {rel[8]:.2f}, {rel[kbest]:.2f}; bootstrap ARI {np.mean(aris):.2f} "
          f"(min {np.min(aris):.2f})", flush=True)
    return lab, kbest, rel, float(np.mean(aris)), S


def main():
    rows = collect()
    src = np.array([r[0] for r in rows])
    split = np.array([r[1] for r in rows])
    task = np.array([r[2] for r in rows])
    Z = [r[3] for r in rows]
    out = {}
    for clock in ("log", "lin"):
        P = np.stack([progress_curves(z[None], clock)[0] for z in Z])
        out[clock] = P
    print(f"{len(rows)} curves; by source:", pd.Series(src).value_counts().to_dict(), flush=True)
    results = []
    for clock in ("log", "lin"):
        for part in ("fit", "select"):
            m = split == part
            lab, k, rel, ari, S = mixture_analysis(out[clock][m], f"{clock}/{part}")
            results.append({"clock": clock, "split": part, "k_best": k, "ari": ari,
                            **{f"bic_gain_k{kk}": rel[kk] for kk in (2, 4, 8)}})
            if clock == "log" and part == "fit":
                lab_fit, S_fit, k_fit = lab, S, k
    F = features(out["log"])
    for name, x in F.items():
        for part in ("fit", "select"):
            for s in ["all"] + sorted(set(src[split == part])):
                m = (split == part) & ((src == s) if s != "all" else True)
                if m.sum() < 50:
                    continue
                bc, dbic = bimodality(x[m])
                results.append({"feature": name, "split": part, "source": s, "n": int(m.sum()),
                                "bimodality_coef": bc, "bic_gain_2_vs_1": dbic,
                                "median": float(np.median(x[m])), "q10": float(np.quantile(x[m], 0.1)),
                                "q90": float(np.quantile(x[m], 0.9))})
    pd.DataFrame(results).to_csv(REPO / "results" / "shape_clusters.csv", index=False)
    # cluster shares per source and cluster-median curves (log clock, fit split)
    m = split == "fit"
    tab = pd.crosstab(src[m], lab_fit, normalize="index").round(2)
    print("\ncluster shares per source (log clock, fit):\n", tab.to_string(), flush=True)
    # within-task: number of clusters per task and share of its curves in its main cluster
    tt = pd.DataFrame({"task": task[m], "lab": lab_fit, "src": src[m]})
    g = tt.groupby("task").lab.agg(lambda s: pd.Series(s).value_counts(normalize=True).iloc[0])
    nc = tt.groupby("task").lab.nunique()
    print("\nper task: share of curves in the task's main cluster, median by source:",
          g.groupby(tt.groupby("task").src.first()).median().round(2).to_dict(),
          "| clusters per task (median):", nc.groupby(tt.groupby("task").src.first()).median().to_dict(), flush=True)
    np.savez(REPO / "results" / "shape_clusters_curves.npz", P_log=out["log"], P_lin=out["lin"], src=src,
             split=split, task=task, lab_fit=lab_fit)
    plot(out["log"][m], lab_fit, src[m], k_fit)


def plot(P, lab, src, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.exp(np.linspace(np.log(0.02), 0, G))
    order = np.argsort(-np.bincount(lab, minlength=k))
    cols = min(k, 6)
    rows_ = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.3 * rows_), sharey=True, squeeze=False)
    for j, c in enumerate(order):
        ax = axes.flat[j]
        Pc = P[lab == c]
        sel = RNG.choice(len(Pc), min(40, len(Pc)), replace=False)
        for r in Pc[sel]:
            ax.plot(grid, r, color="#9aa5b1", lw=0.6, alpha=0.6)
        ax.plot(grid, np.median(Pc, 0), color=COLORS[0], lw=2)
        ax.set_xscale("log")
        ax.set_ylim(-0.4, 1.25)
        share = pd.Series(src[lab == c]).value_counts(normalize=True)
        ax.set_title(f"cluster {j}: {len(Pc) / len(P):.0%}\n" + ", ".join(f"{s} {v:.0%}" for s, v in share.head(2).items()),
                     fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[k:]:
        ax.axis("off")
    fig.supxlabel("budget fraction (log scale)", fontsize=9)
    fig.supylabel("progress (0 = start, 1 = best)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "shape_clusters.png", dpi=110)
    plt.close(fig)
    try:
        import umap
        emb = umap.UMAP(random_state=0).fit_transform(PCA(6).fit_transform(P))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for s_i, s in enumerate(sorted(set(src))):
            mm = src == s
            axes[0].scatter(emb[mm, 0], emb[mm, 1], s=2, color=COLORS[s_i], label=s, alpha=0.6)
        axes[0].legend(markerscale=5, fontsize=8, frameon=False)
        axes[0].set_title("by source", fontsize=9)
        for j, c in enumerate(order):
            mm = lab == c
            axes[1].scatter(emb[mm, 0], emb[mm, 1], s=2, color=COLORS[j % len(COLORS)], label=f"cluster {j}",
                            alpha=0.6)
        axes[1].legend(markerscale=5, fontsize=7, frameon=False, ncol=2)
        axes[1].set_title("by mixture cluster", fontsize=9)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIG / "shape_umap.png", dpi=110)
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()

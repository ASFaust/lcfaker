"""Analysis S2b / S3b: are the shape clusters one template shifted in time?

On the progress curves of scripts/shape_clusters.py (results/shape_clusters_curves.npz), fit per curve
    p(x) ~ alpha + beta * sigmoid(k (x - m))
with x = log-time (u, budget fraction on a log scale) or x = linear time. (m, k) by grid search, alpha / beta by
exact least squares. Reports per source and clock: share of curves the single template explains within RMSE 0.05 /
0.10 (progress units: 1 = whole improvement), the distribution of the location m (speed) and steepness k, and the
residual shapes of the curves it does not explain (their mixture clusters).

    PYTHONPATH=. python scripts/shape_template.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from curvefaker.paths import REPO

G = 40


def fit_template(P, x, m_grid, k_grid, chunk=2000):
    """P [n, G]; returns rmse, m, k, alpha, beta, fitted [n, G]."""
    n = len(P)
    best = np.full(n, np.inf)
    out = {k: np.zeros(n) for k in ("m", "k", "a", "b")}
    fitted = np.zeros_like(P)
    for m in m_grid:
        for k in k_grid:
            s = 1 / (1 + np.exp(-k * (x - m)))                        # [G]
            A = np.stack([np.ones(G), s], 1)                           # [G, 2]
            coef, *_ = np.linalg.lstsq(A, P.T, rcond=None)              # [2, n]
            f = (A @ coef).T
            sse = ((f - P) ** 2).mean(1)
            better = sse < best
            best[better] = sse[better]
            out["m"][better], out["k"][better] = m, k
            out["a"][better], out["b"][better] = coef[0, better], coef[1, better]
            fitted[better] = f[better]
    return np.sqrt(best), out, fitted


def main():
    Z = np.load(REPO / "results" / "shape_clusters_curves.npz", allow_pickle=True)
    src, split = Z["src"], Z["split"]
    u = np.linspace(np.log(0.02), 0, G)
    lin = np.linspace(0.02, 1, G)
    res = {}
    for clock, P, x, m_grid, k_grid in (
            ("log", Z["P_log"], u, np.linspace(-5, 1.5, 66), np.exp(np.linspace(np.log(0.3), np.log(40), 30))),
            ("lin", Z["P_lin"], lin, np.linspace(-0.3, 1.5, 73), np.exp(np.linspace(np.log(1), np.log(300), 30)))):
        rmse, par, fitted = fit_template(P, x, m_grid, k_grid)
        res[clock] = (rmse, par, fitted)
    rows = []
    for s in ["all"] + sorted(set(src)):
        msk = np.ones(len(src), bool) if s == "all" else src == s
        r = {"source": s, "n": int(msk.sum())}
        for clock in ("log", "lin"):
            rmse, par, _ = res[clock]
            r[f"{clock}_share_rmse<0.05"] = float((rmse[msk] < 0.05).mean())
            r[f"{clock}_share_rmse<0.10"] = float((rmse[msk] < 0.10).mean())
            r[f"{clock}_rmse_med"] = float(np.median(rmse[msk]))
        rmse, par, _ = res["log"]
        good = msk & (rmse < 0.10)
        for q in (0.1, 0.5, 0.9):
            r[f"m_q{int(q * 100)}"] = float(np.quantile(par["m"][good], q)) if good.any() else np.nan
            r[f"k_q{int(q * 100)}"] = float(np.quantile(par["k"][good], q)) if good.any() else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    df.to_csv(REPO / "results" / "shape_template.csv", index=False)
    np.savez(REPO / "results" / "shape_template_fits.npz", rmse_log=res["log"][0], m=res["log"][1]["m"],
             k=res["log"][1]["k"], a=res["log"][1]["a"], b=res["log"][1]["b"], rmse_lin=res["lin"][0])
    plot(Z, res, u)


def plot(Z, res, u):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    src = Z["src"]
    rmse, par, fitted = res["log"]
    P = Z["P_log"]
    srcs = ["taskset", "quicktune", "jahs", "nasbench201", "fcnet", "bbomix"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, s in zip(axes.flat, srcs):
        m = (src == s) & (rmse < 0.10)
        ax.hist(par["m"][m], bins=np.linspace(-5, 1.5, 40), color="#2a78d6")
        ax.set_title(f"{s}: template location m (log budget fraction)\n{m.sum()} of {(src == s).sum()} curves fit",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(REPO / "docs" / "figures" / "shape_template_location.png", dpi=105)
    plt.close(fig)
    # residual shapes of the curves the template does not explain
    bad = rmse >= 0.10
    R = P[bad]
    k = 6
    lab = GaussianMixture(k, random_state=0, n_init=2).fit_predict(PCA(6).fit_transform(R))
    order = np.argsort(-np.bincount(lab, minlength=k))
    fig, axes = plt.subplots(1, k, figsize=(2.6 * k, 2.4), sharey=True)
    grid = np.exp(u)
    for j, c in enumerate(order):
        ax = axes[j]
        Rc = R[lab == c]
        sel = np.random.default_rng(0).choice(len(Rc), min(30, len(Rc)), replace=False)
        for r in Rc[sel]:
            ax.plot(grid, r, color="#9aa5b1", lw=0.6, alpha=0.6)
        ax.plot(grid, np.median(Rc, 0), color="#eb6834", lw=2)
        ax.set_xscale("log")
        ax.set_ylim(-0.4, 1.25)
        sh = pd.Series(src[bad][lab == c]).value_counts(normalize=True)
        ax.set_title(f"{len(Rc) / len(src):.1%} of all\n" + ", ".join(f"{a} {v:.0%}" for a, v in sh.head(2).items()),
                     fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("curves NOT explained by one shifted S-shape in log-time (rmse >= 0.10), clustered", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPO / "docs" / "figures" / "shape_template_residual.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()

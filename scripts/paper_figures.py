"""Figures and tables for the paper (paper/). Fit / select data and generated tasks only; no held-out benchmark.

    PYTHONPATH=. python scripts/paper_figures.py [version]      (default: 11)

Writes paper/figures/*.pdf and paper/tables/*.tex.
"""
from __future__ import annotations

import json
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from curvefaker import curves, data, faker_hgp as F, faker_swt as S, sim  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

VERSION = int(sys.argv[1]) if len(sys.argv) > 1 else 11
OUT = REPO / "paper" / "figures"
TAB = REPO / "paper" / "tables"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d3a73", "#2a78d6", "#a9cdf2"])   # good (low) final = dark

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "font.family": "serif", "pdf.fonttype": 42,
})


def _z(y):
    y0 = y[:, 0] if y.ndim == 3 else y
    z, _ = curves.normalise(y0)
    return np.nan_to_num(z, nan=curves.CAP)


def _curves(ax, z, n=80, rng=None, title=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(z), min(n, len(z)), replace=False)
    z = z[idx]
    T = z.shape[1]
    t = np.arange(1, T + 1) / T
    fin = curves.final_value(z)
    r = np.argsort(np.argsort(-fin)) / max(len(fin) - 1, 1)   # 1 = best
    for k in np.argsort(-fin):                                  # best drawn last (on top)
        ax.plot(t, np.clip(z[k], -0.15, curves.CAP + 0.05), lw=0.6, alpha=0.75, color=SEQ(1 - r[k]))
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 2.1)
    ax.set_xlim(t[0], 1)
    if title:
        ax.set_title(title, loc="left")


def real_examples():
    picks = [("taskset", "TaskSet"), ("jahs", "JAHS-Bench-201"), ("quicktune", "QuickTune"),
             ("nasbench201", "NAS-Bench-201")]
    rng = np.random.default_rng(4)
    out = []
    for src, name in picks:
        ids = data.ids("fit", [src])
        for tid in rng.permutation(ids):
            t = data.load(tid)
            y = t.y[:, 0]
            y = y[np.isfinite(y).all(1)]
            if len(y) < 100:
                continue
            z = _z(y)
            live = curves.classify(z) != 1
            if live.mean() > 0.6:       # an example with visible structure
                out.append((z, name))
                break
    return out


def fig_examples():
    real = real_examples()
    # one generated task per source of the real row: the record (template, warp / noise / class distributions)
    # is drawn from that source's fit records; configurations, fields and curves are new
    R = dict(np.load(S.PATH_ALL))
    P5 = dict(np.load(F.PATH_V5))
    P5["table_rough"] = np.load(S.ETA_ROUGH)["table"]
    if VERSION >= 10:
        P5["rho_live"] = np.load(S.RHO_LIVE)["rho"]
    src = np.array([t.split("/")[0] for t in R["task"]])
    rng = np.random.default_rng(11)
    gen = []
    for s_ in ("taskset", "jahs", "quicktune", "nasbench201"):
        w = ((src == s_) & R["ok"].astype(bool)).astype(float)
        gen.append(S.sample_task(R, P5, w / w.sum(), rng, VERSION))
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.4), sharey=True)
    for ax, (z, title) in zip(axs[0], real):
        _curves(ax, z, title=title)
    for k, (ax, (y, X)) in enumerate(zip(axs[1], gen)):
        _curves(ax, _z(y), title=f"{real[k][1]} record")
    axs[0, 0].set_ylabel("real (fit split)\nnormalised loss z")
    axs[1, 0].set_ylabel("generated (swt)\nnormalised loss z")
    for ax in axs[1]:
        ax.set_xlabel("budget fraction t/T")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "examples.pdf")
    plt.close(fig)


def fig_gallery():
    """More generated tasks, for the appendix."""
    gen = S.faker(12, np.random.default_rng(23), version=VERSION)
    fig, axs = plt.subplots(3, 4, figsize=(7.0, 4.8), sharey=True)
    for k, (ax, (y, X)) in enumerate(zip(axs.flat, gen)):
        _curves(ax, _z(y), rng=np.random.default_rng(k), title=f"d={X.shape[1]}, T={y.shape[-1]}")
    for ax in axs[:, 0]:
        ax.set_ylabel("z")
    for ax in axs[-1]:
        ax.set_xlabel("t/T")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "gallery.pdf")
    plt.close(fig)


def fig_model():
    """The shape-invariant template and its four warps, on a fitted TaskSet record."""
    R = dict(np.load(S.PATH_ALL))
    src = np.array([t.split("/")[0] for t in R["task"]])
    cand = np.where((src == "taskset") & R["ok"].astype(bool))[0]
    drop = []
    for j in cand:   # the record whose template drops most over the covered range: a full S-shape
        xs = np.linspace(*R["x_range"][j], 60)
        fx = np.interp(xs, sim.KNOTS, R["f"][j])
        drop.append(fx[0] - fx[-1] - 0.5 * np.abs(np.diff(fx, 2)).sum())
    i = int(cand[int(np.argmax(drop))])
    f, xr = R["f"][i], R["x_range"][i]
    T = 200
    u = np.log(np.arange(1, T + 1) / T)
    t = np.exp(u)
    base = dict(alpha=0.05, amp=0.9, beta=0.0, gamma=0.0)

    def curve(**kw):
        p = {**base, **kw}
        return sim.evaluate(f, np.array([p["alpha"]]), np.array([p["amp"]]), np.array([p["beta"]]),
                            np.array([p["gamma"]]), u, xr)[0]

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    panels = [("shift β (log-time)", "beta", [-1.5, -0.5, 0.5, 1.5]),
              ("stretch γ", "gamma", [-0.6, -0.2, 0.2, 0.6]),
              ("amplitude a", "amp", [0.4, 0.6, 0.8, 1.0]),
              ("level α", "alpha", [0.0, 0.1, 0.2, 0.3])]
    for ax, (title, key, vals) in zip(axs, panels):
        for k, v in enumerate(vals):
            ax.plot(t, curve(**{key: v}), lw=1.2, color=SEQ(k / (len(vals) - 1) * 0.85))
        ax.plot(t, curve(), lw=1.0, color=INK, ls=(0, (2, 1.5)))
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("t/T")
    axs[0].set_ylabel("z")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "model_warps.pdf")
    plt.close(fig)


def fig_anatomy():
    """One generated task: final value along the LR-like axis, with classes; curves by class."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        (y, X), = S.faker(1, rng, version=VERSION)
        z = _z(y)
        c = curves.classify(z)
        if (c == 1).mean() > 0.1 and (c == 3).mean() > 0.05 and X.shape[1] >= 2:
            break
    fin = curves.final_value(z)
    cols = {0: BLUE, 2: BLUE, 1: INK2, 3: ORANGE}
    names = {0: "live", 1: "dead", 3: "diverged"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for cl in (1, 3, 0):
        m = (c == cl) | ((cl == 0) & (c == 2))
        axs[0].scatter(X[m, 0], np.minimum(fin[m], curves.CAP), s=8, lw=0.4, edgecolor="white", color=cols[cl],
                       label=f"{names[cl]} ({m.mean():.0%})")
    axs[0].set_xlabel("config dimension 1 (learning-rate-like axis)")
    axs[0].set_ylabel("final value")
    axs[0].legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    T = z.shape[1]
    t = np.arange(1, T + 1) / T
    idx = np.random.default_rng(0).choice(len(z), 90, replace=False)
    for k in idx:
        cl = 0 if c[k] == 2 else c[k]
        axs[1].plot(t, np.clip(z[k], -0.15, 2.05), lw=0.6, alpha=0.7, color=cols[cl])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("t/T")
    axs[1].set_ylabel("z")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "anatomy.pdf")
    plt.close(fig)


def fig_hpo():
    df = pd.read_csv(REPO / "results" / "hpo_regret.csv", index_col=0)
    budgets = [1, 2, 5, 10, 20]
    rows = [("real", "real (select)", INK, "-"), (f"swt-v{VERSION}", "swt (ours)", BLUE, "-"),
            ("ifbo", "ifBO prior", ORANGE, "-")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for ax, (alg, title) in zip(axs, [("random", "random search"), ("hyperband", "Hyperband"),
                                      ("freezethaw", "freeze-thaw heuristic")]):
        for key, lab, col, ls in rows:
            if key not in df.index:
                continue
            v = [df.loc[key, f"{alg}@{b}"] for b in budgets]
            ax.plot(budgets, v, color=col, lw=1.6 if key == "real" else 1.2, ls=ls, marker="o", ms=3.5, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_title(title, loc="left")
        ax.set_xlabel("budget (full trainings)")
    axs[0].set_ylabel("mean rank regret")
    axs[0].legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "hpo.pdf")
    plt.close(fig)


def tables():
    names = [("ifbo", "ifBO prior"), ("hgp-v5", "wavetable (hgp-v5)"), (f"swt-v{VERSION}", "swt (ours)"),
             ("real-fit", "real fit tasks")]
    ax = {n: json.load(open(REPO / "results" / f"axes_{n}.json")) for n, _ in names}
    real = {"A_median_rho200": 0.771, "A_median_near_best": 0.136, "A_median_optima_ratio": 0.461,
            "A_median_shape_from_config": 0.730, "C_median_kurtosis": 23.3, "C_median_late_vs_early_sd": 0.152}
    rows = [
        ("B", "classifier AUC real vs.\\ generated $\\downarrow$", "B_c2st_auc", None),
        ("B", "precision $\\uparrow$", "B_precision", None),
        ("B", "recall $\\uparrow$", "B_recall", None),
        ("B", "gap: time to 50\\% progress $\\downarrow$", "B_gap_u50", None),
        ("B", "gap: share of rising steps $\\downarrow$", "B_gap_rise_share", None),
        ("B", "gap: final value $\\downarrow$", "B_gap_final", None),
        ("C", "gap: residual kurtosis $\\downarrow$", "C_gap_kurtosis", None),
        ("C", "gap: spike rate $\\downarrow$", "C_gap_spikes", None),
        ("C", "gap: late/early noise ratio $\\downarrow$", "C_gap_late_vs_early_sd", None),
        ("A", "landscape predictability $\\rho_{200}$", "A_median_rho200", "A_median_rho200"),
        ("A", "near-best share", "A_median_near_best", "A_median_near_best"),
        ("A", "local-optima ratio", "A_median_optima_ratio", "A_median_optima_ratio"),
        ("A", "speed from config ($R^2$)", "A_median_shape_from_config", "A_median_shape_from_config"),
    ]
    L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         "axis & statistic & real & " + " & ".join(n for _, n in names) + " \\\\", "\\midrule"]
    last = None
    for axis, lab, key, rk in rows:
        if last and axis != last:
            L.append("\\midrule")
        last = axis
        rv = f"{real[rk]:.2f}" if rk else "--"
        vals = []
        for n, _ in names:
            v = ax[n].get(key, np.nan)
            vals.append("--" if v is None or not np.isfinite(v) else f"{v:.2f}")
        L.append(f"{axis} & {lab} & {rv} & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "axes.tex").write_text("\n".join(L) + "\n")

    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    e = pd.read_csv(REPO / "results" / "faker_eval.csv", index_col=0)
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         " & regret MAE $\\downarrow$ & Hyperband gain & freeze-thaw gain & coverage $\\uparrow$ & novelty \\\\",
         "\\midrule"]
    for key, name in [("real", "real (select)")] + names:
        if key not in h.index:
            continue
        r = h.loc[key]
        se = lambda c: f" $\\pm$ {r[c]:.3f}" if c in r and np.isfinite(r[c]) else ""
        cov = f"{e.loc[key, 'coverage']:.3f}" if key in e.index else "--"
        nov = f"{e.loc[key, 'novelty']:.2f}" if key in e.index and np.isfinite(e.loc[key, "novelty"]) else "--"
        L.append(f"{name} & {r['mae']:.3f} & {r['gain_hb']:.3f}{se('se_gain_hb')} & "
                 f"{r['gain_ft']:.3f}{se('se_gain_ft')} & {cov} & {nov} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "hpo.tex").write_text("\n".join(L) + "\n")


HISTORY = [
    ("swt-v1", "template + warps; vectors assigned in strata along one field"),
    ("swt-v2", "speed as its own correlated field (nearest real vector)"),
    ("swt-v3", "unpredictable landscape part spatially rough, not white"),
    ("swt-v4", "noise level from step differences, not fit residuals"),
    ("swt-v5", "warps from all live curves (no duplicated vectors)"),
    ("swt-v7", "warps constructed: exact final value and speed quantiles"),
    ("swt-v8", "rough part on the final-value field only"),
    ("swt-v9", "fallback for constructed curves below the real best"),
    ("swt-v10", "field calibrated to live configs' predictability"),
    ("swt-v11", "dead level in the generated task's own units"),
]


def history():
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    L = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "version & change & precision & recall & $\\rho_{200}$ & speed $R^2$ & optima & HB / FT gain \\\\",
         "\\midrule", "real & & & & 0.77 & 0.73 & 0.46 & 0.155 / 0.112 \\\\", "\\midrule"]
    for key, change in HISTORY:
        path = REPO / "results" / f"axes_{key}.json"
        if not path.exists():
            continue
        a = json.load(open(path))
        g = f"{h.loc[key, 'gain_hb']:.3f} / {h.loc[key, 'gain_ft']:.3f}" if key in h.index else "--"
        L.append(f"{key.replace('swt-', '')} & {change} & {a['B_precision']:.2f} & {a['B_recall']:.2f} & "
                 f"{a['A_median_rho200']:.2f} & {a['A_median_shape_from_config']:.2f} & "
                 f"{a['A_median_optima_ratio']:.2f} & {g} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "history.tex").write_text("\n".join(L) + "\n")


def numbers():
    """LaTeX macros for numbers quoted in the text, so prose and tables cannot drift apart."""
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    a = json.load(open(REPO / "results" / f"axes_swt-v{VERSION}.json"))
    b = json.load(open(REPO / "results" / "axes_ifbo.json"))
    k = f"swt-v{VERSION}"
    m = {"swtVersion": str(VERSION), "swtAUC": f"{a['B_c2st_auc']:.2f}", "ifboAUC": f"{b['B_c2st_auc']:.2f}",
         "swtPrec": f"{a['B_precision']:.2f}", "swtRec": f"{a['B_recall']:.2f}",
         "ifboPrec": f"{b['B_precision']:.2f}", "ifboRec": f"{b['B_recall']:.2f}",
         "swtRho": f"{a['A_median_rho200']:.2f}", "swtNear": f"{a['A_median_near_best']:.2f}",
         "swtOpt": f"{a['A_median_optima_ratio']:.2f}", "swtSpeed": f"{a['A_median_shape_from_config']:.2f}",
         "ifboSpeed": f"{b['A_median_shape_from_config']:.2f}", "ifboNear": f"{b['A_median_near_best']:.2f}",
         "swtMAE": f"{h.loc[k, 'mae']:.3f}", "ifboMAE": f"{h.loc['ifbo', 'mae']:.3f}",
         "swtHB": f"{h.loc[k, 'gain_hb']:.3f}", "swtFT": f"{h.loc[k, 'gain_ft']:.3f}",
         "ifboHB": f"{h.loc['ifbo', 'gain_hb']:.3f}", "ifboFT": f"{h.loc['ifbo', 'gain_ft']:.3f}",
         "realHB": f"{h.loc['real', 'gain_hb']:.3f}", "realFT": f"{h.loc['real', 'gain_ft']:.3f}"}
    (TAB / "numbers.tex").write_text("".join(f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in m.items()))


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for fn in (fig_model, fig_examples, fig_gallery, fig_anatomy, fig_hpo, tables, history, numbers):
        fn()
        print(fn.__name__, "done", flush=True)


if __name__ == "__main__":
    main()

"""Figures and tables for the bpl paper (paper/). Fit / select data and generated tasks only; no held-out benchmark.

    PYTHONPATH=. python scripts/paper_figures_bpl.py [version]     (default 17)

Writes paper/figures/bpl_*.pdf and paper/tables/bpl_*.tex. (The swt draft's figures: scripts/paper_figures.py.)
"""
from __future__ import annotations

import json
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from curvefaker import curves, data, faker_bpl as B, faker_hgp as F, faker_swt as S  # noqa: E402
from curvefaker.paths import REPO  # noqa: E402

import sys

VERSION = int(sys.argv[1]) if len(sys.argv) > 1 else 17
OUT = REPO / "paper" / "figures"
TAB = REPO / "paper" / "tables"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fd0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d3a73", "#2a78d6", "#a9cdf2"])   # good (low) final = dark
SOURCES = [("taskset", "TaskSet"), ("jahs", "JAHS-Bench-201"), ("quicktune", "QuickTune"),
           ("nasbench201", "NAS-Bench-201")]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "font.family": "serif", "pdf.fonttype": 42,
})


_BT = {}


def _sample(R, P5, w, rng, return_meta=False):
    """One generated task as (y [n, 1, T'], X[, meta]) from records weighted by w: bpl-v17 (the GPU prior,
    release/lcfaker/torch_prior.py; record's d, n configs and curve length, TaskSet start point prepended) or the
    numpy versions."""
    if VERSION < 17:
        return B.sample_task(R, P5, w, rng, VERSION, return_meta=return_meta)
    import torch
    sys.path.insert(0, str(REPO / "release"))
    from lcfaker.torch_prior import BplTasks
    if "P" not in _BT:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=dev)
        _BT["P"], _BT["g"], _BT["dev"] = BplTasks(1, 10, dev, generator=g), g, dev
    P, g, dev = _BT["P"], _BT["g"], _BT["dev"]
    g.manual_seed(int(rng.integers(2 ** 31)))
    r = int(rng.choice(len(w), p=w))
    d = int(np.clip(R["n_hps"][r], 1, 10))
    N, T = int(R["n"][r]), int(R["T"][r])
    P.resample(torch.zeros(1, dtype=torch.long), torch.tensor([d]), record=torch.tensor([r]))
    X = torch.rand(1, N, 10, device=dev, generator=g)
    X[..., d:] = 0
    y = P.curves(X, torch.tensor([T]), T)[0]
    if R["has_start"][r] > 0:
        y = torch.cat([P.start_values(X)[0][:, None], y], 1)
    out = (y.cpu().numpy()[:, None, :], X[0, :, :d].cpu().numpy())
    if return_meta:
        c = P.configs_info(X)
        wh = lambda m: np.where(m[0].cpu().numpy())[0]
        out = out + ({"record": r, "dead": wh(c["dead"]), "div": wh(c["diverged"]),
                      "over": wh(c["rising"] & ~c["dead"] & ~c["diverged"])},)
    return out


def _records():
    R = dict(np.load(B.PATH))
    P5 = dict(np.load(F.PATH_V5))
    P5["table_rough"] = np.load(S.ETA_ROUGH)["table"]
    rl = np.load(S.RHO_LIVE)["rho"]
    P5["rho"] = np.where(np.isfinite(rl), rl, P5["rho"])
    return R, P5


def _z(y):
    y0 = y[:, 0] if y.ndim == 3 else y
    z, _ = curves.normalise(y0)
    return np.nan_to_num(z, nan=curves.CAP)


def _rankcol(fin):
    r = np.argsort(np.argsort(-fin)) / max(len(fin) - 1, 1)   # 1 = best
    return r


def _curves(ax, z, n=80, rng=None, title=None, skip_start=False):
    rng = rng or np.random.default_rng(0)
    if skip_start:
        z = z[:, 1:]
    idx = rng.choice(len(z), min(n, len(z)), replace=False)
    z = z[idx]
    T = z.shape[1]
    t = np.arange(1, T + 1) / T
    fin = curves.final_value(z)
    r = _rankcol(fin)
    for k in np.argsort(-fin):
        ax.plot(t, np.clip(z[k], -0.15, curves.CAP + 0.05), lw=0.6, alpha=0.75, color=SEQ(1 - r[k]))
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 2.1)
    ax.set_xlim(t[0], 1)
    if title:
        ax.set_title(title, loc="left")


# ------------------------------------------------------------------------------------------------ figures
def fig_representation():
    """One real TaskSet task with decaying schedules, on three axes: raw loss / step; log-excess / log step;
    log-excess / log optimiser progress."""
    R, _ = _records()
    best = None
    for i in np.where((R["opt"] == "adam6p") & R["ok"].astype(bool))[0]:
        if R["kappa"][i] < 0.1:
            continue
        t = data.load(str(R["task"][i]))
        y = t.y[:, 0].astype(float)
        ok = np.isfinite(y).all(1)
        z = _z(y[ok])
        live = curves.classify(z) == 0
        dec = t.hps["linear_decay"].to_numpy()[ok] * 1e4 + t.hps["exponential_decay"].to_numpy()[ok] * 1e4
        score = live.sum() * (np.median(dec[live]) > 0.5)
        if best is None or score > best[0]:
            best = (score, i)
        if best[0] > 120:
            break
    i = best[1]
    tid = str(R["task"][i])
    t = data.load(tid)
    y = t.y[:, 0].astype(float)
    ok = np.isfinite(y).all(1)
    y = y[ok]
    z = _z(y)
    live = np.where(curves.classify(z) == 0)[0]
    rng = np.random.default_rng(1)
    live = rng.choice(live, min(60, len(live)), replace=False)
    y = y[live][:, 1:]                                       # drop step 0 (no log-time)
    T = y.shape[1]
    frac = np.arange(1, T + 1) / T
    steps = frac * 10000
    L = R["L"][i]
    oid, sch, eta, _ = B._schedule_of(t, tid)
    sch = [sch[k] for k in np.where(ok)[0][live]]
    eta = eta[ok][live]
    x = np.stack([B.progress(oid, s, frac) for s in sch]) * (eta / R["eta_ref"][i])[:, None] ** R["kappa"][i]
    fin = y[:, -5:].mean(1)
    col = SEQ(1 - _rankcol(-fin))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for k in np.argsort(-fin):
        axs[0].plot(steps, y[k], lw=0.6, color=col[k])
        axs[1].plot(steps, np.log10(y[k] - L), lw=0.6, color=col[k])
        axs[2].plot(x[k], np.log10(y[k] - L), lw=0.6, color=col[k])
    axs[0].set_xlabel("training step")
    axs[0].set_ylabel("validation loss")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("training step (log)")
    axs[1].set_ylabel("log$_{10}$(loss $-$ floor)")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("optimiser progress $x$ (log)")
    axs[0].set_title("(a) linear scales", loc="left")
    axs[1].set_title("(b) log-log", loc="left")
    axs[2].set_title("(c) log-log, progress time", loc="left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "bpl_representation.pdf")
    plt.close(fig)
    return tid


def fig_model():
    """The broken power law and how schedule and learning rate act on it."""
    L, d, A, tau, a = 0.0, 0.05, 1.0, 0.02, 1.2
    xg = np.logspace(-4, 0, 300)
    f = lambda x, d=d, A=A, tau=tau, a=a: L + d + A / (1 + (x / tau) ** a)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    ax = axs[0]
    ax.plot(xg, f(xg) - L, color=BLUE, lw=1.6)
    ax.plot(xg, d + A * (xg / tau) ** -a, color=INK2, lw=0.8, ls=(0, (2, 1.5)))
    ax.axhline(d, color=INK2, lw=0.8, ls=(0, (2, 1.5)))
    ax.axvline(tau, color=INK2, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 3)
    ax.text(1.5e-4, 1.25, "plateau $d+A$", fontsize=7, color=INK2)
    ax.text(tau * 1.2, 0.5, r"knee $\tau$", fontsize=7, color=INK2)
    ax.text(0.08, 0.32, "slope $-a$", fontsize=7, color=INK2)
    ax.text(1.5e-4, d * 1.15, "floor offset $d$", fontsize=7, color=INK2)
    ax.set_xlabel("progress $x$ (log)")
    ax.set_ylabel("loss $-$ task floor $L$ (log)")
    ax.set_title("(a) one curve, log-log", loc="left")
    frac = np.arange(1, 201) / 200
    scheds = [("constant", 0, {}), ("linear decay to 0", 1, {"linear_decay": 1e-4, "exponential_decay": 0.0}),
              ("warm-up + cosine", 2, {"warmup_fraction": 0.05, "constant_fraction": 0.3,
                                       "min_learning_rate_mult": 0.0})]
    for (lab, oid, s), c in zip(scheds, (INK, BLUE, ORANGE)):
        axs[1].plot(frac, f(B.progress(oid, s, frac)), color=c, lw=1.2, label=lab)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("budget fraction (log)")
    axs[1].set_ylabel("loss")
    axs[1].set_title("(b) three schedules", loc="left")
    axs[1].legend(frameon=False, loc="upper right")
    for m, c in zip((0.1, 1.0, 10.0), (SEQ(0.85), SEQ(0.45), SEQ(0.0))):
        axs[2].plot(frac, f(frac * m ** 0.15), color=c, lw=1.2, label=fr"lr $\times${m:g}")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("budget fraction (log)")
    axs[2].set_title(r"(c) learning rate, $\kappa$=0.15", loc="left")
    axs[2].legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "bpl_model.pdf")
    plt.close(fig)


def _real_examples():
    rng = np.random.default_rng(4)
    out = []
    for src, name in SOURCES:
        for tid in rng.permutation(data.ids("fit", [src])):
            t = data.load(tid)
            y = t.y[:, 0]
            y = y[np.isfinite(y).all(1)]
            if len(y) < 100:
                continue
            z = _z(y)
            if (curves.classify(z) != 1).mean() > 0.6:
                out.append((z, name, src))
                break
    return out


def fig_examples():
    R, P5 = _records()
    real = _real_examples()
    rng = np.random.default_rng(11)
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.4), sharey=True)
    for k, (ax, (z, title, src)) in enumerate(zip(axs[0], real)):
        _curves(ax, z, title=title, skip_start=src == "taskset")
        w = ((R["source"] == src) & R["ok"].astype(bool)).astype(float)
        y, X = _sample(R, P5, w / w.sum(), rng)
        _curves(axs[1, k], _z(y), title="generated from its records", skip_start=src == "taskset")
    axs[0, 0].set_ylabel("real (fit split)\nnormalised value")
    axs[1, 0].set_ylabel("generated (bpl)\nnormalised value")
    for ax in axs[1]:
        ax.set_xlabel("budget fraction (log)")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "bpl_examples.pdf")
    plt.close(fig)


def fig_anatomy():
    R, P5 = _records()
    rng = np.random.default_rng(3)
    w = ((R["source"] == "taskset") & R["ok"].astype(bool)).astype(float)
    for _ in range(200):
        y, X, meta = _sample(R, P5, w / w.sum(), rng, return_meta=True)
        n_ = len(X)
        if 0.08 * n_ < len(meta["dead"]) < 0.3 * n_ and 0.05 * n_ < len(meta["div"]) < 0.25 * n_ and \
                len(meta["over"]) >= 0.02 * n_:
            break
    i = meta["record"]
    lo, hi = R["lr_q_live"][i]
    leta = (lo + X[:, 0] * (hi - lo)) / math.log(10)
    z = _z(y)[:, 1:]
    fin = curves.final_value(z)
    n = len(X)
    cls = np.zeros(n, int)
    cls[meta["dead"]] = 1
    cls[meta["div"]] = 3
    cls[meta["over"]] = 2
    cols = {0: BLUE, 1: INK2, 2: AQUA, 3: ORANGE}
    names = {0: "live", 1: "dead", 2: "rising (overfit)", 3: "diverged"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for cl in (1, 3, 0, 2):
        m = cls == cl
        axs[0].scatter(leta[m], np.minimum(fin[m], curves.CAP), s=8, lw=0.4, edgecolor="white", color=cols[cl],
                       label=f"{names[cl]} ({m.mean():.0%})")
    axs[0].set_xlabel("config dimension 0: log$_{10}$ learning rate" +
                      (" (other dimensions vary too)" if X.shape[1] > 1 else " (the task's only dimension)"))
    axs[0].set_ylabel("final value (normalised)")
    axs[0].legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.2))
    T = z.shape[1]
    t = np.arange(1, T + 1) / T
    pick = np.concatenate([rng.choice(np.where(cls == c)[0], min(k, (cls == c).sum()), replace=False)
                           for c, k in ((0, 60), (1, 10), (3, 10), (2, 8))])
    for k in pick:
        axs[1].plot(t, np.clip(z[k], -0.15, 2.05), lw=0.8 if cls[k] == 2 else 0.6, alpha=0.8, color=cols[cls[k]])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("budget fraction (log)")
    axs[1].set_ylabel("normalised value")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "bpl_anatomy.pdf")
    plt.close(fig)


def _seed_mean(df, key):
    rows = [k for k in (key, f"{key}@7", f"{key}@8") if k in df.index]
    return df.loc[rows].mean(numeric_only=True), len(rows)


def fig_hpo(skip=(), out="bpl_hpo.pdf"):
    df = pd.read_csv(REPO / "results" / "hpo_regret.csv", index_col=0)
    budgets = [1, 2, 5, 10, 20]
    rows = [("real", "real select tasks (reference)", INK), ("ifbo", "ifBO prior", ORANGE),
            ("swt-v11", "swt (first model)", AQUA), (f"bpl-v{VERSION}", "bpl", BLUE)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for ax, (alg, title) in zip(axs, [("random", "random search"), ("hyperband", "Hyperband"),
                                      ("freezethaw", "freeze-thaw heuristic")]):
        for key, lab, col in rows:
            if key in skip:
                continue
            r, _ = _seed_mean(df, key)
            v = [r[f"{alg}@{b}"] for b in budgets]
            ax.plot(budgets, v, color=col, lw=1.8 if key == "real" else 1.2, marker="o", ms=3.5, label=lab,
                    ls="-" if key in ("real", f"bpl-v{VERSION}") else (0, (3, 1.5)))
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_title(title, loc="left")
        ax.set_xlabel("budget (full trainings)")
    axs[0].set_ylabel("mean rank regret")
    axs[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / out)
    plt.close(fig)


def fig_gallery():
    R, P5 = _records()
    src = R["source"]
    from curvefaker import data as _d
    counts = {s_: len(_d.ids("fit", [s_])) for s_ in np.unique(src)}
    w = np.array([counts[s_] / (src == s_).sum() for s_ in src]) * R["ok"]
    rng = np.random.default_rng(23)
    gen = [_sample(R, P5, w / w.sum(), rng) for _ in range(12)]
    fig, axs = plt.subplots(3, 4, figsize=(7.0, 4.8), sharey=True)
    for k, (ax, (y, X)) in enumerate(zip(axs.flat, gen)):
        _curves(ax, _z(y), rng=np.random.default_rng(k), title=f"d={X.shape[1]}, T={y.shape[-1]}")
    for ax in axs[:, 0]:
        ax.set_ylabel("normalised value")
    for ax in axs[-1]:
        ax.set_xlabel("budget fraction (log)")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "bpl_gallery.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- tables
def _bold_best(vals, ref, better):
    v = np.array(vals, float)
    if better == "closer":
        s = np.abs(v - ref)
    elif better == "lower":
        s = v.copy()
    else:
        s = -v
    s[~np.isfinite(s)] = np.inf
    return int(np.argmin(s))


def _fmt(x, nd=3):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.{nd}f}"


def table_representation():
    r = pd.read_csv(REPO / "results" / "representation_study.csv")
    f = pd.read_csv(REPO / "results" / "fpca_study.csv")
    g = r.groupby("source")
    gf = f.groupby("source")
    names = {"taskset": "TaskSet", "jahs": "JAHS", "quicktune": "QuickTune", "nasbench201": "NB201"}
    order = ["taskset", "quicktune", "jahs", "nasbench201"]
    L = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         " & tasks & \\multicolumn{2}{c}{variance explained by 1 component $\\uparrow$} & "
         "\\multicolumn{3}{c}{rank corr.\\ of predicted finals $\\uparrow$} & error of finals $\\downarrow$ & noise "
         "slope \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}",
         "source & & linear & log-excess & linear & log-excess & current value & linear / log-excess & $k$ \\\\",
         "\\midrule"]
    for s in order:
        m, mf = g.get_group(s).median(numeric_only=True), gf.get_group(s).median(numeric_only=True)
        e1 = [mf["lin_ev1"], mf["logex_ev1"]]
        rho = [m["lin_rho_0.4"], m["logex_0.02_rho_0.4"]]
        b1 = "linear" if e1[0] > e1[1] else "log"
        cells_e = [f"\\textbf{{{x:.3f}}}" if (j == int(np.argmax(e1))) else f"{x:.3f}" for j, x in enumerate(e1)]
        cells_r = [f"\\textbf{{{x:.2f}}}" if (j == int(np.argmax(rho))) else f"{x:.2f}" for j, x in enumerate(rho)]
        L.append(f"{names[s]} & {len(g.get_group(s))} & {cells_e[0]} & {cells_e[1]} & {cells_r[0]} & {cells_r[1]} & "
                 f"{m['base_rho_0.4']:.2f} & {m['lin_mae_0.4']:.3f} / {m['logex_0.02_mae_0.4']:.3f} & "
                 f"{m['noise_slope']:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "bpl_representation.tex").write_text("\n".join(L) + "\n")


def table_schedule():
    df = pd.read_csv(REPO / "results" / "schedule_study.csv")
    df["group"] = np.where(df.end_mult < 0.5, "decayed", np.where(df.end_mult >= 0.9, "constant", "mild"))
    L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         " & & & \\multicolumn{2}{c}{rank corr.\\ of predicted finals $\\uparrow$} & "
         "\\multicolumn{2}{c}{error of finals $\\downarrow$} \\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "learning rate at the end & family & curves & step time & progress time & step time & progress time \\\\",
         "\\midrule"]
    for grp, glab in (("decayed", "$<$ 0.5 $\\times$ initial"), ("constant", "$\\ge$ 0.9 $\\times$ initial")):
        for opt in ("adam6p", "adam8p", "nadamw"):
            d = df[(df.group == grp) & (df.opt == opt)]
            rho = {}
            for kind in ("t", "s1"):
                rr = [curves.spearman(dt[f"{kind}_pred"].fillna(dt.cur).to_numpy(), dt.fin.to_numpy())
                      for _, dt in d.groupby("task") if len(dt) >= 8]
                rho[kind] = np.nanmedian(rr) if len(rr) else np.nan
            if not np.isfinite(rho["t"]):
                continue
            mae = {k: ((d[f"{k}_pred"] - d.fin).abs() / d.range).median() for k in ("t", "s1")}
            br = "s1" if rho["s1"] > rho["t"] + 1e-9 else ("t" if rho["t"] > rho["s1"] + 1e-9 else None)
            bm = "s1" if mae["s1"] < mae["t"] - 1e-9 else ("t" if mae["t"] < mae["s1"] - 1e-9 else None)
            def c(v, k, b):
                v = f"{v:.3f}"
                return f"\\textbf{{{v}}}" if b == k else v
            L.append(f"{glab} & {opt} & {len(d)} & {c(rho['t'], 't', br)} & {c(rho['s1'], 's1', br)} & "
                     f"{c(mae['t'], 't', bm)} & {c(mae['s1'], 's1', bm)} \\\\")
            glab = ""
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "bpl_schedule.tex").write_text("\n".join(L) + "\n")


ROWS = [
    ("Curves", "time to 50\\% of progress (log budget fraction)", "B_median_u50", "closer"),
    ("Curves", "time to 90\\% of progress (log budget fraction)", "B_median_u90", "closer"),
    ("Curves", "progress in the last 20\\% of the budget", "B_median_late", "closer"),
    ("Curves", "share of curves rising after their minimum", "B_share_over", "closer"),
    ("Curves", "final value (normalised)", "B_median_final", "closer"),
    ("Curves", "classifier telling real from generated (AUC, 0.5 = cannot)", "B_c2st_auc", "lower"),
    ("Curves", "generated curves that look real (precision)", "B_precision", "higher"),
    ("Curves", "real curves the generator can produce (recall)", "B_recall", "higher"),
    ("Noise", "excess kurtosis of residuals", "C_median_kurtosis", "closer"),
    ("Noise", "residual spread, last third / first third", "C_median_late_vs_early_sd", "closer"),
    ("Landscape", "final value predictable from config (rank corr.)", "A_median_rho200", "closer"),
    ("Landscape", "share of configs within 0.05 of the best", "A_median_near_best", "closer"),
    ("Landscape", "local optima relative to random (1 = random)", "A_median_optima_ratio", "closer"),
    ("Landscape", "curve speed predictable from config ($R^2$)", "A_median_shape_from_config", "closer"),
    ("HPO", "Hyperband gain over random search", "gain_hb", "closer"),
    ("HPO", "freeze-thaw gain over random search", "gain_ft", "closer"),
    ("HPO", "regret difference from real (mean abs.)", "mae", "lower"),
]
COMPARE = [("ifbo", "ifBO prior"), ("swt-v11", "swt"), ("bpl-v1", "bpl-v1"), ("bpl-v13", "bpl-v13"),
           ("bpl-v16", "bpl-v16"), ("bpl-v17", "bpl-v17")]
BETTER = {"closer": "closer to real", "lower": "lower", "higher": "higher"}


def _hpo_means():
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    out = {}
    for key in ["real", "real-fit"] + [k for k, _ in COMPARE] + [f"bpl-v{VERSION}"]:
        rows = [k for k in (key, f"{key}@7", f"{key}@8") if k in h.index]
        m = h.loc[rows, ["mae", "gain_hb", "gain_ft"]].mean()
        sd = h.loc[rows, ["mae", "gain_hb", "gain_ft"]].std() if len(rows) > 1 else m * np.nan
        out[key] = (m, sd, len(rows))
    return out


def table_compare(compare=None, out="bpl_compare.tex"):
    compare = compare or COMPARE
    real = json.load(open(REPO / "results" / "axes_real.json"))
    hm = _hpo_means()
    real.update({k: float(hm["real"][0][k]) for k in ("mae", "gain_hb", "gain_ft")})
    vals = {}
    for key, _ in compare:
        a = json.load(open(REPO / "results" / f"axes_{key}.json"))
        a.update({k: float(hm[key][0][k]) for k in ("mae", "gain_hb", "gain_ft")})
        vals[key] = a
    L = ["\\begin{tabular}{llr" + "r" * len(compare) + "l}", "\\toprule",
         " & statistic & \\textbf{real (reference)} & " + " & ".join(n for _, n in compare) + " & better \\\\",
         "\\midrule"]
    last = None
    for sec, lab, key, better in ROWS:
        if last and sec != last:
            L.append("\\midrule")
        last = sec
        r = real.get(key, np.nan)
        v = [vals[k].get(key, np.nan) for k, _ in compare]
        v = [np.nan if x is None else x for x in v]
        b = _bold_best(v, r, better)
        cells = [f"\\textbf{{{_fmt(x)}}}" if j == b else _fmt(x) for j, x in enumerate(v)]
        rr = "--" if key in ("B_c2st_auc", "B_precision", "B_recall") else _fmt(r)
        L.append(f"{sec} & {lab} & {rr} & " + " & ".join(cells) + f" & {BETTER[better]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / out).write_text("\n".join(L) + "\n")


def table_hpo(compare=None, out="bpl_hpo.tex", fit_row=True):
    compare = compare or COMPARE
    hm = _hpo_means()
    keys = [("real", "\\textbf{real select tasks (reference)}"), ] + ([("real-fit", "real fit tasks")] if fit_row else []) + compare
    L = ["\\begin{tabular}{lrrrr}", "\\toprule",
         " & seeds & regret difference from real $\\downarrow$ & Hyperband gain & freeze-thaw gain \\\\",
         "\\midrule"]
    gen = [k for k, _ in compare]
    best = {c: _bold_best([hm[k][0][c] for k in gen], hm["real"][0][c], "lower" if c == "mae" else "closer")
            for c in ("mae", "gain_hb", "gain_ft")}
    for key, name in keys:
        m, sd, n = hm[key]
        cells = []
        for c in ("mae", "gain_hb", "gain_ft"):
            s = f"{m[c]:.3f}"
            if key in gen and gen.index(key) == best[c]:
                s = f"\\textbf{{{s}}}"
            if n > 1 and np.isfinite(sd[c]):
                s += f" {{\\scriptsize$\\pm${sd[c]:.3f}}}"
            cells.append(s)
        seeds = "--" if key.startswith("real") else str(n)
        L.append(f"{name} & {seeds} & " + " & ".join(cells) + " \\\\")
        if key == ("real-fit" if fit_row else "real"):
            L.append("\\midrule")
    L.append("\\midrule")
    L.append(f"better & & lower & closer to {hm['real'][0]['gain_hb']:.3f} & closer to {hm['real'][0]['gain_ft']:.3f}"
             " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / out).write_text("\n".join(L) + "\n")


DEV = [  # (key, change, kept?)
    ("bpl-v1", "broken power law on log-excess scale, progress time, copula over parameters, overfitting, noise", "base"),
    ("bpl-v2", "real joint parameter vectors (nearest) instead of the copula", "no"),
    ("bpl-v4", "v2 + vectors matched on the config's progress budget", "no"),
    ("bpl-v5", "v4 + second slow stage (floor offset decays as $x^{-b}$)", "no"),
    ("bpl-v6", "v5 + noise level from second differences", "no"),
    ("bpl-v10", "v5 + the task's real learning-rate relation in targets and matching", "no"),
    ("bpl-v13", "v1 + the task's real learning-rate relation (conditional means of the copula scores)", "yes"),
    ("bpl-v15", "v13 + floor offset capped: TaskSet curves do not start above the shared initial loss", "yes"),
    ("bpl-v16", "v15 + amplitude capped as well", "yes"),
    ("bpl-v17", "v16 as a batched GPU prior: configuration maps as functions (random Fourier features), class "
                "shares as thresholds, closed-form progress (any curve length)", "yes"),
]


def table_dev():
    hm = _hpo_means()
    real = json.load(open(REPO / "results" / "axes_real.json"))
    L = ["\\begin{tabular}{p{6.2cm}lrrrrrr}", "\\toprule",
         "change (one per version) & kept & recall & time to 90\\% & speed $R^2$ & $\\rho_{200}$ & HB gain & FT gain"
         " \\\\", "\\midrule",
         f"\\textbf{{real select tasks (reference)}} & & -- & {real['B_median_u90']:.2f} & "
         f"{real['A_median_shape_from_config']:.2f} & {real['A_median_rho200']:.2f} & "
         f"{hm['real'][0]['gain_hb']:.3f} & {hm['real'][0]['gain_ft']:.3f} \\\\", "\\midrule"]
    h = pd.read_csv(REPO / "results" / "hpo_summary.csv", index_col=0)
    for key, change, kept in DEV:
        p = REPO / "results" / f"axes_{key}.json"
        if not p.exists():
            continue
        a = json.load(open(p))
        rows = [k for k in (key, f"{key}@7", f"{key}@8") if k in h.index]
        g = h.loc[rows, ["gain_hb", "gain_ft"]].mean()
        L.append(f"{key.replace('bpl-', '')}: {change} & {kept} & {a['B_recall']:.2f} & {a['B_median_u90']:.2f} & "
                 f"{a['A_median_shape_from_config']:.2f} & {a['A_median_rho200']:.2f} & {g['gain_hb']:.3f} & "
                 f"{g['gain_ft']:.3f}{'$^*$' if len(rows) > 1 else ''} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "bpl_dev.tex").write_text("\n".join(L) + "\n")


def tables_results_only():
    """Tables for paper/results.tex: real data, the ifBO prior and bpl only."""
    short = [("ifbo", "ifBO prior"), (f"bpl-v{VERSION}", "bpl")]
    table_compare(short, "res_compare.tex")
    table_hpo(short, "res_hpo.tex", fit_row=True)
    fig_hpo(skip=("swt-v11",), out="res_hpo.pdf")


def numbers():
    hm = _hpo_means()
    a = json.load(open(REPO / "results" / f"axes_bpl-v{VERSION}.json"))
    s = json.load(open(REPO / "results" / "axes_swt-v11.json"))
    b = json.load(open(REPO / "results" / "axes_ifbo.json"))
    real = json.load(open(REPO / "results" / "axes_real.json"))
    k = f"bpl-v{VERSION}"
    m = {
        "bplMAE": hm[k][0]["mae"], "swtMAE": hm["swt-v11"][0]["mae"], "ifboMAE": hm["ifbo"][0]["mae"],
        "bplHB": hm[k][0]["gain_hb"], "bplFT": hm[k][0]["gain_ft"], "swtHB": hm["swt-v11"][0]["gain_hb"],
        "swtFT": hm["swt-v11"][0]["gain_ft"], "ifboHB": hm["ifbo"][0]["gain_hb"], "ifboFT": hm["ifbo"][0]["gain_ft"],
        "realHB": hm["real"][0]["gain_hb"], "realFT": hm["real"][0]["gain_ft"],
        "fitHB": hm["real-fit"][0]["gain_hb"], "fitFT": hm["real-fit"][0]["gain_ft"],
        "fitMAE": hm["real-fit"][0]["mae"],
        "bplAUC": a["B_c2st_auc"], "swtAUC": s["B_c2st_auc"], "ifboAUC": b["B_c2st_auc"],
        "bplPrec": a["B_precision"], "bplRec": a["B_recall"], "swtPrec": s["B_precision"], "swtRec": s["B_recall"],
        "ifboPrec": b["B_precision"], "ifboRec": b["B_recall"],
        "bplRho": a["A_median_rho200"], "realRho": real["A_median_rho200"],
        "bplSpeed": a["A_median_shape_from_config"], "realSpeed": real["A_median_shape_from_config"],
        "ifboSpeed": b["A_median_shape_from_config"],
        "bplNinety": a["B_median_u90"], "realNinety": real["B_median_u90"],
        "bplOver": a["B_share_over"], "realOver": real["B_share_over"],
        "bplNoise": a["C_median_late_vs_early_sd"], "realNoise": real["C_median_late_vs_early_sd"],
        "bplFinal": a["B_median_final"], "realFinal": real["B_median_final"],
    }
    txt = "".join(f"\\newcommand{{\\{n}}}{{{v:.3f}}}\n" if n.endswith(("MAE", "HB", "FT"))
                  else f"\\newcommand{{\\{n}}}{{{v:.2f}}}\n" for n, v in m.items())
    txt += f"\\newcommand{{\\bplVersion}}{{{VERSION}}}\n"
    (TAB / "bpl_numbers.tex").write_text(txt)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    tid = fig_representation()
    print("representation task:", tid, flush=True)
    for fn in (fig_model, fig_examples, fig_anatomy, fig_hpo, fig_gallery, table_representation, table_schedule,
               table_compare, table_hpo, table_dev, numbers, tables_results_only):
        fn()
        print(fn.__name__, "done", flush=True)


if __name__ == "__main__":
    main()

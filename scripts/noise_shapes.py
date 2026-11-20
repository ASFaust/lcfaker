"""Analyses N1-N3: the shape of the noise, spikes, and seed effects.

N1 / N2: residuals r = y - fit of live curves (one seed, normalised z), fit = the two-stage S-shape in log-time
(logsig2, the best family in S1). Per source:
    kurtosis (excess; Gaussian 0), skew, spike rate (|r| > 5 robust sd; Gaussian ~6e-7), share of spikes that go up
    (worse), lag-1 autocorrelation of r, robust sd in the early / middle / late third of the budget (relative to the
    whole curve), rank correlation of a curve's noise sd with its level (median z) and, for TaskSet, with its
    learning rate; spike co-occurrence: for steps with a spike in one config, the share of other configs of the same
    task that also spike at that step, relative to the base rate.
N3 (sources with several seeds): for pairs of seeds of the same config, the difference d(t) = y_s1 - y_s2 is split
into a constant (level), a linear trend in log-time (speed / shape) and the rest (white); shares of the sum of
squares. Seed share of final-value variance: variance across seeds / total variance of final values, overall and
among the 10% best configs.

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/noise_shapes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from curvefaker import curves, data, families
from curvefaker.paths import REPO

RNG = np.random.default_rng(15)
PLAN = {"taskset": ("fit", 150, 10), "nasbench201": ("fit", 3, 500), "quicktune": ("fit", 86, 15),
        "jahs": ("fit", 81, 18), "fcnet": ("select", 4, 400), "bbomix": ("select", 35, 40)}


def robust_sd(x, axis=-1):
    return np.median(np.abs(x - np.median(x, axis, keepdims=True)), axis) * 1.4826


def main():
    rows, seed_rows = [], []
    for src, (split, n_tasks, per) in PLAN.items():
        ids = data.ids(split, [src])
        if len(ids) > n_tasks:
            ids = list(RNG.choice(ids, n_tasks, replace=False))
        Y, tidx, lrs, cooc = [], [], [], []
        seed_parts, seed_share, seed_share_top = [], [], []
        for k, tid in enumerate(ids):
            t = data.load(tid)
            y = t.y
            if src == "quicktune":
                y = y[np.isfinite(y[:, 0, -1])]
            z, _ = curves.normalise(y)
            z0 = z[:, 0]
            ok = np.isfinite(z0).all(1)
            c = np.full(len(z0), 3)
            c[ok] = curves.classify(z0[ok])
            s = curves.smooth(np.nan_to_num(z0, nan=curves.CAP))
            live = np.where(((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1))[0]
            if len(live) < 5:
                continue
            pick = RNG.choice(live, min(per, len(live)), replace=False)
            Y += list(z0[pick])
            tidx += [k] * len(pick)
            if "learning_rate" in t.hps:
                lrs += list(np.log(t.hps["learning_rate"].to_numpy(float)[pick]))
            # N3: seeds
            R = z.shape[1]
            if R >= 2:
                both = live[np.isfinite(z[live, 1]).all(1)]
                if len(both) >= 5:
                    d = z[both, 0] - z[both, 1]
                    T = d.shape[1]
                    uu = np.log(np.arange(1, T + 1) / T)
                    A = np.stack([np.ones(T), uu - uu.mean()], 1)
                    coef, *_ = np.linalg.lstsq(A, d.T, rcond=None)
                    nz = (d ** 2).sum(1) > 1e-12          # deterministic tasks (identical seeds) carry no seed info
                    d, coef = d[nz], coef[:, nz]
                    tot = (d ** 2).sum()
                    if tot <= 1e-12:
                        continue
                    lvl = (np.outer(coef[0], A[:, 0]) ** 2).sum()
                    trd = (np.outer(coef[1], A[:, 1]) ** 2).sum()
                    seed_parts.append((lvl / tot, trd / tot, 1 - (lvl + trd) / tot))
                    fin = curves.final_value(np.nan_to_num(z[both].reshape(-1, T), nan=curves.CAP)).reshape(len(both), -1)
                    fin = fin[:, :R]
                    fin = fin[np.isfinite(fin).all(1)]
                    if len(fin) >= 10:
                        within = fin.var(1, ddof=1).mean()
                        seed_share.append(within / fin.mean(1).var(ddof=1) if fin.mean(1).var() > 0 else np.nan)
                        top = fin.mean(1) <= np.quantile(fin.mean(1), 0.1)
                        if top.sum() >= 3:
                            seed_share_top.append(fin[top].var(1, ddof=1).mean() / max(fin[top].mean(1).var(ddof=1), 1e-12))
        Y = np.array(Y, np.float32)
        tidx = np.array(tidx)
        T = Y.shape[1]
        fit, _ = families.fit("logsig2", Y, np.ones(T, bool))
        r = Y - fit
        sd = np.maximum(robust_sd(r), 1e-6)
        rz = r / sd[:, None]
        spike = np.abs(rz) > 5
        thirds = np.array_split(np.arange(T), 3)
        sd3 = [np.median(robust_sd(r[:, th]) / sd) for th in thirds]
        ac1 = np.median([np.corrcoef(x[:-1], x[1:])[0, 1] for x in r if x.std() > 0])
        # spike co-occurrence within tasks at the same step
        co = []
        for k in np.unique(tidx):
            S = spike[tidx == k]
            if len(S) < 5 or S.sum() == 0:
                continue
            base = S.mean()
            for tt in np.where(S.any(0))[0]:
                n_sp = S[:, tt].sum()
                co.append(((n_sp - 1) / (len(S) - 1)) / max(base, 1e-9))
        row = {"source": src, "curves": len(Y), "T": T,
               "excess_kurtosis_med": float(np.median(kurtosis(rz, axis=1))),
               "skew_med": float(np.median(skew(rz, axis=1))),
               "spike_rate": float(spike.mean()),
               "spike_up_share": float((rz[spike] > 0).mean()) if spike.any() else np.nan,
               "lag1_autocorr_med": float(ac1),
               "sd_early_rel": float(sd3[0]), "sd_mid_rel": float(sd3[1]), "sd_late_rel": float(sd3[2]),
               "rho_sd_vs_level": curves.spearman(sd, np.median(Y, 1)),
               "rho_sd_vs_lr": curves.spearman(sd, np.array(lrs)) if len(lrs) == len(Y) else np.nan,
               "spike_cooccurrence_vs_base": float(np.median(co)) if co else np.nan}
        if seed_parts:
            P = np.array(seed_parts)
            row.update({"seed_diff_level_share": float(np.nanmedian(P[:, 0])),
                        "seed_diff_trend_share": float(np.nanmedian(P[:, 1])),
                        "seed_diff_white_share": float(np.nanmedian(P[:, 2])),
                        "seed_share_final_var": float(np.nanmedian(seed_share)) if seed_share else np.nan,
                        "seed_share_final_var_top10": float(np.nanmedian(seed_share_top)) if seed_share_top else np.nan})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(REPO / "results" / "noise_shapes.csv", index=False)


if __name__ == "__main__":
    main()

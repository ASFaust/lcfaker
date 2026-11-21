"""Build release/lcfaker/assets/bpl_v16.npz: everything the standalone generator (lcfaker, bpl-v16) needs, from the
fit-split records (results/bpl_records.npz). No benchmark data is needed afterwards.

Precomputed per record, exactly as curvefaker.faker_bpl.sample_task (version 16) computes them on the fly:
  weight      sampling weight (by source, as faker_bpl.faker)
  eta         weight of the rough landscape part (matches the record's landscape predictability)
  lr_rel      1 if the record has a learning-rate relation (spread of log learning rates > 0.3)
  lr_mids     [10] bin centres of log learning rate; lr_mb [4, 10] conditional mean of each parameter's normal
              score per bin; lr_r2 [4] share of each score's variance explained by the learning rate

    PYTHONPATH=. python scripts/export_release_bpl.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from curvefaker import data, faker_bpl as B, faker_hgp as F, faker_swt as S
from curvefaker.paths import REPO

KEYS = ["source", "kind", "T", "n", "n_hps", "opt_id", "share_dead", "share_div", "L", "y0", "has_start", "start_sd",
        "kappa", "eta_ref", "share_over", "noise_sd", "noise_k", "noise_med_ex", "df", "dead_level", "lr_q_live",
        "copula", "noise_thirds", "sched_q", "params", "params_offs", "over_onset", "over_onset_offs", "over_rise",
        "over_rise_offs", "div_times", "div_times_offs"]


def main():
    R = dict(np.load(B.PATH))
    P5 = dict(np.load(F.PATH_V5))
    table = np.load(S.ETA_ROUGH)["table"]
    rl = np.load(S.RHO_LIVE)["rho"]
    rho = np.where(np.isfinite(rl), rl, P5["rho"])
    src = R["source"]
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    m = len(src)
    eta = np.zeros(m)
    lr_rel = np.zeros(m, np.int8)
    lr_mids = np.zeros((m, 10))
    lr_mb = np.zeros((m, 4, 10))
    lr_r2 = np.zeros((m, 4))
    for i in range(m):
        d = int(np.clip(R["n_hps"][i], 1, 10))
        eta[i] = F._eta_for(table[d], rho[i])
        le_r = B._seg(R, "leta", i)
        if not (len(le_r) and np.ptp(le_r) > 0.3):
            continue
        P = B._seg(R, "params", i)
        lr_rel[i] = 1
        Sr = norm.ppf(np.apply_along_axis(rankdata, 0, P[:, :4]) / (len(P) + 1))
        edges = np.quantile(le_r, np.linspace(0, 1, 11))
        lr_mids[i] = 0.5 * (edges[1:] + edges[:-1])
        for j in range(4):
            mb = np.array([Sr[(le_r >= a) & (le_r <= b), j].mean() if ((le_r >= a) & (le_r <= b)).any() else 0.0
                           for a, b in zip(edges[:-1], edges[1:])])
            lr_mb[i, j] = mb
            lr_r2[i, j] = float(np.clip(1 - np.var(Sr[:, j] - np.interp(le_r, lr_mids[i], mb)) /
                                        max(np.var(Sr[:, j]), 1e-12), 0, 0.95))
    out = {k: R[k] for k in KEYS}
    out["source"] = src.astype("U16")
    out["opt"] = R["opt"].astype("U8")
    out.update(weight=w, eta=eta, lr_rel=lr_rel, lr_mids=lr_mids, lr_mb=lr_mb, lr_r2=lr_r2)
    path = REPO / "release" / "lcfaker" / "assets" / "bpl_v16.npz"
    np.savez_compressed(path, **out)
    print(path, path.stat().st_size // 1024, "KB;", int((w > 0).sum()), "usable records;",
          {s: round(float(w[src == s].sum()), 3) for s in np.unique(src)})


if __name__ == "__main__":
    main()

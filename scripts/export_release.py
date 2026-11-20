"""Build release/lcfaker/assets/swt_v11.npz: everything the standalone generator needs, from the fit-split records.

Precomputes per record the sampling weight (by source, as curvefaker.faker_swt.faker) and the weight eta of the
rough landscape part (as faker_swt.sample_task, version 11), so the release needs no benchmark data.

    PYTHONPATH=. python scripts/export_release.py
"""
from __future__ import annotations

import numpy as np

from curvefaker import data, faker_hgp as F, faker_swt as S
from curvefaker.paths import REPO

KEYS = ["T", "n", "n_hps", "f", "x_range", "warps", "warp_offs", "share_dead", "share_div", "dead_level",
        "div_times", "div_offs", "noise_sd_diff", "noise_thirds_diff", "df"]


def main():
    R = dict(np.load(S.PATH_ALL))
    P5 = dict(np.load(F.PATH_V5))
    table = np.load(S.ETA_ROUGH)["table"]
    rho_live = np.load(S.RHO_LIVE)["rho"]
    src = np.array([t.split("/")[0] for t in R["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    eta = np.zeros(len(src))
    for i in range(len(src)):
        d = int(np.clip(R["n_hps"][i], 1, 10))
        target = rho_live[i] if np.isfinite(rho_live[i]) else P5["rho"][i]
        eta[i] = F._eta_for(table[d], target)
    out = {k: R[k] for k in KEYS}
    out.update(weight=w, eta=eta, source=src.astype("U16"))
    path = REPO / "release" / "lcfaker" / "assets" / "swt_v11.npz"
    np.savez_compressed(path, **out)
    print(path, path.stat().st_size // 1024, "KB;", int((w > 0).sum()), "usable records;",
          {s: round(float(w[src == s].sum()), 3) for s in np.unique(src)})


if __name__ == "__main__":
    main()

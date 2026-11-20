"""Analysis H3: does the learning-rate schedule act as the curve's clock? (TaskSet adam8p / nadamw, fit split)

Per live curve (one seed, normalised z, 51 points at steps 0, 200, ..., 10000), three fits, compared on held-out
points (every 5th: interpolation; last 30% after fitting the first 70%: extrapolation), relative to the curve's
noise sd:
    pow3     c + a t^(-alpha) in step time (t = (k + 1) / T)                         3 params
    anneal   L0 + A (S1 + s0)^(-alpha) - C S2, S1 / S2 from the config's own schedule  4 params
             (curvefaker/schedules.py; s0 = 200 steps; Tissue et al. 2024)
    shuffled the same with the schedule of another random config of the same family  4 params (control)
If the schedule is the clock, anneal beats shuffled clearly; if anneal only beats pow3 by about as much as shuffled
does, the gain is just the extra parameter.
Also direct checks on nadamw: rank correlation of the late share of progress (last 20%) with constant_fraction and
with min_learning_rate_mult.

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/schedule_clock.py [within]
(`within`: the nonparametric within-task test, see within_task_test.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from curvefaker import curves, data, families, schedules
from curvefaker.paths import REPO

RNG = np.random.default_rng(12)
DEV = "cuda"


def collect(opt, n_tasks=150, per=12):
    ids = [i for i in data.ids("fit", ["taskset"]) if i.endswith("/" + opt)]
    ids = list(RNG.choice(ids, min(n_tasks, len(ids)), replace=False))
    Y, H = [], []
    for tid in ids:
        t = data.load(tid)
        z, _ = curves.normalise(t.y[:, 0])
        ok = np.isfinite(z).all(1)
        c = np.full(len(z), 3)
        c[ok] = curves.classify(z[ok])
        s = curves.smooth(np.nan_to_num(z, nan=curves.CAP))
        live = np.where(((c == 0) | (c == 2)) & (s[:, 0] - s.min(1) >= 0.1))[0]
        pick = RNG.choice(live, min(per, len(live)), replace=False) if len(live) else []
        for i in pick:
            Y.append(z[i])
            H.append(t.hps.iloc[i].to_dict())
        steps = t.steps
    return np.array(Y, np.float32), H, np.asarray(steps)


def fit_anneal(Y, S1, S2, mask, steps=2000, s0=200.0):
    y = torch.tensor(Y, device=DEV)
    s1 = torch.tensor(S1 + s0, dtype=torch.float32, device=DEV)
    s2 = torch.tensor(S2, dtype=torch.float32, device=DEV)
    s1 = s1 / s1[:, -1:]                              # scale-free per curve
    s2 = s2 / torch.clamp(s2[:, -1:], min=1e-6)
    m = torch.tensor(mask, device=DEV)
    best, best_f = torch.full((len(y),), float("inf"), device=DEV), torch.zeros_like(y)
    for la0 in (-1.0, 0.0):
        L0 = torch.nn.Parameter(y[:, -1].clone())
        lA = torch.nn.Parameter(torch.log((y[:, 0] - y[:, -1]).abs() + 1e-3))
        la = torch.nn.Parameter(torch.full((len(y),), la0, device=DEV))
        C = torch.nn.Parameter(torch.zeros(len(y), device=DEV))
        opt = torch.optim.Adam([L0, lA, la, C], lr=0.03)
        f = lambda: L0[:, None] + torch.exp(lA)[:, None] * s1 ** (-torch.exp(la)[:, None]) - C[:, None] * s2
        for _ in range(steps):
            opt.zero_grad()
            loss = ((f()[:, m] - y[:, m]) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ff = f()
            sse = ((ff[:, m] - y[:, m]) ** 2).sum(1)
            better = sse < best
            best = torch.where(better, sse, best)
            best_f[better] = ff[better]
    return best_f.cpu().numpy()


def main():
    rows = []
    for opt in ("nadamw", "adam8p"):
        Y, H, steps = collect(opt)
        T = Y.shape[1]
        sd = np.maximum(families.noise_sd(Y), 1e-4)
        A = [schedules.areas(opt, h, steps) for h in H]
        S1, S2 = np.stack([a[0] for a in A]), np.stack([a[1] for a in A])
        perm = RNG.permutation(len(Y))
        idx = np.arange(T)
        masks = {"interp": idx % 5 != 2, "extrap": idx < int(0.7 * T)}
        for split, mk in masks.items():
            out = {"pow3": families.fit("pow3", Y, mk)[0],
                   "anneal": fit_anneal(Y, S1, S2, mk),
                   "shuffled": fit_anneal(Y, S1[perm], S2[perm], mk)}
            for name, f in out.items():
                r = np.sqrt(((f - Y)[:, ~mk] ** 2).mean(1)) / sd
                rows.append({"opt": opt, "eval": split, "model": name, "n": len(Y), "rel_rmse_med": float(np.median(r)),
                             "rel_rmse_mean": float(np.mean(np.minimum(r, 50)))})
            if split == "extrap":
                d = np.sqrt(((out["shuffled"] - Y)[:, ~mk] ** 2).mean(1)) / sd - \
                    np.sqrt(((out["anneal"] - Y)[:, ~mk] ** 2).mean(1)) / sd
                rows.append({"opt": opt, "eval": "extrap", "model": "shuffled - anneal (paired)",
                             "rel_rmse_med": float(np.median(d)), "rel_rmse_mean": float(np.mean(np.clip(d, -50, 50))),
                             "share_anneal_better": float((d > 0).mean()), "n": len(Y)})
        if opt == "nadamw":
            s = curves.smooth(Y)
            p = (s[:, :1] - s) / (s[:, :1] - s.min(1, keepdims=True))
            late = 1 - p[:, int(0.8 * (T - 1))]
            hp = pd.DataFrame(H)
            for k in ("constant_fraction", "min_learning_rate_mult", "warmup_fraction", "learning_rate"):
                rows.append({"opt": opt, "eval": "late_share vs hp", "model": k, "n": len(Y),
                             "rel_rmse_med": curves.spearman(late, hp[k].to_numpy(float))})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(REPO / "results" / "schedule_clock.csv", index=False)


if __name__ == "__main__" and "within" not in __import__("sys").argv:
    main()


def within_task_test(opt, n_tasks=300):
    """Nonparametric version: within each task, does more annealing (S2 at the end of training, from the config's
    schedule) go with more progress in the second half? Spearman per task between the share of the curve's total
    gain made after 50% of the budget and S2(T); median over tasks and share of tasks with a positive correlation.
    Same for the learning rate (comparison)."""
    ids = [i for i in data.ids("fit", ["taskset"]) if i.endswith("/" + opt)]
    ids = list(RNG.choice(ids, min(n_tasks, len(ids)), replace=False))
    out = {"anneal": [], "learning_rate": [], "anneal_given_lr": [], "s1_second_half_share": [],
           "s1_share_given_lr": []}
    for tid in ids:
        t = data.load(tid)
        z, _ = curves.normalise(t.y[:, 0])
        ok = np.isfinite(z).all(1)
        c = np.full(len(z), 3)
        c[ok] = curves.classify(z[ok])
        s = curves.smooth(np.nan_to_num(z, nan=curves.CAP))
        gain = s[:, 0] - s.min(1)
        live = ((c == 0) | (c == 2)) & (gain >= 0.1)
        if live.sum() < 30:
            continue
        T = s.shape[1]
        late = (s[live, T // 2] - curves.final_value(s[live])) / gain[live]
        H = t.hps[live].to_dict("records")
        AR = [schedules.areas(opt, h, t.steps) for h in H]
        s2 = np.array([a[1][-1] for a in AR])
        s1h = np.array([(a[0][-1] - a[0][T // 2]) / a[0][-1] for a in AR])   # share of cumulative LR in 2nd half
        lr = np.log(np.array([h["learning_rate"] for h in H]))
        if np.std(s2) == 0:
            continue
        out["anneal"].append(curves.spearman(late, s2))
        out["learning_rate"].append(curves.spearman(late, lr))
        # partial: residualise ranks of late and s2 on the rank of lr
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        rl, rs, rr = rk(late), rk(s2), rk(lr)
        res = lambda a: a - np.polyval(np.polyfit(rr, a, 1), rr)
        out["anneal_given_lr"].append(float(np.corrcoef(res(rl), res(rs))[0, 1]))
        out["s1_second_half_share"].append(curves.spearman(late, s1h))
        rh = rk(s1h)
        out["s1_share_given_lr"].append(float(np.corrcoef(res(rl), res(rh))[0, 1]) if np.std(res(rh)) > 0 else np.nan)
    rows = []
    for k, v in out.items():
        v = np.array(v)
        rows.append({"opt": opt, "eval": "within-task late gain vs", "model": k, "n": len(v),
                     "rel_rmse_med": float(np.nanmedian(v)), "share_anneal_better": float(np.nanmean(v > 0))})
    return rows


if __name__ == "__main__" and "within" in __import__("sys").argv:
    rows = within_task_test("nadamw") + within_task_test("adam8p")
    df = pd.DataFrame(rows).rename(columns={"rel_rmse_med": "median_spearman", "share_anneal_better": "share_positive",
                                            "n": "tasks"})
    print(df.round(3).to_string(index=False))
    df.to_csv(REPO / "results" / "schedule_clock_within.csv", index=False)

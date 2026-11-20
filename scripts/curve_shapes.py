"""Question B: what do curves look like? Curve families compared on held-out curves.

Curves: replica-mean normalised z (curvefaker.curves), resampled to G = 50 points evenly over the budget; curves
that hit the divergence cap are left out (divergence is its own class). Learned models are fitted on curves of the
*fit* split, everything is evaluated on curves of the *select* split (TaskSet select families, FCNet, bbomix).

Families:
    pow3     y = c + a * u^(-b),  u = (k+1)/G          (3 parameters per curve)
    exp3     y = c + a * exp(-b u)                     (3)
    pca-k    mean curve + k principal components learned on fit curves (k parameters)
    vae-k    decoder of a variational autoencoder with a k-dim latent, trained on fit curves (k)
    knn      the 50 fit curves closest on the observed part (extrapolation only)
    last     the last observed value (extrapolation only)

Scores:
    rmse_full : RMSE of the best fit to the whole curve (how well the family can represent real curves)
    mae_final : observe the first 20% of the curve, predict the final value (mean of the last 10%): mean abs error
    rank_final: per task, rank correlation between predicted and true final values (median over tasks)

Diagnostic (`loso` argument): within the *fit* split, leave out JAHS, NAS-Bench-201 or QuickTune in turn, fit on the
rest, test on it (the select split has no cosine-schedule source, where curves cross most).

    CUDA_VISIBLE_DEVICES=<one gpu> PYTHONPATH=. python scripts/curve_shapes.py [loso]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

from curvefaker import curves, data
from curvefaker.paths import REPO

G = 50
OBS = 10                      # observed prefix: first 20% of the budget
DEV = "cuda"
RNG = np.random.default_rng(4)
TS_FIT_TASKS, TS_SEL_TASKS, PER_TASK = 300, 150, 60


def resample(m):
    T = m.shape[1]
    if T == G:
        return m
    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, G)
    return np.stack([np.interp(dst, src, r) for r in m])


def collect(split):
    out, task = [], []
    srcs = [s for s in data.SOURCES]
    for src in srcs:
        ids = data.ids(split=split, sources=[src])
        if not ids:
            continue
        if src == "taskset":
            ids = list(RNG.choice(ids, TS_FIT_TASKS if split == "fit" else TS_SEL_TASKS, replace=False))
        per = PER_TASK if len(ids) > 20 else 1500 // len(ids)
        for tid in ids:
            t = data.load(tid)
            y = t.y
            if src == "quicktune":
                y = y[np.isfinite(y[:, 0, -1])]
            z, _ = curves.normalise(y)
            m = curves.mean_curve(z)
            m = m[np.isfinite(m).all(1) & (m < curves.CAP).all(1)]
            m = m[RNG.choice(len(m), min(per, len(m)), replace=False)]
            out.append(resample(m))
            task += [tid] * len(m)
    return np.concatenate(out).astype(np.float32), np.array(task)


# ---- parametric families (batched least squares with Adam) ----------------------------------------------------
def fit_param(Y, mask, kind, steps=1500):
    """Y [n, G], mask [G] bool: fit on masked points, return the fitted full curves [n, G]."""
    y = torch.tensor(Y, device=DEV)
    u = torch.arange(1, G + 1, device=DEV, dtype=torch.float32) / G
    msk = torch.tensor(mask, device=DEV)
    n = len(y)
    obs = y[:, msk]
    c = torch.nn.Parameter(obs[:, -1].clone())
    la = torch.nn.Parameter(torch.log((obs[:, 0] - obs[:, -1]).abs() + 1e-3))
    sgn = torch.sign(obs[:, 0] - obs[:, -1] + 1e-9)
    lb = torch.nn.Parameter(torch.zeros(n, device=DEV))
    opt = torch.optim.Adam([c, la, lb], lr=0.03)

    def curve():
        a = sgn[:, None] * torch.exp(la)[:, None]
        b = torch.exp(lb)[:, None]
        if kind == "pow3":
            return c[:, None] + a * (u[None] ** (-b) - 1) / (u[None, 0] ** (-b) - 1 + 1e-6) * 1.0
        return c[:, None] + a * torch.exp(-b * 5 * u[None])

    for _ in range(steps):
        opt.zero_grad()
        loss = ((curve()[:, msk] - obs) ** 2).mean(1).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            lb.clamp_(-5, 3)
    with torch.no_grad():
        return curve().cpu().numpy()


# ---- PCA ----------------------------------------------------------------------------------------------------------
class PCA:
    def __init__(self, Yfit, k):
        self.mu = Yfit.mean(0)
        U, S, Vt = np.linalg.svd(Yfit - self.mu, full_matrices=False)
        self.W = Vt[:k]                                    # [k, G]
        self.var = (S[:k] ** 2) / len(Yfit)                # latent prior variances
        self.noise = max(float(((S[k:] ** 2).sum()) / len(Yfit) / G), 1e-6)

    def fit(self, Y, mask):
        """Posterior mean latent given the masked points (probabilistic PCA), return full curves."""
        W = self.W[:, mask]
        A = W @ W.T / self.noise + np.diag(1 / self.var)
        b = (Y[:, mask] - self.mu[mask]) @ W.T / self.noise
        z = np.linalg.solve(A, b.T).T
        return self.mu + z @ self.W


# ---- VAE ----------------------------------------------------------------------------------------------------------
class VAE(torch.nn.Module):
    def __init__(self, k, h=256):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Sequential(torch.nn.Linear(G, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                       torch.nn.GELU(), torch.nn.Linear(h, 2 * k))
        self.dec = torch.nn.Sequential(torch.nn.Linear(k, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                       torch.nn.GELU(), torch.nn.Linear(h, G))
        self.log_noise = torch.nn.Parameter(torch.tensor(-3.0))

    def loss(self, y):
        mu, lv = self.enc(y).chunk(2, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rec = self.dec(z)
        s2 = torch.exp(2 * self.log_noise)
        nll = 0.5 * ((rec - y) ** 2 / s2).sum(1) + G * self.log_noise
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - lv - 1).sum(1)
        return (nll + kl).mean()

    def fit_latent(self, Y, mask, steps=500):
        """MAP latent given the masked points (Gaussian noise with the learned scale, N(0, I) prior)."""
        y = torch.tensor(Y, device=DEV)
        msk = torch.tensor(mask, device=DEV)
        with torch.no_grad():
            z0 = self.enc(torch.where(msk[None], y, y[:, msk][:, -1:]))[:, :self.k]   # pad unseen with last value
        z = torch.nn.Parameter(z0.clone())
        opt = torch.optim.Adam([z], lr=0.05)
        s2 = float(torch.exp(2 * self.log_noise))
        for _ in range(steps):
            opt.zero_grad()
            l = (0.5 * ((self.dec(z)[:, msk] - y[:, msk]) ** 2).sum(1) / s2 + 0.5 * (z ** 2).sum(1)).sum()
            l.backward()
            opt.step()
        with torch.no_grad():
            return self.dec(z).cpu().numpy()


def train_vae(Yfit, k, epochs=60):
    torch.manual_seed(0)
    v = VAE(k).to(DEV)
    for p in v.dec.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    y = torch.tensor(Yfit, device=DEV)
    for ep in range(epochs):
        perm = torch.randperm(len(y), device=DEV)
        for i in range(0, len(y), 256):
            opt.zero_grad()
            l = v.loss(y[perm[i:i + 256]])
            l.backward()
            opt.step()
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def knn_final(Yfit, Y, k=50):
    a = torch.tensor(Y[:, :OBS], device=DEV)
    b = torch.tensor(Yfit[:, :OBS], device=DEV)
    fin = torch.tensor(curves.final_value(Yfit), device=DEV)
    out = []
    for i in range(0, len(a), 2048):
        d = torch.cdist(a[i:i + 2048], b)
        idx = d.topk(k, largest=False).indices
        out.append(fin[idx].mean(1))
    return torch.cat(out).cpu().numpy()


def main():
    t0 = time.time()
    Yfit, tfit = collect("fit")
    if "loso" in sys.argv[1:]:   # diagnostic within *fit*: leave one source out, test on it
        sfit = np.array([t.split("/")[0] for t in tfit])
        frames = []
        for held in ("jahs", "nasbench201", "quicktune"):
            frames.append(evaluate(Yfit[sfit != held], Yfit[sfit == held], tfit[sfit == held], t0))
        df = pd.concat(frames)
        df = df[df.source != "all"]
        out = REPO / "results" / "curve_shapes_loso.csv"
    else:
        Ysel, tsel = collect("select")
        df = evaluate(Yfit, Ysel, tsel, t0)
        out = REPO / "results" / "curve_shapes.csv"
    df.to_csv(out, index=False)
    for col in ("rmse_full", "mae_final", "rank_final"):
        print(f"\n{col}\n", df.pivot(index="model", columns="source", values=col).round(3).to_string())


def evaluate(Yfit, Ysel, tsel, t0):
    src = np.array([t.split("/")[0] for t in tsel])
    print(f"fit curves {len(Yfit)}, select curves {len(Ysel)} ({time.time() - t0:.0f}s)", flush=True)
    full = np.ones(G, bool)
    pre = np.arange(G) < OBS
    true_fin = curves.final_value(Ysel)
    preds_full, preds_fin = {}, {}
    for kind in ("pow3", "exp3"):
        preds_full[kind] = fit_param(Ysel, full, kind)
        preds_fin[kind] = curves.final_value(fit_param(Ysel, pre, kind))
    for k in (1, 2, 4, 8):
        p = PCA(Yfit, k)
        preds_full[f"pca-{k}"] = p.fit(Ysel, full)
        preds_fin[f"pca-{k}"] = curves.final_value(p.fit(Ysel, pre))
    for k in (2, 4, 8):
        v = train_vae(Yfit, k)
        preds_full[f"vae-{k}"] = v.fit_latent(Ysel, full)
        preds_fin[f"vae-{k}"] = curves.final_value(v.fit_latent(Ysel, pre))
        print(f"vae-{k} done ({time.time() - t0:.0f}s)", flush=True)
    preds_fin["knn"] = knn_final(Yfit, Ysel)
    preds_fin["last"] = Ysel[:, OBS - 1]
    rows = []
    for name in list(preds_fin):
        for s in ["all"] + sorted(set(src)):
            sel = np.ones(len(src), bool) if s == "all" else src == s
            r = {"model": name, "source": s}
            if name in preds_full:
                r["rmse_full"] = float(np.sqrt(((preds_full[name][sel] - Ysel[sel]) ** 2).mean()))
            r["mae_final"] = float(np.abs(preds_fin[name][sel] - true_fin[sel]).mean())
            rk = [curves.spearman(preds_fin[name][(tsel == t)], true_fin[tsel == t])
                  for t in np.unique(tsel[sel]) if (tsel == t).sum() >= 10]
            r["rank_final"] = float(np.nanmedian(rk))
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

"""A generative model of whole tasks' curve sets (question B, learned; GPU).

Data: per fit task, S random configs' curves of one seed, normalised z (curvefaker.curves; diverged values at the
cap), resampled to G = 50 points over the budget.

Model (a hierarchical VAE over sets):
    task latent   tau ~ N(0, I_16)          inferred from the mean of per-curve embeddings (a set encoder)
    config latent c_i ~ N(0, I_4)           inferred from curve i and tau
    curve         y_i ~ N(dec(tau, c_i), diag(sigma(tau)^2))   sigma per time point, set by the task
Trained by the evidence lower bound. A task's curve distribution p(y | tau) = int p(y | tau, c) N(c) dc can be
multimodal (dead clusters, divergence) because the decoder is nonlinear and task-conditioned, unlike the per-task
Gaussian of faker v0.

Generation (`sample_task`): tau from its prior; the config latents c(x) of a new task come from 4 GP fields over
random configs (the landscape kernel of curvefaker.faker_hgp, with the nugget of faker v5: d and eta taken from a
random fit record); curves = decoder mean + Gaussian noise with the task's sigma.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from curvefaker import curves, data
from curvefaker.evaluate import resample
from curvefaker.paths import REPO

G = 50
S = 128
DT, DC, H = 16, 4, 256
DATA_PATH = REPO / "results" / "setvae_data.npz"
MODEL_PATH = REPO / "results" / "setvae.pt"


def build_data(seed=0):
    rng = np.random.default_rng(seed)
    Y, ids = [], []
    for tid in data.ids("fit"):
        t = data.load(tid)
        y = t.y[:, 0]
        if t.source == "quicktune":
            y = y[np.isfinite(y[:, -1])]
        if len(y) < 32:
            continue
        idx = rng.choice(len(y), S, replace=len(y) < S)
        z, _ = curves.normalise(y)
        Y.append(resample(np.nan_to_num(z[idx], nan=curves.CAP), G).astype(np.float32))
        ids.append(tid)
    np.savez(DATA_PATH, Y=np.stack(Y), task=np.array(ids))


def mlp(i, o, h=H, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class SetVAE(nn.Module):
    def __init__(self, nu: float | None = None):
        """nu: Student-t degrees of freedom of the curve likelihood (None = Gaussian)."""
        super().__init__()
        self.nu = nu
        self.emb = mlp(G, H)
        self.task_post = mlp(H, 2 * DT)
        self.cfg_post = mlp(H + DT, 2 * DC)
        self.dec = mlp(DT + DC, G, n=3)
        self.noise = mlp(DT, G, h=128)

    def forward(self, y):
        """y [B, S, G] -> negative ELBO per task (mean over points), parts."""
        B, S_, _ = y.shape
        h = self.emb(y)                                              # [B, S, H]
        mt, lt = self.task_post(h.mean(1)).chunk(2, -1)
        tau = mt + torch.randn_like(mt) * torch.exp(0.5 * lt)
        mc, lc = self.cfg_post(torch.cat([h, tau[:, None].expand(-1, S_, -1)], -1)).chunk(2, -1)
        c = mc + torch.randn_like(mc) * torch.exp(0.5 * lc)
        mu = self.dec(torch.cat([tau[:, None].expand(-1, S_, -1), c], -1))
        log_sig = self.noise(tau).clamp(-9, 1)[:, None]               # [B, 1, G]
        r = (y - mu) / torch.exp(log_sig)
        if self.nu is None:
            nll = (0.5 * r ** 2 + log_sig + 0.5 * math.log(2 * math.pi)).sum((1, 2))
        else:
            nu = self.nu
            const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            nll = (-const + log_sig + (nu + 1) / 2 * torch.log1p(r ** 2 / nu)).sum((1, 2))
        kl_c = 0.5 * (mc ** 2 + torch.exp(lc) - lc - 1).sum((1, 2))
        kl_t = 0.5 * (mt ** 2 + torch.exp(lt) - lt - 1).sum(1)
        n_pts = S_ * G
        return (nll + kl_c + kl_t) / n_pts, {"nll": nll / n_pts, "kl_c": kl_c / n_pts, "kl_t": kl_t / n_pts}

    @torch.no_grad()
    def generate(self, c, tau=None, rng=None):
        """c [n, DC] config latents -> curves [n, G] (decoder mean + noise) for one task."""
        dev = next(self.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(int(rng.integers(1 << 30))) if rng is not None else None
        if tau is None:
            tau = torch.randn(DT, generator=g)
        tau = tau.to(dev)
        c = torch.as_tensor(c, dtype=torch.float32, device=dev)
        mu = self.dec(torch.cat([tau[None].expand(len(c), -1), c], -1))
        sig = torch.exp(self.noise(tau[None]).clamp(-9, 1))
        eps = torch.randn(mu.shape, generator=g).to(dev)
        return (mu + eps * sig).cpu().numpy()


def train(epochs=200, batch=32, lr=3e-4, device="cuda", log=print, data_path=None, model_path=None, nu=None):
    Z = dict(np.load(data_path or DATA_PATH))
    Y = torch.tensor(Z["Y"])
    n = len(Y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    va, tr = perm[: n // 20], perm[n // 20:]                         # 5% of fit tasks for monitoring only
    torch.manual_seed(0)
    model = SetVAE(nu).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Ytr, Yva = Y[tr].to(device), Y[va].to(device)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Ytr), device=device)
        tot = 0.0
        for i in range(0, len(order), batch):
            yb = Ytr[order[i:i + batch]]
            yb = yb[:, torch.randperm(S, device=device)[:64]]          # random subset of the set
            loss, _ = model(yb)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(yb)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                lv, parts = model(Yva)
            log(f"epoch {ep} train {tot / len(Ytr):.4f} val {float(lv.mean()):.4f} "
                f"nll {float(parts['nll'].mean()):.4f} kl_c {float(parts['kl_c'].mean()):.4f} "
                f"kl_t {float(parts['kl_t'].mean()):.4f}")
    torch.save(model.state_dict(), model_path or MODEL_PATH)
    return model


def load(device="cpu", path=None, nu=None):
    m = SetVAE(nu)
    m.load_state_dict(torch.load(path or MODEL_PATH, map_location=device))
    return m.to(device).eval()


def faker(n_tasks, rng, n_cfg=1000):
    from curvefaker import faker_hgp as F
    model = load()
    P = dict(np.load(F.PATH_V1))
    C5 = dict(np.load(F.PATH_V5))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src])
    w /= w.sum()
    out = []
    for _ in range(n_tasks):
        i = int(rng.choice(len(w), p=w))
        d = int(np.clip(P["n_hps"][i], 1, 10))
        eta = F._eta_for(C5["table"][d], C5["rho"][i])
        X = rng.random((n_cfg, d))
        f = F._kernel_field(rng, X, DC)
        f = (f - f.mean(0)) / f.std(0)
        if eta > 0:
            f = math.sqrt(1 - eta) * f + math.sqrt(eta) * rng.standard_normal(f.shape)
        y = model.generate(f, rng=rng)
        out.append((np.clip(y, -0.5, curves.CAP)[:, None, :].astype(np.float32), X))
    return out


# ---- v2: sample the task latent from the aggregate posterior --------------------------------------------------------
# Change (one): the first set-VAE faker drew tau from the N(0, I) prior and failed (coverage 0): the fit tasks'
# posteriors are narrow (sd ~0.06) and sit in a small part of the prior ("prior hole"); there the noise head
# returns sigma ~0.33 instead of ~0.03. v2 fits a Gaussian mixture (20 components, full covariance) to the posterior
# means of all fit tasks and draws tau from it.
GMM_PATH = REPO / "results" / "setvae_gmm.npz"


def fit_gmm(k=20, seed=0):
    from sklearn.mixture import GaussianMixture
    m = load()
    Y = torch.tensor(np.load(DATA_PATH)["Y"])
    with torch.no_grad():
        mt = torch.cat([m.task_post(m.emb(Y[i:i + 256]).mean(1)).chunk(2, -1)[0] for i in range(0, len(Y), 256)])
    gm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(mt.numpy())
    np.savez(GMM_PATH, w=gm.weights_, mu=gm.means_, cov=gm.covariances_)


def faker_v2(n_tasks, rng, n_cfg=1000):
    from curvefaker import faker_hgp as F
    model = load()
    gm = dict(np.load(GMM_PATH))
    P = dict(np.load(F.PATH_V1))
    C5 = dict(np.load(F.PATH_V5))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src])
    w /= w.sum()
    out = []
    for _ in range(n_tasks):
        k = rng.choice(len(gm["w"]), p=gm["w"] / gm["w"].sum())
        tau = torch.tensor(rng.multivariate_normal(gm["mu"][k], gm["cov"][k]), dtype=torch.float32)
        i = int(rng.choice(len(w), p=w))
        d = int(np.clip(P["n_hps"][i], 1, 10))
        eta = F._eta_for(C5["table"][d], C5["rho"][i])
        X = rng.random((n_cfg, d))
        f = F._kernel_field(rng, X, DC)
        f = (f - f.mean(0)) / f.std(0)
        if eta > 0:
            f = math.sqrt(1 - eta) * f + math.sqrt(eta) * rng.standard_normal(f.shape)
        y = model.generate(f, tau=tau, rng=rng)
        out.append((np.clip(y, -0.5, curves.CAP)[:, None, :].astype(np.float32), X))
    return out

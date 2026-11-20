"""Faker "comp": explicit components, with learned live curves.

Fitted on the *fit* split only:
    live model : the set-VAE of curvefaker/setvae.py trained on the *live* curves of fit tasks only (improving or
                 turning, see curvefaker.curves.classify), with a Student-t likelihood (nu = 3) so heavy-tailed
                 noise does not inflate the noise scale (the failure of the Gaussian set-VAE).
    records    : the v1 / v5 records of curvefaker.faker_hgp (one per real fit task): number of hps, landscape
                 nugget eta, class shares, dead level, plus the live model's posterior task latent tau.

Generating one task:
    1. A record i, drawn in proportion to its source in the fit split (as v5).
    2. Task latent tau = tau_i + N(0, s^2 I), s = TAU_JITTER (new tasks near a real one).
    3. Configs uniform in [0, 1]^d; 4 GP landscape fields (+ nugget eta_i) -> config latents -> live curves
       (decoder mean + Student-t noise with the task's per-time scale).
    4. Dead configs: the top share_dead_i of an independent GP field; flat at the record's dead level + the task's
       noise. Diverged configs: the top share_div_i of another field; their live curve jumps to the cap at a
       uniform random time.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from curvefaker import curves, data, faker_hgp as F, setvae as V
from curvefaker.paths import REPO

LIVE_DATA = REPO / "results" / "comp_live_data.npz"
LIVE_MODEL = REPO / "results" / "comp_live.pt"
RECORDS = REPO / "results" / "comp_records.npz"
NU = 3.0
TAU_JITTER = 0.3


def build_live_data(seed=0):
    rng = np.random.default_rng(seed)
    Z = dict(np.load(V.DATA_PATH))
    Y = Z["Y"]
    out, ids = [], []
    for k in range(len(Y)):
        c = curves.classify(Y[k])
        live = np.where((c == 0) | (c == 2))[0]
        if len(live) < 16:
            continue
        out.append(Y[k][rng.choice(live, V.S, replace=len(live) < V.S)])
        ids.append(Z["task"][k])
    np.savez(LIVE_DATA, Y=np.stack(out), task=np.array(ids))


def train_live(**kw):
    return V.train(data_path=LIVE_DATA, model_path=LIVE_MODEL, nu=NU, **kw)


def fit_records():
    """Posterior task latents of the live model for every v1 record, plus class shares and dead levels."""
    model = V.load(path=LIVE_MODEL, nu=NU)
    P = dict(np.load(F.PATH_V1))
    cls = F._classes(P)
    tau, share_dead, share_div, dead_level, ok = [], [], [], [], []
    for i in range(len(P["task"])):
        sl = slice(P["offs"][i], P["offs"][i + 1])
        C, c = P["curves"][sl], cls[sl]
        live = C[(c == 0) | (c == 2)]
        share_dead.append((c == 1).mean())
        share_div.append((c == 3).mean())
        dead_level.append(float(np.median(curves.final_value(C[c == 1]))) if (c == 1).any() else 1.0)
        if len(live) < 16:
            tau.append(np.zeros(V.DT))
            ok.append(False)
            continue
        with torch.no_grad():
            h = model.emb(torch.tensor(np.nan_to_num(live, nan=curves.CAP), dtype=torch.float32)[None])
            mt, _ = model.task_post(h.mean(1)).chunk(2, -1)
        tau.append(mt[0].numpy())
        ok.append(True)
    np.savez(RECORDS, tau=np.stack(tau), share_dead=np.array(share_dead), share_div=np.array(share_div),
             dead_level=np.array(dead_level), ok=np.array(ok))


def faker(n_tasks, rng, n_cfg=1000, tau_jitter=TAU_JITTER):
    model = V.load(path=LIVE_MODEL, nu=NU)
    P = dict(np.load(F.PATH_V1))
    C5 = dict(np.load(F.PATH_V5))
    R = dict(np.load(RECORDS))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    out = []
    for _ in range(n_tasks):
        i = int(rng.choice(len(w), p=w))
        d = int(np.clip(P["n_hps"][i], 1, 10))
        eta = F._eta_for(C5["table"][d], C5["rho"][i])
        X = rng.random((n_cfg, d))
        f = F._kernel_field(rng, X, V.DC + 2)
        f = (f - f.mean(0)) / f.std(0)
        if eta > 0:
            f[:, :V.DC] = math.sqrt(1 - eta) * f[:, :V.DC] + math.sqrt(eta) * rng.standard_normal((n_cfg, V.DC))
        tau = torch.tensor(R["tau"][i] + tau_jitter * rng.standard_normal(V.DT), dtype=torch.float32)
        with torch.no_grad():
            c = torch.tensor(f[:, :V.DC], dtype=torch.float32)
            mu = model.dec(torch.cat([tau[None].expand(n_cfg, -1), c], -1)).numpy()
            sig = torch.exp(model.noise(tau[None]).clamp(-9, 1)).numpy()            # [1, G]
        noise = rng.standard_t(NU, mu.shape) * sig
        y = mu + noise
        nd = int(round(R["share_dead"][i] * n_cfg))
        if nd:
            dead = np.argsort(-f[:, V.DC])[:nd]
            y[dead] = R["dead_level"][i] + noise[dead]
        nv = int(round(R["share_div"][i] * n_cfg))
        if nv:
            div = np.argsort(-f[:, V.DC + 1])[:nv]
            for a, t0 in zip(div, rng.integers(1, V.G, nv)):
                y[a, t0:] = curves.CAP
        out.append((np.clip(y, -0.5, curves.CAP)[:, None, :].astype(np.float32), X))
    return out


# ---- comp-v2: config latents from the record's own posterior (latent-space copula) ---------------------------------
# Change (one): comp drew the 4 config latents as independent N(0, 1) fields. Within a real task the posterior
# latents are correlated (median |r| 0.42) and two of the four dimensions barely vary (sd 0.02-0.03), so N(0, I)
# draws decode to curves the model never saw (rank(early, final) 0.77 vs 0.87 real; shuffling the dimensions
# independently: 0.69). comp-v2 keeps each record's posterior latents of its live curves, jitters them (sd
# C_JITTER x their per-dimension sd), and hands them to configs in order of a GP field (rank of field = rank of
# decoded final value), as faker v1 did with real curves.
RECORDS_C = REPO / "results" / "comp_records_c.npz"
C_JITTER = 0.2


def fit_records_c():
    model = V.load(path=LIVE_MODEL, nu=NU)
    P = dict(np.load(F.PATH_V1))
    R = dict(np.load(RECORDS))
    cls = F._classes(P)
    blocks, offs = [], [0]
    for i in range(len(P["task"])):
        sl = slice(P["offs"][i], P["offs"][i + 1])
        C, c = P["curves"][sl], cls[sl]
        live = C[(c == 0) | (c == 2)]
        if not R["ok"][i]:
            blocks.append(np.zeros((0, V.DC), np.float32))
            offs.append(offs[-1])
            continue
        with torch.no_grad():
            yt = torch.tensor(np.nan_to_num(live, nan=curves.CAP), dtype=torch.float32)[None]
            h = model.emb(yt)
            tau = torch.tensor(R["tau"][i], dtype=torch.float32)[None, None].expand(1, len(live), -1)
            mc, _ = model.cfg_post(torch.cat([h, tau], -1)).chunk(2, -1)
        blocks.append(mc[0].numpy())
        offs.append(offs[-1] + len(live))
    np.savez(RECORDS_C, c=np.concatenate(blocks), offs=np.array(offs))


def faker_v2(n_tasks, rng, n_cfg=1000, tau_jitter=TAU_JITTER, c_jitter=C_JITTER):
    model = V.load(path=LIVE_MODEL, nu=NU)
    P = dict(np.load(F.PATH_V1))
    C5 = dict(np.load(F.PATH_V5))
    R = dict(np.load(RECORDS))
    RC = dict(np.load(RECORDS_C))
    src = np.array([t.split("/")[0] for t in P["task"]])
    counts = {s: len(data.ids("fit", [s])) for s in np.unique(src)}
    w = np.array([counts[s] / (src == s).sum() for s in src]) * R["ok"]
    w /= w.sum()
    out = []
    for _ in range(n_tasks):
        i = int(rng.choice(len(w), p=w))
        d = int(np.clip(P["n_hps"][i], 1, 10))
        eta = F._eta_for(C5["table"][d], C5["rho"][i])
        X = rng.random((n_cfg, d))
        f = F._kernel_field(rng, X, 3)
        f = (f - f.mean(0)) / f.std(0)
        if eta > 0:
            f[:, 0] = math.sqrt(1 - eta) * f[:, 0] + math.sqrt(eta) * rng.standard_normal(n_cfg)
        tau = torch.tensor(R["tau"][i] + tau_jitter * rng.standard_normal(V.DT), dtype=torch.float32)
        cs = RC["c"][RC["offs"][i]:RC["offs"][i + 1]]
        cs = cs[rng.integers(len(cs), size=n_cfg)]
        cs = cs + c_jitter * cs.std(0, keepdims=True) * rng.standard_normal(cs.shape)
        with torch.no_grad():
            dec = lambda c: model.dec(torch.cat([tau[None].expand(len(c), -1),
                                                 torch.tensor(c, dtype=torch.float32)], -1)).numpy()
            mu = dec(cs)
            sig = torch.exp(model.noise(tau[None]).clamp(-9, 1)).numpy()
        order = np.argsort(curves.final_value(mu), kind="stable")
        mu_assigned = np.empty_like(mu)
        mu_assigned[np.argsort(f[:, 0])] = mu[order]
        noise = rng.standard_t(NU, mu.shape) * sig
        y = mu_assigned + noise
        nd = int(round(R["share_dead"][i] * n_cfg))
        if nd:
            dead = np.argsort(-f[:, 1])[:nd]
            y[dead] = R["dead_level"][i] + noise[dead]
        nv = int(round(R["share_div"][i] * n_cfg))
        if nv:
            div = np.argsort(-f[:, 2])[:nv]
            for a, t0 in zip(div, rng.integers(1, V.G, nv)):
                y[a, t0:] = curves.CAP
        out.append((np.clip(y, -0.5, curves.CAP)[:, None, :].astype(np.float32), X))
    return out

"""Fakers: the ifBO prior (rogue-dagger's batched port) plus optional components, each with a per-task strength.

    Faker = make_faker(dead=dict(p_task=0.7, frac=(0.02, 0.5)))
    pr = Faker(n_tasks, d_max, device, generator=g)      # same constructor as roguedagger.prior.PriorTasks
    pr.resample(idx, d); y = pr.curves(x, steps_per_run, t_max, idx)

With no components, a Faker is exactly the ifBO prior (tests/test_fakers.py checks this draw for draw), so each
component is a single, separable change (README "Rule 2"). Components (docs/avenues.md):

dead   (V2a) a dead region per task: with probability p_task, a half-space of config space on 1 dim (prob p_1d)
       or 2 dims; its size is a fraction ~ U(frac) of uniform configs. Configs inside never learn: their loss stays
       at the task's initial loss (1 - y0) plus white noise with a per-task sd ~ log-uniform(flat_sd) (exactly
       constant like a diverged run clipped at the ceiling, up to visibly noisy). Motivation: 3-44% flat curves on every real
       source vs none in ifBO, and flat configs are predictable from the config (CV AUC 0.79-0.99,
       scripts/flat_clustering.py), i.e. a region, not luck.
noise  (V1) per-task noise: sd multiplier ~ log-uniform(scale), smoothing width (points) 0 with probability
       p_white, else ~ U(smooth). Contains both ifBO (1, 0) and rogue-dagger's noise fix (0.25, 1).
"""
from __future__ import annotations

import math
import sys

import torch

from curvefaker.measures import ROGUE

if str(ROGUE) not in sys.path:
    sys.path.insert(0, str(ROGUE))
from roguedagger.prior import PriorTasks, comb  # noqa: E402

SMOOTH_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])   # smoothing widths are snapped to this grid


def make_faker(dead: dict | None = None, noise: dict | None = None, name: str = "faker"):
    dead = None if dead is None else {"p_task": 0.7, "frac": (0.02, 0.5), "p_1d": 0.6, "flat_sd": (1e-4, 1e-2), **dead}
    noise = None if noise is None else {"scale": (0.25, 1.0), "p_white": 0.5, "smooth": (0.5, 2.0), **noise}

    class Faker(PriorTasks):
        components = {"dead": dead, "noise": noise}

        def __init__(self, n_tasks, d_max, device, generator=None):
            super().__init__(n_tasks, d_max, device, generator)
            E = n_tasks
            self.dead_on = torch.zeros(E, dtype=torch.bool, device=self.device)
            self.dead_w = torch.zeros(E, d_max, device=self.device)
            self.dead_c = torch.zeros(E, device=self.device)
            self.row_noise = torch.ones(E, device=self.device)
            self.row_smooth = torch.zeros(E, device=self.device)
            self.dead_sd = torch.zeros(E, device=self.device)
            # components draw from their own stream, so the ifBO draws stay identical across fakers
            self.cgen = torch.Generator(device=self.device)
            base = generator.initial_seed() if generator is not None else torch.initial_seed()
            self.cgen.manual_seed((base + 7919) % (2 ** 63))

        def _cr(self, *shape):
            return torch.rand(*shape, device=self.device, generator=self.cgen)

        def _crn(self, *shape):
            return torch.randn(*shape, device=self.device, generator=self.cgen)

        def resample(self, idx, d):
            super().resample(idx, d)          # the ifBO draws come first and are unchanged
            n = idx.numel()
            if n == 0:
                return
            if dead is not None:
                D = self.d_max
                on = self._cr(n) < dead["p_task"]
                frac = dead["frac"][0] + (dead["frac"][1] - dead["frac"][0]) * self._cr(n)
                # direction: 1 or 2 of the task's active dims, random signs / weights
                active = (torch.arange(D, device=self.device)[None] < d[:, None]).float()
                score = self._cr(n, D) * active - (1 - active)            # pick dims among active ones
                order = score.argsort(-1, descending=True)
                k2 = (self._cr(n) >= dead["p_1d"]) & (d >= 2)
                w = torch.zeros(n, D, device=self.device)
                r = torch.arange(n, device=self.device)
                sign = torch.where(self._cr(n) < 0.5, -1.0, 1.0)
                w[r, order[:, 0]] = sign
                w2 = self._crn(n)
                w[r[k2], order[k2, 1]] = w2[k2]
                u = self._cr(n, 1024, D)
                s = (u * w[:, None]).sum(-1)                               # [n, 1024]
                kq = ((1 - frac).clamp(0, 1) * (s.shape[1] - 1)).round().long()
                c = s.sort(1).values[r, kq]                                  # per-row (1 - frac) quantile
                self.dead_on[idx], self.dead_w[idx], self.dead_c[idx] = on, w, c
                lo, hi = math.log(dead["flat_sd"][0]), math.log(dead["flat_sd"][1])
                self.dead_sd[idx] = torch.exp(lo + (hi - lo) * self._cr(n))
            if noise is not None:
                lo, hi = math.log(noise["scale"][0]), math.log(noise["scale"][1])
                self.row_noise[idx] = torch.exp(lo + (hi - lo) * self._cr(n))
                sm = noise["smooth"][0] + (noise["smooth"][1] - noise["smooth"][0]) * self._cr(n)
                sm = torch.where(self._cr(n) < noise["p_white"], torch.zeros_like(sm), sm)
                grid = SMOOTH_GRID.to(self.device)
                self.row_smooth[idx] = grid[(sm[:, None] - grid[None]).abs().argmin(-1)]

        @torch.no_grad()
        def curves(self, x, steps_per_run, t_max, idx=None):
            if noise is None:
                loss = super().curves(x, steps_per_run, t_max, idx)
            else:
                loss = self._curves_rownoise(x, steps_per_run, t_max, idx)
            if dead is not None:
                g = (lambda t: t) if idx is None else (lambda t: t[idx])
                s = (x * g(self.dead_w)[:, None, :]).sum(-1)                 # [E', M]
                is_dead = g(self.dead_on)[:, None] & (s > g(self.dead_c)[:, None])
                flat = (1.0 - g(self.y0))[:, None, None] + g(self.dead_sd)[:, None, None] * self._crn(*loss.shape)
                loss = torch.where(is_dead[..., None], flat.clamp(0.0, 1.0), loss)
            return loss

        def _curves_rownoise(self, x, steps_per_run, t_max, idx):
            """PriorTasks.curves with a per-row noise multiplier and smoothing width."""
            g = (lambda t: t) if idx is None else (lambda t: t[idx])
            p = self.curve_params(x, idx)
            k = torch.arange(1, t_max + 1, device=self.device, dtype=torch.float32)
            t = (k[None] / steps_per_run.float()[:, None])[:, None, :]
            y = comb(t.expand(*x.shape[:2], t_max), p)
            z = self._randn(*y.shape)
            sm = g(self.row_smooth)
            for wv in sm.unique().tolist():
                if wv <= 0:
                    continue
                rows = sm == wv
                r = int(3 * wv)
                kern = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=self.device, dtype=torch.float32) / wv) ** 2)
                kern = kern / kern.norm()
                zz = z[rows]
                z[rows] = torch.nn.functional.conv1d(zz.reshape(-1, 1, t_max), kern[None, None],
                                                     padding=r).reshape(zz.shape)
            y = y + z * (g(self.row_noise)[:, None] * p["sigma"].float())[..., None]
            return 1.0 - y.clamp(0.0, 1.0)

    Faker.__name__ = Faker.__qualname__ = name
    return Faker


FAKERS = {   # named variants, used by scripts/train_rd.py and scripts/m1.py
    "ifbo": dict(),
    "noisefix": dict(noise=dict(scale=(0.25, 0.25), p_white=0.0, smooth=(1.0, 1.0))),   # rogue-dagger's fix
    "dead": dict(dead={}),
    "noisewide": dict(noise={}),
}


def get(name):
    return make_faker(name=name, **FAKERS[name])

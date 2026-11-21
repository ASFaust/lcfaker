"""bpl-v17: the bpl prior batched over tasks on the GPU, with configuration maps as functions. The main way to use
lcfaker.

bpl-v16 (lcfaker.generator, numpy, fixed configuration samples) with the changes needed for a batched,
function-based prior (release/consistency.md compares the two on the paper's statistics):

  * GP fields -> random Fourier features (RFF) of the same kernel family: Matern-5/2 per dimension with random
    relevances, a random mix of an additive and a product kernel, and a rough companion (length-scales x 0.15) for
    the floor-offset field. A task's fields are functions of x, so the same x in the same task always gives the
    same noise-free curve, for any set of configurations and any curve length. Fields are standardised with their
    mean / sd on 512 reference configurations drawn at resample time (generator.py standardises over the task's own
    sample).
  * Class shares (dead, diverged, rising) become thresholds on the class scores, set on the same reference
    configurations, so that the expected shares are the record's.
  * Per-configuration random choices of generator.py (divergence time, rise onset / size, the regularisation
    stand-in for d = 1) use high-frequency RFF "hash" fields: deterministic in x, nearly independent between
    configurations.
  * Optimiser progress S1 (cumulative learning-rate multiplier of the schedule) is computed in closed form, so any
    curve length T works. The search-space dimension d (1..d_max) is chosen by the caller for every task, with
    any record; the landscape roughness is calibrated for that d.

Only observation noise (Student-t, sd ~ excess^k, record's time profile) is drawn anew on each `curves` call.

Dimensions: x[..., 0] is the log learning rate (over the record's range of live configurations); for TaskSet
records x[..., 1:] start with the schedule hyperparameters of the record's optimiser family (adam6p / adam8p:
linear decay, exponential decay; nadamw: warm-up fraction, constant fraction, final learning-rate multiplier);
the remaining dimensions are generic (they act only through the fields). `dims(idx)` returns the names.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

ASSET = Path(__file__).resolve().parent / "assets" / "bpl_records.npz"
VERSION = "bpl-v17"
T_TOTAL = 10000
N_FIELDS = 7          # 0 floor offset (with rough part), 1-3 copula scores, 4 dead, 5 diverged, 6 rising
N_HASH = 3            # 0 rise onset / size choice, 1 divergence time, 2 regularisation stand-in (d = 1)
ROUGH_SCALE = 0.15
LR_AXIS_WEIGHT = 0.7
N_REF = 512
Z_MAX, N_TGRID = 8.0, 1025
SCHED_NAMES = {1: ["linear_decay", "exponential_decay"],
               2: ["warmup_fraction", "constant_fraction", "min_learning_rate_mult"]}


class BplTasks:
    """``n_tasks`` independent bpl tasks with hp dims up to ``d_max`` (rows are filled by ``resample``).

    source_weights: optional {source: weight} over "taskset", "jahs", "quicktune", "nasbench201"; default as the
    paper (in proportion to the number of real fit tasks per source, 96% TaskSet).
    compile: fuse `curves` with torch.compile (recommended on GPU: 256 tasks x 64 configs x 300 points in ~1 ms
    instead of ~7 ms; the first call compiles for ~10-60 s, new shapes may recompile once).
    """

    def __init__(self, n_tasks: int, d_max: int = 10, device: torch.device | str = "cuda",
                 generator: torch.Generator | None = None, source_weights: dict | None = None,
                 n_features: int = 256, n_hash_features: int = 64, compile: bool = False):
        self.n, self.d_max, self.device, self.gen = n_tasks, d_max, torch.device(device), generator
        self.M, self.Mh = n_features, n_hash_features
        self._points = torch.compile(_points, dynamic=True) if compile else _points
        if compile:
            self.configs_info = torch.compile(self.configs_info)
        A = dict(np.load(ASSET, allow_pickle=False))
        self.sources = A["source"]
        w = A["weight"].astype(float).copy()
        if source_weights is not None:
            unknown = set(source_weights) - set(np.unique(self.sources))
            if unknown:
                raise ValueError(f"unknown sources {unknown}")
            w2 = np.zeros_like(w)
            for s, v in source_weights.items():
                m = (self.sources == s) & (w > 0)
                w2[m] = v / m.sum()
            w = w2
        self.opt_names = A["opt"]
        self._build_tables(A, w)
        E, D, M, Mh = n_tasks, d_max, self.M, self.Mh
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.rec = torch.zeros(E, dtype=torch.long, device=self.device)
        self.d = torch.ones(E, dtype=torch.long, device=self.device)
        # field features in float64: phases of high-frequency features need it for "same x -> same curve" to hold
        # to float precision whatever other configs are evaluated with it
        z64 = lambda *s: torch.zeros(*s, device=self.device, dtype=torch.float64)
        self.Om, self.bm, self.Wm = z64(E, 2 * M, D), z64(E, 2 * M), z64(E, 2 * M, N_FIELDS)     # smooth fields
        self.Or, self.br, self.Wr = z64(E, 2 * M, D), z64(E, 2 * M), z64(E, 2 * M, 1)            # rough part, field 0
        self.Oh, self.bh, self.Wh = z64(E, Mh, D), z64(E, Mh), z64(E, Mh, N_HASH)                 # hash fields
        self.fmu, self.fsd = z(E, N_FIELDS + 1), torch.ones(E, N_FIELDS + 1, device=self.device)
        self.hmu, self.hsd = z(E, N_HASH), torch.ones(E, N_HASH, device=self.device)
        self.thr_dead, self.thr_div, self.thr_over = z(E), z(E), z(E)

    # ------------------------------------------------------------------------------------------------ tables
    def _build_tables(self, A, w):
        dev, f32 = self.device, torch.float32
        t = lambda a, dt=f32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.weight = t(w / w.sum())
        R = len(w)
        for k in ("L", "y0", "kappa", "eta_ref", "share_dead", "share_div", "share_over", "noise_sd", "noise_k",
                  "noise_med_ex", "df", "dead_level", "start_sd", "has_start", "T"):
            setattr(self, "r_" + k, t(A[k]))
        self.r_opt = t(A["opt_id"], torch.long)
        self.r_lr = t(A["lr_q_live"])
        self.r_thirds = t(A["noise_thirds"])
        self.r_sched = t(A["sched_q"])                       # [R, 3, 101]
        self.r_eta_d = t(A["eta_d"])                         # [R, 11]
        err = A["kind"] == 1
        clip = np.where(err, np.where(np.isin(A["source"], ["jahs", "quicktune"]), 100.0, 1.0), np.inf)
        self.r_clip = t(clip)
        C = A["copula"]
        c0 = C[:, 1:, 0]
        Lc = np.linalg.cholesky(C[:, 1:, 1:] - c0[:, :, None] * c0[:, None, :] + 1e-6 * np.eye(3))
        self.r_c0, self.r_Lc = t(c0), t(Lc)
        self.r_lrrel = t(A["lr_rel"])
        # Student-t draws as a transform of a normal: t = F_t^-1(Phi(z)), tabulated on a z grid (exact up to
        # linear interpolation; |z| > 8 has probability ~1e-15). One table per record (its dof) plus dof 5.
        from scipy.stats import norm as _norm, t as _t
        zg = np.linspace(-Z_MAX, Z_MAX, N_TGRID)
        u = _norm.cdf(zg)
        self.r_ttab = t(_t.ppf(u[None, :], A["df"][:, None]))                      # [R, N_TGRID]
        self.t5tab = t(_t.ppf(u, 5.0))
        self.r_mids, self.r_mb, self.r_r2 = t(A["lr_mids"]), t(A["lr_mb"]), t(A["lr_r2"])

        def padded(key, cols=None, sort=True):
            offs = A[key + "_offs"]
            cnt = np.diff(offs)
            K = max(int(cnt.max()), 1)
            if cols is None:
                out = np.zeros((R, K))
                for i in range(R):
                    v = A[key][offs[i]:offs[i + 1]]
                    out[i, :len(v)] = np.sort(v) if sort else v
            else:
                out = np.zeros((R, cols, K))
                for i in range(R):
                    v = A[key][offs[i]:offs[i + 1]]
                    out[i, :, :len(v)] = np.sort(v, axis=0).T
            return t(out), t(cnt, torch.long)

        self.r_P, self.r_Pn = padded("params", cols=4)       # sorted marginals [R, 4, K]: exact np.quantile
        self.r_div, self.r_divn = padded("div_times")
        self.r_on, self.r_onn = padded("over_onset", sort=False)   # onset and rise stay paired
        self.r_ri, _ = padded("over_rise", sort=False)

    # ------------------------------------------------------------------------------------------ random helpers
    def _rand(self, *s):
        return torch.rand(*s, device=self.device, generator=self.gen)

    def _randn(self, *s):
        return torch.randn(*s, device=self.device, generator=self.gen)

    def _gamma(self, a):
        """Gamma(a, 1) for a tensor of shapes a (Marsaglia-Tsang; a < 1 via the boost a + 1, U^(1/a))."""
        boost = a < 1
        aa = torch.where(boost, a + 1, a)
        d = aa - 1 / 3
        c = 1 / torch.sqrt(9 * d)
        out = d.clone()
        done = torch.zeros_like(a, dtype=torch.bool)
        for _ in range(10):
            z = self._randn(*a.shape)
            v = (1 + c * z) ** 3
            u = self._rand(*a.shape)
            ok = (v > 0) & (torch.log(u) < 0.5 * z * z + d - d * v + d * torch.log(v.clamp_min(1e-30))) & ~done
            out = torch.where(ok, d * v, out)
            done |= ok
            if bool(done.all()):
                break
        return torch.where(boost, out * self._rand(*a.shape) ** (1 / a), out)

    @staticmethod
    def _t_from_normal(z, tab):
        """Student-t values for standard normal draws z [E, ...], tables tab [E, N_TGRID] (or [N_TGRID])."""
        pos = ((z.clamp(-Z_MAX, Z_MAX) + Z_MAX) * ((N_TGRID - 1) / (2 * Z_MAX)))
        i0 = pos.floor().clamp(max=N_TGRID - 2)
        fr = pos - i0
        i0 = i0.long()
        if tab.dim() == 1:
            return tab[i0] * (1 - fr) + tab[i0 + 1] * fr
        flat = i0.reshape(z.shape[0], -1)
        lo = tab.gather(1, flat).reshape(z.shape)
        hi = tab.gather(1, flat + 1).reshape(z.shape)
        return lo * (1 - fr) + hi * fr

    # ---------------------------------------------------------------------------------------------- resample
    @torch.no_grad()
    def resample(self, idx: torch.Tensor, d: torch.Tensor, record: torch.Tensor | None = None) -> None:
        """Draw fresh tasks for rows ``idx`` with hp dims ``d`` (1..d_max, same length as idx).

        record: optional record ids (default: drawn by the source weights)."""
        n = idx.numel()
        if n == 0:
            return
        idx, d = idx.to(self.device), d.to(self.device).long().clamp(1, self.d_max)
        D, M, Mh = self.d_max, self.M, self.Mh
        rec = record.to(self.device).long() if record is not None else \
            torch.multinomial(self.weight, n, replacement=True, generator=self.gen)
        self.rec[idx], self.d[idx] = rec, d
        inp = (torch.arange(D, device=self.device)[None] < d[:, None]).float()        # [n, D]
        # kernel: relevances r ~ Dirichlet(0.7) over the d dims, length-scales, additive / product mix w
        g = self._gamma(torch.full((n, D), 0.7, device=self.device)) * inp
        r = g / g.sum(1, keepdim=True)
        ell = torch.exp(math.log(0.2) * (1 - self._rand(n)))[:, None] / torch.sqrt(
            (r / r.max(1, keepdim=True).values).clamp_min(1e-12))                           # [n, D]
        wmix = self._rand(n)

        def t5(*s):   # Student-t with 5 dof: the spectral density of a 1-D Matern-5/2 kernel (unit length-scale)
            return self._t_from_normal(self._randn(*s), self.t5tab)

        def features(scale):
            # product kernel: one frequency per dim; additive kernel: one dim per feature, chosen with prob r_j
            Op = t5(n, M, D) / (ell[:, None, :] * scale) * inp[:, None, :]
            dim = torch.multinomial(r, M, replacement=True, generator=self.gen)        # [n, M]
            Oa = torch.zeros(n, M, D, device=self.device)
            Oa.scatter_(2, dim[..., None], (t5(n, M, 1)[..., 0] / (ell.gather(1, dim) * scale))[..., None])
            O = torch.cat([Oa, Op], 1)                                                       # [n, 2M, D]
            b = 2 * math.pi * self._rand(n, 2 * M)
            return O, b

        O, b = features(1.0)
        self.Om[idx], self.bm[idx] = O.double(), b.double()
        O, b = features(ROUGH_SCALE)
        self.Or[idx], self.br[idx] = O.double(), b.double()
        amp = torch.cat([torch.sqrt(wmix)[:, None].expand(n, M), torch.sqrt(1 - wmix)[:, None].expand(n, M)], 1)
        amp = amp * math.sqrt(2 / M)
        self.Wm[idx] = (self._randn(n, 2 * M, N_FIELDS) * amp[..., None]).double()
        self.Wr[idx] = (self._randn(n, 2 * M, 1) * amp[..., None]).double()
        self.Oh[idx] = (self._randn(n, Mh, D) * 300.0 * inp[:, None, :]).double()
        self.bh[idx] = (2 * math.pi * self._rand(n, Mh)).double()
        self.Wh[idx] = (self._randn(n, Mh, N_HASH) * math.sqrt(2 / Mh)).double()
        # standardise fields and set class thresholds on reference configurations
        self.fmu[idx], self.fsd[idx] = 0.0, 1.0
        self.hmu[idx], self.hsd[idx] = 0.0, 1.0
        xr = self._rand(n, N_REF, D) * inp[:, None, :]
        raw, hraw = self._raw_fields(xr, idx, precise=False)
        self.fmu[idx], self.fsd[idx] = raw.mean(1), raw.std(1).clamp_min(1e-6)
        self.hmu[idx], self.hsd[idx] = hraw.mean(1), hraw.std(1).clamp_min(1e-6)
        gf, h = self._fields(xr, idx, precise=False)
        ds, vs, os_ = self._scores(xr, gf, h, idx)
        k = lambda share: (share * N_REF).round().long()
        nd = k(self.r_share_dead[rec])
        srt = ds.sort(1, descending=True).values
        self.thr_dead[idx] = torch.where(nd > 0, srt.gather(1, (nd - 1).clamp_min(0)[:, None])[:, 0],
                                         torch.full_like(nd, 1e9, dtype=torch.float32))
        dead = ds >= self.thr_dead[idx][:, None]
        nv = k(self.r_share_div[rec])
        srt = torch.where(dead, torch.full_like(vs, -1e9), vs).sort(1, descending=True).values
        self.thr_div[idx] = torch.where(nv > 0, srt.gather(1, (nv - 1).clamp_min(0)[:, None])[:, 0],
                                        torch.full_like(nv, 1e9, dtype=torch.float32))
        no = torch.where(self.r_onn[rec] > 0, k(self.r_share_over[rec]), torch.zeros_like(nd))
        srt = os_.sort(1, descending=True).values
        self.thr_over[idx] = torch.where(no > 0, srt.gather(1, (no - 1).clamp_min(0)[:, None])[:, 0],
                                         torch.full_like(no, 1e9, dtype=torch.float32))

    # ------------------------------------------------------------------------------------------------ fields
    def _raw_fields(self, x, idx, precise=True):
        """precise: phase in float64 (reduced mod 2 pi), cosine in float32, weighted sum in float64, so a config's
        fields do not depend on which other configs are evaluated with it. Not needed for the reference configs
        of `resample` (field statistics and class thresholds only)."""
        if not precise:
            def ev(O, b, W):
                return torch.cos(torch.bmm(x, O[idx].float().transpose(1, 2)) + b[idx].float()[:, None]) @ \
                    W[idx].float()
        else:
            x = x.double()

            def ev(O, b, W):
                ph = torch.remainder(torch.bmm(x, O[idx].transpose(1, 2)) + b[idx][:, None], 2 * math.pi)
                return (torch.cos(ph.float()).double() @ W[idx]).float()
        sm = ev(self.Om, self.bm, self.Wm)
        ro = ev(self.Or, self.br, self.Wr)
        hs = ev(self.Oh, self.bh, self.Wh)
        return torch.cat([sm, ro], -1), hs

    def _fields(self, x, idx, precise=True):
        """Standardised fields g [E', N, 7] (field 0 mixed with its rough part) and hash uniforms h [E', N, 3]."""
        raw, hraw = self._raw_fields(x, idx, precise)
        f = (raw - self.fmu[idx][:, None]) / self.fsd[idx][:, None]
        eta = self.r_eta_d[self.rec[idx], self.d[idx]][:, None]
        g = f[..., :N_FIELDS].clone()
        g[..., 0] = torch.sqrt(1 - eta) * f[..., 0] + torch.sqrt(eta) * f[..., N_FIELDS]
        h = torch.special.ndtr((hraw - self.hmu[idx][:, None]) / self.hsd[idx][:, None])
        return g, h

    def _scores(self, x, g, h, idx):
        ax = (x[..., 0] - 0.5) / math.sqrt(1 / 12)
        dead_s = LR_AXIS_WEIGHT * (-ax) + (1 - LR_AXIS_WEIGHT) * g[..., 4]
        div_s = LR_AXIS_WEIGHT * ax + (1 - LR_AXIS_WEIGHT) * g[..., 5]
        reg = torch.where((self.d[idx] > 1)[:, None], x[..., 1 % self.d_max], h[..., 2])
        over_s = 0.5 * g[..., 6] + 0.5 * (reg - 0.5) / math.sqrt(1 / 12)
        return dead_s, div_s, over_s

    # ------------------------------------------------------------------------------------- progress (S1 / T)
    def _sched_hps(self, x, rec, d):
        """Schedule hyperparameters of every config [E', N] x 3: interpolated in the record's quantile tables at
        x[..., 1 + j] (the record's median if the task has fewer dims). Unused entries are harmless."""
        sq = self.r_sched[rec]                                              # [E', 3, 101]
        out = []
        for j in range(3):
            col = 1 + j
            have = (col < d)[:, None]
            u = x[..., col] if col < self.d_max else torch.zeros_like(x[..., 0])
            pos = torch.where(have, u, torch.full_like(u, 0.5)) * 100
            i0 = pos.floor().long().clamp(0, 99)
            fr = pos - i0
            tab = sq[:, j]
            out.append(tab.gather(1, i0) * (1 - fr) + tab.gather(1, i0 + 1) * fr)
        return out

    # ------------------------------------------------------------------------------------------------ curves
    @staticmethod
    def _interp(v, xp, fp):
        """Batched np.interp (clamped ends): v [E, N], xp [E, K] increasing, fp [E, K]."""
        K = xp.shape[1]
        i = torch.searchsorted(xp.contiguous(), v.contiguous()).clamp(1, K - 1)
        x0, x1 = xp.gather(1, i - 1), xp.gather(1, i)
        y0, y1 = fp.gather(1, i - 1), fp.gather(1, i)
        w = ((v - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
        return y0 + w * (y1 - y0)

    @staticmethod
    def _quantile(sorted_, cnt, q):
        """np.quantile (linear) of padded sorted samples [E, K] with counts [E] at q [E, N]."""
        pos = q * (cnt[:, None] - 1).clamp_min(0).float()
        i0 = pos.floor().long()
        i1 = (i0 + 1).clamp(max=(cnt[:, None] - 1).clamp_min(0))
        fr = pos - i0
        return sorted_.gather(1, i0) * (1 - fr) + sorted_.gather(1, i1) * fr

    @torch.no_grad()
    def configs_info(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> dict:
        """Per-config quantities (noise-free): curve parameters, learning rate, classes. x [E', N, d_max]."""
        idx = torch.arange(self.n, device=self.device) if idx is None else idx.to(self.device)
        rec, d = self.rec[idx], self.d[idx]
        inp = (torch.arange(self.d_max, device=self.device)[None] < d[:, None]).float()
        x = x.to(self.device).float() * inp[:, None, :]
        g, h = self._fields(x, idx)
        lo, hi = self.r_lr[rec, 0][:, None], self.r_lr[rec, 1][:, None]
        log_eta = lo + x[..., 0] * (hi - lo)
        c0, Lc = self.r_c0[rec], self.r_Lc[rec]
        zsc = torch.empty(*g.shape[:2], 4, device=self.device)
        zsc[..., 0] = g[..., 0]
        zsc[..., 1:] = g[..., :1] * c0[:, None, :] + torch.bmm(g[..., 1:4], Lc.transpose(1, 2))
        rel = self.r_lrrel[rec] > 0
        for j in range(4):
            shift = self._interp(log_eta, self.r_mids[rec], self.r_mb[rec, j])
            zr = shift + torch.sqrt(1 - self.r_r2[rec, j])[:, None] * zsc[..., j]
            zsc[..., j] = torch.where(rel[:, None], zr, zsc[..., j])
        q = torch.special.ndtr(zsc)
        par = torch.stack([self._quantile(self.r_P[rec, j], self.r_Pn[rec], q[..., j]) for j in range(4)], -1)
        kappa = self.r_kappa[rec][:, None]
        lrfac = torch.where(kappa > 0, torch.exp(kappa * (log_eta - torch.log(self.r_eta_ref[rec])[:, None])),
                            torch.ones_like(log_eta))
        L, y0 = self.r_L[rec][:, None], self.r_y0[rec][:, None]
        start = self.r_has_start[rec][:, None] > 0
        f0 = (1 / self.r_T[rec]).view(-1)
        h0, h1, h2 = self._sched_hps(x, rec, d)
        oid = self.r_opt[rec][:, None, None]
        s0 = (f0 * T_TOTAL + 1.0)[:, None, None]
        x0 = lrfac * _progress(s0, oid, h0[..., None], h1[..., None], h2[..., None])[..., 0]   # first grid point
        f1 = 1 / (1 + (x0 / torch.exp(par[..., 2])) ** torch.exp(par[..., 3]))
        capA = torch.log(0.999 * (y0 - L) / f1.clamp_min(1e-12))
        par[..., 1] = torch.where(start, torch.minimum(par[..., 1], capA), par[..., 1])
        dmax = torch.maximum(y0 - L - torch.exp(par[..., 1]) * f1, 1e-4 * (y0 - L))
        par[..., 0] = torch.where(start, torch.minimum(par[..., 0], torch.log(dmax)), par[..., 0])
        ds, vs, os_ = self._scores(x, g, h, idx)
        dead = ds >= self.thr_dead[idx][:, None]
        div = (vs >= self.thr_div[idx][:, None]) & ~dead
        over = os_ >= self.thr_over[idx][:, None]
        j = (h[..., 0] * self.r_onn[rec][:, None]).floor().long().clamp(0, None)
        j = torch.minimum(j, (self.r_onn[rec] - 1).clamp_min(0)[:, None])
        onset, rise = self.r_on[rec].gather(1, j), self.r_ri[rec].gather(1, j)
        t_div = self._quantile(self.r_div[rec], self.r_divn[rec], h[..., 1])
        t_div = torch.where((self.r_divn[rec] > 0)[:, None], t_div, h[..., 1])
        return dict(x=x, par=par, log_lr=log_eta, lrfac=lrfac, dead=dead, diverged=div, rising=over,
                    onset=onset, rise=rise, t_div=t_div, rec=rec, d=d, sched=(h0, h1, h2))

    @torch.no_grad()
    def curves(self, x: torch.Tensor, T: torch.Tensor, L: int, idx: torch.Tensor | None = None,
               noise: bool = True, scale: str = "raw") -> torch.Tensor:
        """Validation curves [E', N, L] for configs x [E', N, d_max] in [0, 1] of task rows idx (all if None).

        Point k (1-based) is at budget fraction k / T[row]: a full training has T[row] points (any T >= 1).
        Points past T are the value at T (the run has finished; the benchmarks cap there too).
        scale: "raw" (the record's units: a loss, or an error rate in % / fraction) or "unit":
        (y - L) / (y0 - L) with the task's floor L and typical start y0, clipped to [0, 2] (diverged = 2).
        """
        idx = torch.arange(self.n, device=self.device) if idx is None else idx.to(self.device)
        c = self.configs_info(x, idx)
        rec, par = c["rec"], c["par"]
        k = torch.arange(1, L + 1, device=self.device, dtype=torch.float32)
        Tf = T.to(self.device).float().clamp_min(1)
        frac = (k[None] / Tf[:, None]).clamp(max=1.0)                         # [E', L]
        z = self._randn(par.shape[0], par.shape[1], L) if noise else None
        e1 = lambda v: v[:, None, None]
        n1 = lambda v: v[..., None]
        h0, h1, h2 = c["sched"]
        y = self._points(frac[:, None, :], k[None, None, :], e1(Tf), e1(self.r_opt[rec]), n1(h0), n1(h1), n1(h2),
                         n1(c["lrfac"]), n1(par[..., 0]), n1(par[..., 1]), n1(par[..., 2]), n1(par[..., 3]),
                         e1(self.r_L[rec]), e1(self.r_y0[rec]), n1(c["rising"]), n1(c["onset"]), n1(c["rise"]),
                         self.r_thirds[rec][:, None, None, :], e1(self.r_noise_sd[rec]), e1(self.r_noise_med_ex[rec]),
                         e1(self.r_noise_k[rec]), e1(self.r_dead_level[rec]), n1(c["dead"]), n1(c["diverged"]),
                         n1(c["t_div"]), e1(self.r_clip[rec]), z, self.r_ttab[rec])
        if scale == "unit":
            Lr, y0 = e1(self.r_L[rec]), e1(self.r_y0[rec])
            y = ((y - Lr) / (y0 - Lr)).clamp(0.0, 2.0)
        return y

    @torch.no_grad()
    def start_values(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """TaskSet-like tasks: the value at step 0 (before training), near the shared initial loss; NaN for other
        tasks. As generator.py prepends it."""
        idx = torch.arange(self.n, device=self.device) if idx is None else idx.to(self.device)
        c = self.configs_info(x, idx)
        rec = c["rec"]
        Lr, y0 = self.r_L[rec][:, None], self.r_y0[rec][:, None]
        st = Lr + (y0 - Lr) * torch.exp(self.r_start_sd[rec][:, None] * self._randn(*c["dead"].shape))
        st = torch.where(c["diverged"], torch.maximum(st, y0), st)
        return torch.where((self.r_has_start[rec] > 0)[:, None], st, torch.full_like(st, float("nan")))

    def metric(self, idx: torch.Tensor | None = None) -> list[str]:
        """"loss" or "error" per task row."""
        idx = torch.arange(self.n, device=self.device) if idx is None else idx.to(self.device)
        return ["error" if math.isfinite(v) else "loss" for v in self.r_clip[self.rec[idx]].tolist()]

    def dims(self, row: int) -> list[str]:
        """Names of the hp dimensions of task row `row`."""
        d, r = int(self.d[row]), int(self.rec[row])
        oid = int(self.r_opt[r])
        names = ["log_learning_rate"] + SCHED_NAMES.get(oid, [])
        names = names + [f"generic_{j}" for j in range(len(names), d)]
        return names[:d]


# ------------------------------------------------------------------------------------- per-point arithmetic
def _progress(s, oid, h0, h1, h2):
    """Optimiser progress S1 / T_TOTAL at steps s for schedule family oid (0 constant, 1 linear / exponential
    decay with h0, h1, 2 warm-up / constant / cosine with h0, h1, h2 = warm-up fraction, constant fraction, final
    multiplier, 3 cosine to 0). Closed forms of the integral of the learning-rate multiplier; broadcasting."""
    # 1: (1 - l u) e^(-e u), until 1 - l u = 0
    l, e = h0, h1
    se = torch.where(l > 0, torch.minimum(s, 1 / l.clamp_min(1e-12)), s)
    es = e * se
    small = es < 0.1
    ec = e.clamp_min(1e-12)
    I0 = torch.where(small, se * (1 - es / 2 + es ** 2 / 6 - es ** 3 / 24), -torch.expm1(-es) / ec)
    I1 = torch.where(small, se ** 2 * (0.5 - es / 3 + es ** 2 / 8 - es ** 3 / 30),
                     (1 - torch.exp(-es) * (1 + es)) / ec ** 2)
    v1 = I0 - l * I1
    # 2: linear warm-up to W, constant to C, cosine from C to the end (to the final multiplier lo)
    wf, cf, lo = h0, h1, h2
    C = cf * T_TOTAL
    W = torch.minimum(wf, cf) * T_TOTAL
    Tc = (T_TOTAL - C).clamp_min(1e-6)
    warm = s ** 2 / (2 * W.clamp_min(1e-6))
    const = W / 2 + (s - W)
    sc = (s - C).clamp_min(0)
    cos = W / 2 + (C - W) + lo * sc + (1 - lo) * (sc / 2 + Tc / (2 * math.pi) * torch.sin(math.pi * sc / Tc))
    v2 = torch.where(s <= W, warm, torch.where(s <= C, const, cos))
    # 3: cosine to 0 over the full budget
    s3 = s.clamp(max=T_TOTAL)
    v3 = s3 / 2 + T_TOTAL / (2 * math.pi) * torch.sin(math.pi * s3 / T_TOTAL)
    out = torch.where(oid == 1, v1, torch.where(oid == 2, v2, torch.where(oid == 3, v3, s)))
    return (out / T_TOTAL).clamp_min(1e-9)


def _points(frac, k, Tf, oid, h0, h1, h2, lrfac, p0, p1, p2, p3, Lr, y0, rising, onset, rise, thirds, noise_sd,
            med_ex, nk, dead_level, dead, diverged, t_div, clip, z, ttab):
    """All per-point work of `curves` (one fused kernel under torch.compile). Shapes broadcast to [E, N, L]."""
    s = frac * T_TOTAL + 1.0
    xp = _progress(s, oid, h0, h1, h2) * lrfac
    y = Lr + torch.exp(p0) + torch.exp(p1) / (1 + (xp / torch.exp(p2)) ** torch.exp(p3))
    clean_ex = y - Lr
    ramp = ((frac - onset) / (1 - onset).clamp_min(1e-3)).clamp_min(0)
    y = torch.where(rising, y + rise * (y0 - Lr) * ramp, y)
    t0, t1, t2 = thirds[..., 0], thirds[..., 1], thirds[..., 2]
    prof = torch.where(frac <= 1 / 6, t0, torch.where(frac <= 0.5, t0 + (frac - 1 / 6) * 3 * (t1 - t0),
                       torch.where(frac <= 5 / 6, t1 + (frac - 0.5) * 3 * (t2 - t1), t2)))
    sd0 = noise_sd * prof
    if z is not None:
        pos = (z.clamp(-Z_MAX, Z_MAX) + Z_MAX) * ((N_TGRID - 1) / (2 * Z_MAX))
        i0 = pos.floor().clamp(max=N_TGRID - 2)
        fr = pos - i0
        i0 = i0.long()
        E = z.shape[0]
        flat = i0.reshape(E, -1)
        tv = (ttab.gather(1, flat) * (1 - fr.reshape(E, -1)) + ttab.gather(1, flat + 1) * fr.reshape(E, -1))
        eps = tv.reshape(z.shape) * sd0
        y = y + eps * ((clean_ex.clamp_min(1e-12) / med_ex) ** nk).clamp(max=20.0)
        yd = dead_level + eps
    else:
        yd = dead_level + 0 * y
    y = torch.where(dead, yd, y)
    k0 = torch.round(t_div * (Tf - 1).clamp_min(1))
    after = torch.minimum(k - 1, Tf - 1) >= k0
    y = torch.where(diverged & after, y0 + 10 * (y0 - Lr) + 0 * y, y)
    return torch.where(torch.isfinite(clip), y.clamp_min(0.0).minimum(clip), y)

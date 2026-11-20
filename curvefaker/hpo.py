"""Measure 3 (README): existing HPO algorithms run on a task given as curves, real or generated.

A task here is y [n, T]: one seed's curves (lower is better; normalised z), n configs. Training config i for k
steps reveals y[i, :k]. The incumbent is the config with the lowest value observed so far (at any step). Its
*true* quality is its final value (mean of the last 10% of its curve), and regret is its rank:

    regret = share of the task's configs whose final value is better than the incumbent's

so 0 = found the best config, 0.5 = a typical config. (A first version divided by median - best final; it blew up
on tasks where most configs end at the best value.) Budgets are counted in steps, reported in
units of full trainings (T steps). Diverged values (z = cap) count as observed bad values.

Algorithms (no learning involved; parameters fixed, chosen before looking at any result):
    random     : random configs, each trained to the end
    hyperband  : Hyperband with eta = 3 and minimum budget about T / 27, brackets cycled until the budget is spent
    freezethaw : start 10 random configs for 1 step; then repeatedly either start a new random config (prob 0.2)
                 or continue the started config with the lowest last observed value by one step
"""
from __future__ import annotations

import math

import numpy as np

from curvefaker import curves

BUDGETS = (1, 2, 5, 10, 20)


def _finals(y):
    return curves.final_value(np.nan_to_num(y, nan=curves.CAP))


class Run:
    def __init__(self, y, rng):
        self.y, self.rng = np.nan_to_num(y, nan=curves.CAP), rng
        self.n, self.T = y.shape
        self.steps = np.zeros(self.n, int)
        self.spent = 0
        self.best_obs, self.inc = np.inf, -1
        self.trace = []                      # (budget spent, incumbent)

    def train(self, i, k):
        """Train config i up to k steps total (no-op if already there)."""
        k = min(k, self.T)
        new = k - self.steps[i]
        if new <= 0:
            return
        seg = self.y[i, self.steps[i]:k]
        self.steps[i] = k
        self.spent += new
        j = seg.argmin()
        if seg[j] < self.best_obs:
            self.best_obs, self.inc = seg[j], i
        self.trace.append((self.spent, self.inc))


def regret_curve(run: Run, finals: np.ndarray) -> np.ndarray:
    srt = np.sort(finals)
    out = []
    tr = run.trace
    for b in BUDGETS:
        limit = b * run.T
        inc = -1
        for spent, i in tr:
            if spent > limit:
                break
            inc = i
        out.append(1.0 if inc < 0 else np.searchsorted(srt, finals[inc], side="left") / len(srt))
    return np.array(out)


def random_search(y, rng):
    r = Run(y, rng)
    for i in rng.permutation(r.n):
        if r.spent >= BUDGETS[-1] * r.T:
            break
        r.train(i, r.T)
    return r


def hyperband(y, rng, eta=3):
    r = Run(y, rng)
    T = r.T
    s_max = max(0, int(math.floor(math.log(T) / math.log(eta) + 1e-9)) - 0)
    s_max = min(s_max, 3)
    unused = list(rng.permutation(r.n))
    while r.spent < BUDGETS[-1] * T and unused:
        for s in range(s_max, -1, -1):
            n = int(math.ceil((s_max + 1) / (s + 1) * eta ** s))
            b = T * eta ** (-s)
            cfg = [unused.pop() for _ in range(min(n, len(unused)))]
            for k in range(s + 1):
                steps = max(1, int(round(b * eta ** k)))
                for i in cfg:
                    r.train(i, steps)
                if k < s:
                    vals = np.array([r.y[i, :r.steps[i]].min() for i in cfg])
                    cfg = [cfg[j] for j in np.argsort(vals)[:max(1, int(len(cfg) / eta))]]
            if r.spent >= BUDGETS[-1] * T or not unused:
                break
    return r


def freeze_thaw(y, rng, n0=10, p_new=0.2):
    r = Run(y, rng)
    unused = list(rng.permutation(r.n))
    started = []
    for _ in range(min(n0, len(unused))):
        i = unused.pop()
        r.train(i, 1)
        started.append(i)
    while r.spent < BUDGETS[-1] * r.T:
        live = [i for i in started if r.steps[i] < r.T]
        if unused and (rng.random() < p_new or not live):
            i = unused.pop()
            r.train(i, 1)
            started.append(i)
            continue
        if not live:
            break
        last = np.array([r.y[i, r.steps[i] - 1] for i in live])
        i = live[int(last.argmin())]
        r.train(i, r.steps[i] + 1)
    return r


ALGOS = {"random": random_search, "hyperband": hyperband, "freezethaw": freeze_thaw}


def evaluate_task(y: np.ndarray, seeds=(0, 1, 2)) -> dict:
    """y [n, T] (or [n, R, T]: replica 0 is used). Mean regret per algorithm and budget over `seeds`."""
    if y.ndim == 3:
        y = y[:, 0]
    z, _ = curves.normalise(y[:, None])
    z = z[:, 0]
    fin = _finals(z)
    out = {}
    for name, alg in ALGOS.items():
        R = np.mean([regret_curve(alg(z, np.random.default_rng(s)), fin) for s in seeds], 0)
        for b, v in zip(BUDGETS, R):
            out[f"{name}@{b}"] = float(v)
    return out

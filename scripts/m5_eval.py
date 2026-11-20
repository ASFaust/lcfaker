"""M5 downstream evaluation on allowed tasks: run a rogue-dagger policy on the internal `select` split.

Same protocol as rogue-dagger's evaluate.py on the benchmarks (1000 steps, 10 seeds, normalised regret), on a
fixed sample of select tasks normalised by curvefaker.measures.curves (one replica per config, at most 1000 configs,
categoricals as ordinal codes so d stays <= 10 where possible; TabEnv keeps the first 10 dims).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/m5_eval.py \
        --ckpt ../rogue-dagger/runs/wide-focus03-v1/last.pt --tag wf03 [--n-shop 0 --temperature 0.25]
    PYTHONPATH=. python scripts/m5_eval.py --method random_search --tag rs

Writes runs/m5/<split>/<tag>-n<shop>-t<tau>.npz: regret [tasks, seeds, budget], tasks, n_runs.
"""
import argparse
import sys
import time

import numpy as np
import torch

from curvefaker import data
from curvefaker.measures import ROGUE, curves
from curvefaker.paths import REPO

sys.path.insert(0, str(ROGUE))
from roguedagger.bench import Task as RDTask, TabEnv  # noqa: E402
from roguedagger.train import policy_step, random_search_step  # noqa: E402

PER_SOURCE = {"taskset": 60, "fcnet": None, "bbomix": None, "nasbench201": None, "quicktune": 30, "jahs": None}


def task_list(split, seed=0):
    """Fixed sample: TaskSet stratified over families (round-robin over a seeded shuffle), others whole."""
    out = []
    for src, cap in PER_SOURCE.items():
        if src not in data.SOURCES:
            continue
        I = data.ids(split, [src])
        if cap is None or len(I) <= cap:
            out += I
            continue
        rng = np.random.default_rng(seed)
        by_fam = {}
        for i in rng.permutation(I):
            by_fam.setdefault(data.family_of(i), []).append(i)
        pick, k = [], 0
        while len(pick) < cap:
            for f in sorted(by_fam):
                if k < len(by_fam[f]) and len(pick) < cap:
                    pick.append(by_fam[f][k])
            k += 1
        out += sorted(pick)
    return out


def to_rd(ident):
    t = data.load(ident)
    Y, X = curves(t, rng=np.random.default_rng(0), cat="ordinal")
    keep = X.std(0) > 0
    X = X[:, keep] if keep.any() else X[:, :1]
    return RDTask(ident, t.source, X.astype(np.float64), Y.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="rd", choices=["rd", "random_search"])
    ap.add_argument("--ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="select")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--n-shop", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these sources (testing)")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only (testing)")
    a = ap.parse_args()
    ids = task_list(a.split)
    if a.sources:
        ids = [i for i in ids if i.split("/")[0] in a.sources]
    ids = ids[:a.limit]
    model, h = None, (1.0, 1.0)
    if a.method == "rd":
        from roguedagger.model import load_policy
        ck = torch.load(a.ckpt, map_location=a.device)
        model = load_policy(ck, a.device)
        h = (ck["cfg"]["train"]["h_start"], ck["cfg"]["train"]["h_end"])
    regret = np.zeros((len(ids), a.seeds, a.budget), np.float32)
    n_runs = np.zeros((len(ids), a.seeds))
    t_all = time.time()
    for i, ident in enumerate(ids):
        t0 = time.time()
        task = to_rd(ident)
        env = TabEnv(task, a.seeds, a.budget, n_shop=(a.n_shop or None), device=a.device, seed=0)
        with torch.no_grad():
            while not env.done.all():
                if a.method == "rd":
                    act = policy_step(model, env, a.temperature, False, h=h, amp=a.device.startswith("cuda"))
                else:
                    act = random_search_step(env)
                env.step(act)
        regret[i] = env.normalised_regret().cpu().numpy()
        n_runs[i] = env.n_runs.cpu().numpy()
        print(f"[{i + 1}/{len(ids)}] {ident}: regret {regret[i, :, -1].mean():.4f} runs {n_runs[i].mean():.0f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = REPO / "runs" / "m5" / a.split / f"{a.tag}-n{a.n_shop}-t{a.temperature}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, regret=regret, tasks=np.array(ids), n_runs=n_runs, args=str(vars(a)))
    print(f"saved {out}; mean final regret {regret[:, :, -1].mean():.4f} ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()

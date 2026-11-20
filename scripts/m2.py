"""M2: HPO-run statistics, real vs fake (README "Quality measures").

Several HPO methods run on real tasks (internal split, default `select`) and on fake twins (same n, d, T) from each
faker. Per (task, method): final regret, configs started, fidelity reached per run, how often the incumbent
changes, regret at 25 / 50 % of the budget. Then per faker:
  * KS distance real vs fake for each (method, statistic);
  * method-ranking agreement: Spearman over methods of mean final regret, real vs fake, and the fraction of
    (task, method pair) comparisons with the same winner on the real task and on its twin.

Methods (prior-free ones included, so FT-PFN-style in-distribution effects can't dominate):
  rs        full-fidelity random search
  asha      asynchronous successive halving, eta 3, rungs T / 3^k
  greedy    freeze-thaw heuristic: with prob. 0.3 start a new config, else continue the run with the best last loss
  rd:<tag>  a rogue-dagger checkpoint (full table, tau 0.25), e.g. rd:wf03 (trained on ifBO)

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/m2.py --tasks 40 \
        --fakers ifbo noisefix dead --rd wf03=../rogue-dagger/runs/wide-focus03-v1/last.pt
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

from curvefaker.measures import ROGUE, ks, spearman, twins
from curvefaker.paths import REPO

sys.path.insert(0, str(REPO / "scripts"))
from m5_eval import task_list, to_rd  # noqa: E402

sys.path.insert(0, str(ROGUE))
from roguedagger.bench import Task as RDTask, TabEnv  # noqa: E402
from roguedagger.train import policy_step, random_search_step  # noqa: E402


# ---------------------------------------------------------------------------------------------------------- methods
class Asha:
    """Asynchronous successive halving on a TabEnv, numpy state on the CPU. Rung values come from the task's
    running-minimum table (what the run has shown up to that rung), so no curve copies per step."""

    def __init__(self, env, task, eta=3, seed=1):
        self.eta = eta
        T = int(env.T[0])
        K = max(0, int(math.floor(math.log(T, eta))))
        self.rungs = sorted({max(1, int(round(T / eta ** (K - k)))) for k in range(K + 1)})
        self.cm = np.minimum.accumulate(task.Y, axis=1)              # [n_cfg, T]
        E, R = env.E, env.R
        self.target = np.zeros((E, R), np.int64)
        self.promoted = np.zeros((E, R, len(self.rungs)), bool)
        self.rng = np.random.default_rng(seed)

    def __call__(self, env):
        E, R = env.E, env.R
        act = np.zeros(E, np.int64)
        rl, nr = env.run_len.cpu().numpy(), env.n_runs.cpu().numpy()
        cfg = env.run_cfg.cpu().numpy()
        unseen = env.unseen.cpu().numpy()
        for e in range(E):
            n = int(nr[e])
            busy = np.nonzero(rl[e, :n] < self.target[e, :n])[0]
            if len(busy):
                act[e] = busy[0]
                continue
            chosen = None
            for k in range(len(self.rungs) - 2, -1, -1):
                rk = self.rungs[k]
                reached = np.nonzero(rl[e, :n] >= rk)[0]
                top = len(reached) // self.eta
                if top == 0:
                    continue
                vals = self.cm[cfg[e, reached], rk - 1]
                best = reached[np.argsort(vals, kind="stable")[:top]]
                cand = best[~self.promoted[e, best, k]]
                if len(cand):
                    chosen = int(cand[0])
                    self.promoted[e, chosen, k] = True
                    self.target[e, chosen] = self.rungs[k + 1]
                    break
            if chosen is not None:
                act[e] = chosen
            else:
                free = np.nonzero(unseen[e])[0]
                self.target[e, n] = self.rungs[0]
                act[e] = R + int(self.rng.choice(free)) if len(free) else 0   # full table: shop slot = config id
        return torch.as_tensor(act, device=env.dev)


def greedy_step(env, gen, p_new=0.3):
    E = env.E
    ch = env.choosable()
    runs_ok = ch[:, :env.R]
    pos = (env.run_len - 1).clamp(min=0)
    last = env.run_curve.gather(2, pos[..., None])[..., 0].masked_fill(~runs_ok, float("inf"))
    best_run = last.argmin(1)
    shop_ok = ch[:, env.R:].float()
    pick = env.R + torch.multinomial(shop_ok + (shop_ok.sum(1, keepdim=True) == 0).float(), 1)[:, 0]
    new = (torch.rand(E, device=env.dev, generator=gen) < p_new) | ~runs_ok.any(1)
    return torch.where(new, pick, best_run)


def run(task: RDTask, method, models, seeds, budget, device):
    if not method.startswith("rd:"):
        device = "cpu"                      # prior-free methods: small ops, faster on the CPU
    env = TabEnv(task, seeds, budget, n_shop=None, device=device, seed=0)
    gen = torch.Generator(device=env.dev)
    gen.manual_seed(1)
    asha = Asha(env, task) if method == "asha" else None
    with torch.no_grad():
        while not env.done.all():
            if method == "rs":
                a = random_search_step(env)
            elif method == "asha":
                a = asha(env)
            elif method == "greedy":
                a = greedy_step(env, gen)
            else:
                m, h = models[method]
                a = policy_step(m, env, 0.25, False, h=h, amp=str(device).startswith("cuda"))
            env.step(a)
    reg = env.normalised_regret().cpu().numpy()                  # [E, B]
    T = int(env.T[0])
    out = {"final_regret": reg[:, -1].mean(), "regret@0.25": reg[:, budget // 4 - 1].mean(),
           "regret@0.5": reg[:, budget // 2 - 1].mean(), "n_started": env.n_runs.float().mean().item()}
    fid, full, chg = [], [], []
    for e in range(seeds):
        n = int(env.n_runs[e])
        ln = env.run_len[e, :n].float().cpu().numpy() / T
        fid.append(np.median(ln))
        full.append((ln >= 1).mean())
        tr = env.trace[e].cpu().numpy()
        chg.append(int((np.diff(tr) < -1e-9).sum()))
    out.update({"median_fidelity": float(np.mean(fid)), "frac_full": float(np.mean(full)),
                "incumbent_changes": float(np.mean(chg))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="select")
    ap.add_argument("--tasks", type=int, default=40, help="every k-th task of m5_eval's select sample, up to this many")
    ap.add_argument("--fakers", nargs="+", default=["ifbo", "noisefix", "dead"])
    ap.add_argument("--rd", nargs="*", default=[], help="tag=checkpoint")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="m2")
    a = ap.parse_args()
    ids = task_list(a.split)
    ids = ids[::max(1, len(ids) // a.tasks)][:a.tasks]
    models = {}
    for spec in a.rd:
        tag, path = spec.split("=", 1)
        from roguedagger.model import load_policy
        ck = torch.load(path, map_location=a.device)
        models[f"rd:{tag}"] = (load_policy(ck, a.device), (ck["cfg"]["train"]["h_start"], ck["cfg"]["train"]["h_end"]))
    methods = ["rs", "asha", "greedy"] + list(models)
    real = [to_rd(i) for i in ids]
    sets = {"real": real}
    shapes = [(t.Y.shape[0], t.X.shape[1], t.Y.shape[1]) for t in real]
    for f in a.fakers:
        sets[f] = [RDTask(f"{f}:{ids[k]}", f, X, Y.astype(np.float32))
                   for k, X, Y in twins(shapes, 1, f, a.device, seed=0)]
    res = {s: {m: [] for m in methods} for s in sets}
    t0 = time.time()
    for s, tasks in sets.items():
        for k, t in enumerate(tasks):
            for m in methods:
                res[s][m].append(run(t, m, models, a.seeds, a.budget, a.device))
            print(f"[{s} {k + 1}/{len(tasks)}] " + " ".join(f"{m}={res[s][m][-1]['final_regret']:.3f}" for m in methods)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    stats = list(res["real"][methods[0]][0])
    print(f"\n{len(ids)} tasks, methods {methods}")
    print("mean final regret: " + " | ".join(
        f"{s}: " + " ".join(f"{m}={np.mean([r['final_regret'] for r in res[s][m]]):.3f}" for m in methods) for s in sets))
    summary = {}
    for f in a.fakers:
        print(f"\n== {f} vs real")
        ksum = 0.0
        for m in methods:
            line = f"  {m:10s}"
            for st in stats:
                r = np.array([x[st] for x in res["real"][m]])
                z = np.array([x[st] for x in res[f][m]])
                d = ks(r, z)
                ksum += d
                line += f" {st}={np.median(r):.3g}/{np.median(z):.3g}({d:.2f})"
            print(line)
        mr = [np.mean([x["final_regret"] for x in res["real"][m]]) for m in methods]
        mf = [np.mean([x["final_regret"] for x in res[f][m]]) for m in methods]
        rho = spearman(np.array(mr), np.array(mf))
        agree, tot = 0, 0
        for i in range(len(ids)):
            for p in range(len(methods)):
                for q in range(p + 1, len(methods)):
                    dr = res["real"][methods[p]][i]["final_regret"] - res["real"][methods[q]][i]["final_regret"]
                    df = res[f][methods[p]][i]["final_regret"] - res[f][methods[q]][i]["final_regret"]
                    if abs(dr) > 1e-4 and abs(df) > 1e-4:
                        agree += (dr > 0) == (df > 0)
                        tot += 1
        print(f"  KS sum {ksum:.2f}; method-ranking Spearman (mean regret) {rho:+.2f}; "
              f"per-task pairwise agreement {agree / max(tot, 1):.2f} ({tot} comparisons)")
        summary[f] = {"ks_sum": ksum, "rank_rho": rho, "pair_agree": agree / max(tot, 1)}
    p = REPO / "runs" / "m2" / f"{a.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ids": ids, "methods": methods, "res": res, "summary": summary, "args": vars(a)},
                            default=float))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

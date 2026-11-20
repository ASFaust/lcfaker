"""Allowed learning-curve data in one format, behind the held-out guard (heldout.yaml) and the split (splits.yaml).

A Task is one HPO problem: a table of configs, each with a validation curve (lower is better), possibly with
several replicas (seeds).

    from curvefaker import data
    for t in data.tasks(split="fit", sources=["taskset", "fcnet"]):
        t.y      # [n, R, T] float32, NaN where missing
        t.X()    # [n, d] in [0, 1]

Sources (identifier pattern -> what y is):
    taskset/<task>/<opt>           valid1 loss, 5 replicas, 51 points over 10k steps; y_test = valid2
    nasbench201/<dataset>          validation error (%), 3 seeds, 200 epochs; 6 categorical hps
    fcnet/<dataset>                validation MSE, 4 seeds, 100 epochs; mixed space
    bbomix/<model>_<data>/<task>   validation reconstruction loss, 3 seeds, 300 epochs
    quicktune/<split>/<dataset>    100 - eval top-1 (%), 1 replica, up to 50 epochs, varying length per run
    jahs/<dataset>/N<n>_W<w>_R<res> 100 - valid acc (%), 1-5 replicas, 200 epochs; y_test = 100 - test acc
Every task id is checked against the held-out list before its data is read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from curvefaker.heldout import check, check_path, reason
from curvefaker.paths import DATA, REPO


@dataclass
class Task:
    id: str
    source: str
    family: str
    hps: pd.DataFrame            # raw hyperparameters, one row per config
    space: dict                  # name -> {"type": "float"|"int"|"cat", "log": bool}
    steps: np.ndarray            # [T] training step / epoch of each point
    y: np.ndarray                # [n, R, T] validation metric, lower is better
    metric: str
    y_test: np.ndarray | None = None   # [n, T] second held-aside split, where the source has one
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.y.shape[0]

    @property
    def T(self):
        return self.y.shape[2]

    def mean_curve(self):
        """[n, T] mean over replicas, ignoring NaN replicas."""
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(self.y, axis=1)

    def X(self, cat: str = "onehot") -> np.ndarray:
        """Configs in [0, 1]^d: float/int columns min-max scaled (log first where flagged), categoricals one-hot
        (cat="onehot") or as evenly spaced codes (cat="ordinal"). Constant columns are dropped."""
        cols = []
        for name, s in self.space.items():
            c = self.hps[name]
            if s["type"] == "cat":
                codes, uniq = pd.factorize(c.astype(str), sort=True)
                if len(uniq) < 2:
                    continue
                if cat == "onehot":
                    cols += [(codes == k).astype(np.float64) for k in range(len(uniq))]
                else:
                    cols.append(codes / (len(uniq) - 1))
            else:
                v = c.to_numpy(np.float64)
                if s.get("log"):
                    v = np.log(v)
                lo, hi = np.nanmin(v), np.nanmax(v)
                if hi > lo:
                    cols.append((v - lo) / (hi - lo))
        return np.stack(cols, 1) if cols else np.zeros((self.n, 0))


# ---- split -------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _split_of():
    s = yaml.safe_load((REPO / "splits.yaml").read_text())
    out = {}
    for part in ("fit", "select"):
        for fam in s[part]:
            assert fam not in out, f"family {fam} listed twice in splits.yaml"
            out[fam] = part
    return out


def split_of(family: str) -> str:
    try:
        return _split_of()[family]
    except KeyError:
        raise KeyError(f"family {family!r} is not assigned in splits.yaml") from None


# ---- helpers -----------------------------------------------------------------------------------------------------
def _infer_space(df: pd.DataFrame) -> dict:
    space = {}
    for c in df.columns:
        v = df[c]
        if v.dtype == object or v.dtype == bool or str(v.dtype) == "category":
            space[c] = {"type": "cat", "log": False}
        else:
            x = v.to_numpy(np.float64)
            nu = len(np.unique(x[~np.isnan(x)]))
            if nu < 2:
                continue
            log = bool((x > 0).all() and x.max() / x.min() > 100)
            integer = bool(np.all(np.mod(x[~np.isnan(x)], 1) == 0))
            space[c] = {"type": "int" if integer else "float", "log": log}
    return space


def _read(path: Path):
    check_path(path)
    return path


# ---- TaskSet -----------------------------------------------------------------------------------------------------
TASKSET_OPTS = {"adam1p": "adam1p_wide_grid", "adam4p": "adam4p_wide_grid", "adam6p": "adam6p_wide_grid",
                "adam8p": "adam8p_wide_grid", "nadamw": "nadamw_grid"}


def taskset_family(task: str) -> str:
    head = task.split("_")[0]
    return f"taskset-{head}"


@lru_cache(maxsize=None)
def _taskset_hps(opt: str) -> pd.DataFrame:
    cfg = TASKSET_OPTS[opt]
    raw = json.loads(_read(DATA / "taskset" / "hp_configs" / f"{cfg}.json").read_text())
    rows = {int(k.split("_seed")[1]): v[0] for k, v in raw.items()}
    return pd.DataFrame([rows[i] for i in sorted(rows)])


def taskset_ids():
    root = DATA / "taskset"
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "hp_configs"):
        for f in sorted(d.glob("*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            out.append(f"taskset/{d.name}/{f.stem}")
    return out


def load_taskset(ident: str) -> Task:
    check(ident)
    _, task, opt = ident.split("/")
    z = np.load(_read(DATA / "taskset" / task / f"{opt}.npz"))
    ys = z["ys"]                                  # [n, 5, T, 4]: train, valid1, valid2, test
    hps = _taskset_hps(opt)
    n = min(len(hps), ys.shape[0])
    hps = hps.iloc[:n].reset_index(drop=True)
    y = np.moveaxis(ys[:n, :, :, 1], 0, 0).astype(np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (diverged in every replica)
        y_test = np.nanmean(ys[:n, :, :, 2], axis=1).astype(np.float32)
    return Task(ident, "taskset", taskset_family(task), hps, _infer_space(hps), z["xs"], y, "valid1_loss", y_test,
                {"train": ys[:n, :, :, 0]})


# ---- syne-tune blackboxes ----------------------------------------------------------------------------------------
SYNETUNE = {   # source -> (blackboxes, objective used as y)
    "nasbench201": (["nasbench201"], "metric_valid_error"),
    "fcnet": (["fcnet"], "metric_valid_loss"),
    "bbomix": ([f"bbomix_{m}_{d}" for m in ("disentanglix", "ontix", "vanillix", "varix") for d in ("schc", "tcga")],
               "metric_valid_recon_loss"),
}


@lru_cache(maxsize=None)
def _bb(name: str):
    check(f"synetune/{name}")
    d = DATA / "synetune" / name
    meta = json.loads(_read(d / "metadata.json").read_text())
    hps = pd.read_parquet(_read(d / "hyperparameters.parquet"))
    hps.columns = [c[3:] if c.startswith("hp_") else c for c in hps.columns]
    cs = json.loads((d / "configspace.json").read_text())
    space = {}
    for k, v in cs.items():
        k2 = k[3:] if k.startswith("hp_") else k
        if v["domain_cls"] == "Categorical":
            cats = v["domain_kwargs"]["categories"]
            numeric = all(isinstance(c, (int, float)) for c in cats)
            if numeric:   # ordinal choices stored as categorical (e.g. layer counts): keep them numeric
                arr = np.asarray(cats, float)
                space[k2] = {"type": "int" if np.all(np.mod(arr, 1) == 0) else "float",
                             "log": bool(arr.min() > 0 and arr.max() / arr.min() > 8)}
            else:
                space[k2] = {"type": "cat", "log": False}
        else:
            kw = v["domain_kwargs"]
            space[k2] = {"type": "int" if kw.get("cast_int") or v["domain_cls"] == "Integer" else "float",
                         "log": bool(kw.get("log_scale", False)) or "Log" in v.get("sampler_cls", "")}
    return meta, hps, space, d


def synetune_ids():
    out = []
    for src, (bbs, _) in SYNETUNE.items():
        for bb in bbs:
            if not (DATA / "synetune" / bb / "objectives_evaluations.npy").exists():
                continue
            meta = json.loads((DATA / "synetune" / bb / "metadata.json").read_text())
            for t in meta["task_names"]:
                out.append(f"{src}/{bb[7:]}/{t}" if src == "bbomix" else f"{src}/{t}")
    return out


def load_synetune(ident: str) -> Task:
    check(ident)
    parts = ident.split("/")
    src = parts[0]
    bb = f"bbomix_{parts[1]}" if src == "bbomix" else SYNETUNE[src][0][0]
    task = parts[-1]
    meta, hps, space, d = _bb(bb)
    check(f"synetune/{bb}/{task}")
    o = np.load(_read(d / "objectives_evaluations.npy"), mmap_mode="r")
    ti = meta["task_names"].index(task)
    oi = meta["objectives_names"].index(SYNETUNE[src][1])
    y = np.asarray(o[ti, :, :, :, oi], dtype=np.float32)          # [n, seeds, T]
    steps = np.load(d / "fidelities_values.npy")
    return Task(ident, src, src, hps, space, steps, y, SYNETUNE[src][1], meta={"blackbox": bb})


# ---- QuickTune ---------------------------------------------------------------------------------------------------
QT_HPS = ["model", "opt", "opt_betas", "sched", "auto_augment", "batch_size", "lr", "weight_decay", "momentum",
          "drop", "smoothing", "mixup", "mixup_prob", "cutmix", "clip_grad", "layer_decay", "warmup_epochs",
          "warmup_lr", "decay_epochs", "decay_rate", "patience_epochs", "pct_to_freeze", "linear_probing",
          "stoch_norm", "sp_reg", "delta_reg", "cotuning_reg", "bss_reg", "ra_num_ops", "ra_magnitude",
          "random_augment", "trivial_augment"]


@lru_cache(maxsize=None)
def _qt(split: str):
    d = DATA / "quicktune" / split / split
    top1 = json.loads(_read(d / "eval_top1.json").read_text())
    epoch = json.loads((d / "epoch.json").read_text())
    tab = pd.read_csv(d / "unnormalized_args_table.csv", index_col=0)
    # one-hot categoricals back to columns
    for g in ("model", "opt", "opt_betas", "sched", "auto_augment", "dataset"):
        cols = [c for c in tab.columns if c.startswith(f"cat__{g}_") and not
                (g == "opt" and c.startswith("cat__opt_betas_"))]
        if cols:
            tab[g] = tab[cols].to_numpy().argmax(1)
            tab[g] = [cols[i][len(f"cat__{g}_"):] for i in tab[g]]
    return top1, epoch, tab


def quicktune_ids():
    out = []
    for split in ("micro", "mini", "extended"):
        if (DATA / "quicktune" / split / split / "eval_top1.json").exists():
            top1, _, _ = _qt(split)
            out += [f"quicktune/{split}/{ds.replace('/', '__')}" for ds in sorted(top1)]
    return out


def load_quicktune(ident: str) -> Task:
    check(ident)
    _, split, ds = ident.split("/")
    ds = ds.replace("__", "/")
    top1, epoch, tab = _qt(split)
    runs = [r for r in top1[ds] if r in tab.index]
    T = max(len(top1[ds][r]) for r in runs)
    y = np.full((len(runs), 1, T), np.nan, np.float32)
    for i, r in enumerate(runs):
        c = np.asarray(top1[ds][r], np.float64)
        y[i, 0, :len(c)] = 100.0 - c
    hps = tab.loc[runs, [c for c in QT_HPS if c in tab.columns]].reset_index(drop=True)
    space = _infer_space(hps)
    return Task(ident, "quicktune", "quicktune", hps, space, np.arange(1, T + 1), y, "100-top1",
                meta={"curve_length": np.isfinite(y[:, 0]).sum(1)})


# ---- JAHS-Bench-201 ---------------------------------------------------------------------------------------------
JAHS_SPACE = {"Activation": {"type": "cat", "log": False},
              **{f"Op{i}": {"type": "cat", "log": False} for i in range(1, 7)},
              "LearningRate": {"type": "float", "log": True}, "WeightDecay": {"type": "float", "log": True},
              "TrivialAugment": {"type": "cat", "log": False}}


def jahs_ids():
    root = DATA / "jahs" / "tasks"
    if not root.exists():
        return []
    return [f"jahs/{f.parent.name}/{f.stem}" for f in sorted(root.glob("*/*.npz"))]


def load_jahs(ident: str) -> Task:
    """One (dataset, N, W, Resolution) of JAHS-Bench-201 (files from scripts/prepare_jahs.py). Full 200-epoch runs
    only; distinct runs of one config are replicas (NaN-padded), exact duplicate runs were dropped."""
    check(ident)
    _, ds, name = ident.split("/")
    z = np.load(_read(DATA / "jahs" / "tasks" / ds / f"{name}.npz"))
    hps = pd.DataFrame({h: z[f"hp_{h}"] for h in JAHS_SPACE})
    for i in range(1, 7):   # cell operations are integer codes of categorical choices
        hps[f"Op{i}"] = hps[f"Op{i}"].astype(str)
    return Task(ident, "jahs", "jahs", hps, dict(JAHS_SPACE), z["epochs"], z["y"], "100-valid_acc", z["y_test"],
                meta={"dataset": ds, "fixed": name})


# ---- registry ----------------------------------------------------------------------------------------------------
SOURCES = {
    "taskset": (taskset_ids, load_taskset),
    "nasbench201": (lambda: [i for i in synetune_ids() if i.startswith("nasbench201/")], load_synetune),
    "fcnet": (lambda: [i for i in synetune_ids() if i.startswith("fcnet/")], load_synetune),
    "bbomix": (lambda: [i for i in synetune_ids() if i.startswith("bbomix/")], load_synetune),
    "quicktune": (quicktune_ids, load_quicktune),
    "jahs": (jahs_ids, load_jahs),
}


def family_of(ident: str) -> str:
    src = ident.split("/")[0]
    return taskset_family(ident.split("/")[1]) if src == "taskset" else src


def ids(split: str | None = None, sources=None) -> list[str]:
    """Allowed task ids, optionally restricted to one part of the internal split ("fit" / "select")."""
    out = []
    for src in sources or SOURCES:
        for i in SOURCES[src][0]():
            if reason(i) is not None:
                continue
            if split is None or split_of(family_of(i)) == split:
                out.append(i)
    return out


def load(ident: str) -> Task:
    return SOURCES[ident.split("/")[0]][1](ident)


def tasks(split: str | None = None, sources=None):
    for i in ids(split, sources):
        yield load(i)

"""Download every allowed TaskSet task x all 5 optimizer families from gs://gresearch/task_set_data.

Held-out tasks (heldout.yaml: the FixedTextRNNClassification family) are skipped before any request.
Output, under $DATA/taskset:
    hp_configs/<optimizer config>.json    hyperparameters of the 1000 configs (google-research repo)
    <task>/<opt>.npz                      xs [T] steps; ys [1000, 5 replicas, T, 4] float32 with the 4
                                          splits (train, valid1, valid2, test); optimizers [1000]
Raw files are float64 (~8 MB); stored float32 + compressed (~3.5 MB). Re-running skips finished files.

    PYTHONPATH=. python scripts/download_taskset.py [--workers 8] [--opts adam4p adam8p]
"""
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from curvefaker.heldout import check, reason
from curvefaker.paths import DATA

BUCKET = "https://storage.googleapis.com/storage/v1/b/gresearch/o?"
CURVE_URL = "https://storage.googleapis.com/gresearch/task_set_data/{task}/{cfg}_10000_replica5.npz"
HP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs/{cfg}.json"
OPTS = {"adam1p": "adam1p_wide_grid_1k", "adam4p": "adam4p_wide_grid_1k", "adam6p": "adam6p_wide_grid_1k",
        "adam8p": "adam8p_wide_grid_1k", "nadamw": "nadamw_grid_1k"}


def fetch(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def list_tasks():
    out, tok = [], None
    while True:
        q = {"prefix": "task_set_data/", "delimiter": "/", "maxResults": 1000}
        if tok:
            q["pageToken"] = tok
        d = json.loads(fetch(BUCKET + urllib.parse.urlencode(q)))
        out += [p.split("/")[1] for p in d.get("prefixes", [])]
        tok = d.get("nextPageToken")
        if not tok:
            return sorted(t for t in out if t != "task_set_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--opts", nargs="+", default=list(OPTS))
    a = ap.parse_args()
    root = DATA / "taskset"
    (root / "hp_configs").mkdir(parents=True, exist_ok=True)
    for o in a.opts:
        cfg = OPTS[o].replace("_1k", "")
        p = root / "hp_configs" / f"{cfg}.json"
        if not p.exists():
            p.write_bytes(fetch(HP_URL.format(cfg=cfg)))
    tasks = list_tasks()
    allowed = [t for t in tasks if reason(f"taskset/{t}") is None]
    print(f"{len(tasks)} tasks in bucket, {len(tasks) - len(allowed)} held out, {len(allowed)} to fetch", flush=True)

    def one(task, o):
        ident = f"taskset/{task}/{o}"
        check(ident)
        path = root / task / f"{o}.npz"
        if path.exists():
            return "cached"
        try:
            raw = fetch(CURVE_URL.format(task=task, cfg=OPTS[o]))
        except urllib.error.HTTPError as e:
            return f"missing ({e.code})"
        z = np.load(io.BytesIO(raw))
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, xs=z["xs"], ys=z["ys"].astype(np.float32), optimizers=z["optimizers"])
        tmp.rename(path)
        return "ok"

    jobs = [(t, o) for t in allowed for o in a.opts]
    counts, t0 = {}, time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, t, o): (t, o) for t, o in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                r = f"error {type(e).__name__}"
                print(futs[f], r, e, flush=True)
            counts[r.split(" ")[0]] = counts.get(r.split(" ")[0], 0) + 1
            if r.startswith("missing"):
                print(futs[f], r, flush=True)
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

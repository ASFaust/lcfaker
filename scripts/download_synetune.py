"""Download learning-curve blackboxes from syne-tune's repository on Hugging Face (synetune/blackbox-repository).

Only blackboxes with per-epoch curves are fetched; held-out ones (lcbench, pd1, yahpo's lcbench) are refused by
the guard. Files are stored as published, under $DATA/synetune/<name>/.

    PYTHONPATH=. python scripts/download_synetune.py [names ...]
"""
import json
import shutil
import sys
import urllib.request

from curvefaker.heldout import check
from curvefaker.paths import DATA

API = "https://huggingface.co/api/datasets/synetune/blackbox-repository/tree/main/{name}"
FILE = "https://huggingface.co/datasets/synetune/blackbox-repository/resolve/main/{path}"
DEFAULT = ["nasbench201", "fcnet"] + [f"bbomix_{m}_{d}" for m in ("disentanglix", "ontix", "vanillix", "varix")
                                      for d in ("schc", "tcga")]


def main(names):
    for name in names:
        check(f"synetune/{name}")
        out = DATA / "synetune" / name
        out.mkdir(parents=True, exist_ok=True)
        for f in json.load(urllib.request.urlopen(API.format(name=name))):
            dst = DATA / "synetune" / f["path"]
            if f["type"] != "file" or (dst.exists() and dst.stat().st_size == f.get("size")):
                continue
            tmp = dst.with_name(dst.name + ".part")
            with urllib.request.urlopen(FILE.format(path=f["path"]), timeout=600) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w, 1 << 22)
            tmp.rename(dst)
            print(f"{f['path']} {f.get('size', 0) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

"""Train a rogue-dagger policy (default config: wide-focus03) on a curve-faker faker, for M5.

rogue-dagger is used unchanged: its env builds `roguedagger.env.PriorTasks`, which this launcher replaces with the
named faker (curvefaker.fakers.FAKERS) before rogue-dagger's own train.main() runs. Checkpoints go to
curve-faker/runs/<name>/ (train.py writes relative to the working directory). One GPU only (project rule).

    cd curve-faker; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/train_rd.py \
        --faker dead --set name=wf03-dead-s0 seed=0 wandb.project=curve-faker
"""
import sys

from curvefaker import fakers
from curvefaker.measures import ROGUE

if __name__ == "__main__":
    argv = sys.argv[1:]
    i = argv.index("--faker")
    name = argv[i + 1]
    del argv[i:i + 2]
    if "--config" not in argv:
        argv = ["--config", str(ROGUE / "configs" / "wide_focus03.yaml")] + argv
    import roguedagger.env as rd_env
    import roguedagger.train as rd_train
    rd_env.PriorTasks = fakers.get(name)
    print(f"faker: {name} {fakers.FAKERS[name]}", flush=True)
    sys.argv = [sys.argv[0]] + argv
    rd_train.main()

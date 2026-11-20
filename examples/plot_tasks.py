"""Generate a few tasks and plot their curves (colour: final value, dark = good).

    pip install -e .[examples]
    python examples/plot_tasks.py
"""
import matplotlib.pyplot as plt
import numpy as np

from lcfaker import generate

tasks = generate(6, seed=0, source_weights={"taskset": 1, "jahs": 1, "quicktune": 1})
fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), sharey=True)
for ax, t in zip(axs.flat, tasks):
    idx = np.random.default_rng(0).choice(len(t.y), 80, replace=False)
    y = t.y[idx]
    T = y.shape[1]
    fin = y[:, -max(1, T // 10):].mean(1)
    rank = np.argsort(np.argsort(-fin)) / (len(fin) - 1)
    for k in np.argsort(-fin):
        ax.plot(np.arange(1, T + 1) / T, y[k], lw=0.7, color=plt.cm.Blues(0.3 + 0.7 * rank[k]))
    ax.set_xscale("log")
    ax.set_title(f"{t.source} record, d={t.X.shape[1]}, T={T}, {len(t.y)} configs", fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("budget fraction t/T")
for ax in axs[:, 0]:
    ax.set_ylabel("normalised loss")
fig.tight_layout()
fig.savefig("tasks.png", dpi=120)
print("wrote tasks.png")

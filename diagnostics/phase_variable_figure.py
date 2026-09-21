"""
phase_variable_figure.py — the corrected phase-variable figure (paper Fig. 2).

The coupled identity A' = (1+η)A − ηR^{γλ} reduces to a pure rescaling on a
batch only if that batch contains NO nonzero raw reward. The original figure
plotted the per-transition nonzero-reward fraction, which stays small; the
statistic that decides whether normalization annihilates shaping is the fraction
of updates whose batch contains any nonzero reward. This figure shows both.

Row 1: per-transition fraction of nonzero raw reward (%), seed mean.
Row 2: fraction of updates whose batch contains >= 1 nonzero raw reward (%), seed mean.
Both rows: centred rolling mean over 25 updates (~51k steps). The oracle is omitted
(it is 100% on both statistics by construction). On the hard task the statistic
counts the −10 collision penalty as well as successes.

Usage (repo root): python diagnostics/phase_variable_figure.py
Output: figures_s2d/F7b_phase_variable.png
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = 25
SERIES = [  # fixed categorical order (validated palette slots 1-4) + line style for print/CVD
    ("sparse",     "Sparse",            "#2a78d6", "-"),
    ("bsrs_c_e1",  "Coupled η = 1",     "#eb6834", "--"),
    ("bsrs_d_e1",  "Decoupled η = 1",   "#1baf7a", "-."),
    ("her_final",  "PPO-HER",           "#eda100", ":"),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def curves(task, cond):
    mats = []
    for f in sorted(glob.glob(f"results_s2d/{task}/{cond}_seed*/progress.csv")):
        d = pd.read_csv(f)
        mats.append((d["timestep"].values, d["nonzero_raw_reward_frac"].values))
    n = min(len(x) for _, x in mats)
    ts = mats[0][0][:n]
    X = np.vstack([x[:n] for _, x in mats])
    roll = lambda v: pd.Series(v).rolling(WINDOW, center=True, min_periods=1).mean().values
    per_transition = roll(X.mean(0)) * 100
    any_reward = roll((X > 0).mean(0)) * 100
    return ts, per_transition, any_reward


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, facecolor=SURF)
    rows = [("(a) Transitions with\nnonzero reward (%)", 1, (0, 2.0)),
            ("(b) Updates whose batch has\nany nonzero reward (%)", 2, (0, 100))]
    for c, task in enumerate(["easy", "hard"]):
        for r, (ylabel, idx, ylim) in enumerate(rows):
            ax = axes[r, c]
            ax.set_facecolor(SURF)
            for cond, label, color, ls in SERIES:
                ts, a, b = curves(task, cond)
                ax.plot(ts, a if idx == 1 else b, color=color, ls=ls, lw=2, label=label)
            ax.set_ylim(*ylim)
            ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(f"{task.capitalize()} task", color=INK, fontweight="bold")
            if r == 1:
                ax.set_xlabel("Training timestep")
                ax.set_xticks(range(0, 500_001, 100_000))
                ax.set_xticklabels(["0"] + [f"{k}k" for k in range(100, 501, 100)])
    axes[0, 0].axhline(1.5, color=INK2, lw=1, ls=(0, (1, 2)))
    axes[0, 0].text(8e3, 1.56, "1.5% (threshold quoted in the original text)", color=INK2, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs("figures_s2d", exist_ok=True)
    out = "figures_s2d/F7b_phase_variable.png"
    fig.savefig(out, dpi=300, facecolor=SURF)
    print("wrote", out)
    # Numbers quoted in the paper (all training / last 50 updates, easy task)
    for task in ("easy", "hard"):
        for cond in ("sparse", "bsrs_c_e05", "bsrs_c_e1", "bsrs_c_e2", "bsrs_c_e1_nonorm", "her_final"):
            X = np.vstack([pd.read_csv(f)["nonzero_raw_reward_frac"].values[:245]
                           for f in sorted(glob.glob(f"results_s2d/{task}/{cond}_seed*/progress.csv"))])
            print(f"  {task}/{cond:<17} any-reward updates: all {100*(X>0).mean():5.1f}%  "
                  f"last-50 {100*(X[:, -50:]>0).mean():5.1f}%")


if __name__ == "__main__":
    main()

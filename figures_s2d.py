"""
figures_s2d.py
==============
Publication figures for the sparse→dense study → figures_s2d/ (300 DPI).

  F1 final success-rate IQM ± 95% bootstrap CI, all conditions, easy|hard panels,
     with sparse floor and oracle ceiling reference lines.
  F2 learning curves: eval success vs steps (sparse/oracle/coupled/decoupled/HER).
  F3 η-sensitivity: final success vs η for coupled and decoupled.
  F4 P2 panel: sparse vs bsrs_c_e1 vs sparse_nonorm vs bsrs_c_e1_nonorm.
  F5 P4 panel: mean |value target| over training for η ∈ {0(=sparse),0.5,1,2}.
  F6 HER diagnostics: relabeled fraction, split clip fractions, entropy.
  F7 nonzero-reward batch fraction over training (P1 phase-dependence).

Reads results_s2d/eval_s2d.csv, the per-run evaluations.npz and progress.csv.
Robust to partial data — only present conditions/tasks are drawn.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_s2d_study import CONDITIONS, TASK_OBSTACLE, RESULTS_ROOT
from s2d_stats import iqm, bootstrap_ci

FIG_DIR = "figures_s2d"
TASKS = ["easy", "hard"]

PALETTE = {
    "sparse": "#7f7f7f", "oracle": "#111111",
    "bsrs_c_e05": "#9ecae1", "bsrs_c_e1": "#4292c6", "bsrs_c_e2": "#08519c",
    "bsrs_c_e1_nonorm": "#d95f02", "sparse_nonorm": "#e7298a",
    "bsrs_d_e05": "#a1d99b", "bsrs_d_e1": "#41ab5d", "bsrs_d_e2": "#006d2c",
    "her_final": "#6a51a3",
}
COUPLED_ETA = {"bsrs_c_e05": 0.5, "bsrs_c_e1": 1.0, "bsrs_c_e2": 2.0}
DECOUPLED_ETA = {"bsrs_d_e05": 0.5, "bsrs_d_e1": 1.0, "bsrs_d_e2": 2.0}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": "white",
    })


# ── Loaders ───────────────────────────────────────────────────────────────────
def seeds_for(task, cond):
    out = []
    for d in glob.glob(os.path.join(RESULTS_ROOT, task, f"{cond}_seed*")):
        s = os.path.basename(d).rsplit("_seed", 1)[-1]
        if s.isdigit() and os.path.exists(os.path.join(d, "final_model.zip")):
            out.append(int(s))
    return sorted(out)


def load_eval():
    p = os.path.join(RESULTS_ROOT, "eval_s2d.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def learning_curve(task, cond):
    """Return (timesteps, success_matrix (n_seeds, n_evals)) from evaluations.npz."""
    ts_ref, mats = None, []
    for s in seeds_for(task, cond):
        f = os.path.join(RESULTS_ROOT, task, f"{cond}_seed{s}", "evaluations.npz")
        if not os.path.exists(f):
            continue
        data = np.load(f)
        if "successes" not in data.files:
            continue
        ts_ref = data["timesteps"]
        mats.append(data["successes"].mean(axis=1))  # success rate per eval
    if not mats:
        return None, None
    n = min(len(m) for m in mats)
    return ts_ref[:n], np.vstack([m[:n] for m in mats])


def progress_curve(task, cond, col):
    """Return (timesteps, matrix (n_seeds, n_updates)) for a progress.csv column."""
    ts_ref, mats = None, []
    for s in seeds_for(task, cond):
        f = os.path.join(RESULTS_ROOT, task, f"{cond}_seed{s}", "progress.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if col not in df.columns:
            continue
        ts_ref = df["timestep"].values
        mats.append(df[col].values)
    if not mats:
        return None, None
    n = min(len(m) for m in mats)
    return ts_ref[:n], np.vstack([m[:n] for m in mats])


def _band(ax, ts, mat, color, label):
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    ax.plot(ts, mean, color=color, lw=2, label=label)
    ax.fill_between(ts, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)


# ── F1 — success IQM ± CI, easy|hard ──────────────────────────────────────────
def fig1(ev):
    order = list(CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = ev[ev.task == task]
        conds = [c for c in order if c in sub.condition.values]
        pts, los, his = [], [], []
        for c in conds:
            sr = sub[sub.condition == c]["success_rate"].values
            pt, lo, hi = bootstrap_ci(sr, agg=iqm)
            pts.append(pt * 100); los.append((pt - lo) * 100); his.append((hi - pt) * 100)
        x = np.arange(len(conds))
        ax.bar(x, pts, yerr=[los, his], color=[PALETTE[c] for c in conds],
               edgecolor="black", lw=0.6, capsize=3, error_kw={"elinewidth": 1})
        if "sparse" in conds:
            ax.axhline(pts[conds.index("sparse")], ls=":", color=PALETTE["sparse"], lw=1.3)
        if "oracle" in conds:
            ax.axhline(pts[conds.index("oracle")], ls="--", color="black", lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{task} task"); ax.set_ylim(0, 105)
    axes[0].set_ylabel("Final success rate (%)  — IQM ± 95% CI")
    fig.suptitle("Final success by condition (dotted=sparse floor, dashed=oracle ceiling)",
                 fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F1_success_iqm_ci.png")); plt.close(fig)


# ── F2 — learning curves ──────────────────────────────────────────────────────
def fig2():
    show = ["sparse", "oracle", "bsrs_c_e1", "bsrs_d_e1", "her_final"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        for c in show:
            ts, mat = learning_curve(task, c)
            if ts is None:
                continue
            _band(ax, ts, mat * 100, PALETTE[c], c)
        ax.set_title(f"{task} task"); ax.set_xlabel("Training timestep")
        ax.set_ylim(-3, 105); ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Eval success rate (%)")
    axes[1].legend(loc="best")
    fig.suptitle("Learning curves: eval success vs steps", fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F2_learning_curves.png")); plt.close(fig)


# ── F3 — η sensitivity ────────────────────────────────────────────────────────
def fig3(ev):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = ev[ev.task == task]
        for fam, color, name in [(COUPLED_ETA, "#08519c", "coupled"),
                                 (DECOUPLED_ETA, "#006d2c", "decoupled")]:
            xs, ys, es = [], [], []
            for cond, eta in sorted(fam.items(), key=lambda kv: kv[1]):
                sr = sub[sub.condition == cond]["success_rate"].values
                if len(sr) == 0:
                    continue
                pt, lo, hi = bootstrap_ci(sr, agg=iqm)
                xs.append(eta); ys.append(pt * 100); es.append((hi - lo) / 2 * 100)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=color, lw=2,
                            capsize=3, label=name)
        for ref, ls, col in [("sparse", ":", PALETTE["sparse"]), ("oracle", "--", "black")]:
            r = sub[sub.condition == ref]["success_rate"].values
            if len(r):
                ax.axhline(iqm(r) * 100, ls=ls, color=col, lw=1.2, label=ref)
        ax.set_title(f"{task} task"); ax.set_xlabel("η (shaping strength)")
        ax.set_xticks([0.5, 1.0, 2.0]); ax.set_ylim(0, 105)
    axes[0].set_ylabel("Final success rate (%)")
    axes[1].legend(loc="best")
    fig.suptitle("η-sensitivity: coupled vs decoupled BSRS", fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F3_eta_sensitivity.png")); plt.close(fig)


# ── F4 — P2 panel (advantage normalization on/off) ────────────────────────────
def fig4(ev):
    conds = ["sparse", "bsrs_c_e1", "sparse_nonorm", "bsrs_c_e1_nonorm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = ev[ev.task == task]
        present = [c for c in conds if c in sub.condition.values]
        x = np.arange(len(present))
        pts, err = [], []
        for c in present:
            sr = sub[sub.condition == c]["success_rate"].values
            pt, lo, hi = bootstrap_ci(sr, agg=iqm)
            pts.append(pt * 100); err.append([(pt - lo) * 100, (hi - pt) * 100])
        err = np.array(err).T if err else np.zeros((2, 0))
        ax.bar(x, pts, yerr=err, color=[PALETTE[c] for c in present],
               edgecolor="black", lw=0.6, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(present, rotation=25, ha="right", fontsize=8.5)
        ax.set_title(f"{task} task"); ax.set_ylim(0, 105)
    axes[0].set_ylabel("Final success rate (%)")
    fig.suptitle("P2: coupled BSRS as LR-multiplier when advantage-norm is OFF",
                 fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F4_p2_normalization.png")); plt.close(fig)


# ── F5 — P4 panel: value-target magnitude over training ───────────────────────
def fig5():
    fam = [("sparse", 0.0), ("bsrs_c_e05", 0.5), ("bsrs_c_e1", 1.0), ("bsrs_c_e2", 2.0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        for cond, eta in fam:
            ts, mat = progress_curve(task, cond, "value_target_mean_abs")
            if ts is None:
                continue
            _band(ax, ts, mat, PALETTE[cond], f"η={eta:g} ({cond})")
        ax.set_title(f"{task} task"); ax.set_xlabel("Training timestep")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Mean |value target|")
    axes[1].legend(loc="best")
    fig.suptitle("P4: coupled BSRS inflates critic value targets with η",
                 fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F5_p4_value_targets.png")); plt.close(fig)


# ── F6 — HER diagnostics ──────────────────────────────────────────────────────
def fig6():
    cols = [("relabeled_fraction", "Relabeled fraction"),
            ("clip_fraction_original", "Clip frac (original)"),
            ("clip_fraction_relabeled", "Clip frac (relabeled)"),
            ("entropy", "Policy entropy")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), cols):
        for task, ls in [("easy", "-"), ("hard", "--")]:
            ts, mat = progress_curve(task, "her_final", col)
            if ts is None:
                continue
            ax.plot(ts, np.nanmean(mat, axis=0), color=PALETTE["her_final"],
                    ls=ls, lw=2, label=task)
        ax.set_title(title); ax.set_xlabel("Training timestep")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0)); ax.legend()
    fig.suptitle("F6: PPO-HER diagnostics (her_final)", fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F6_her_diagnostics.png")); plt.close(fig)


# ── F7 — nonzero-reward batch fraction (P1 phase-dependence) ──────────────────
def fig7():
    show = ["sparse", "oracle", "bsrs_c_e1", "bsrs_d_e1", "her_final"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        for c in show:
            ts, mat = progress_curve(task, c, "nonzero_raw_reward_frac")
            if ts is None:
                continue
            _band(ax, ts, mat, PALETTE[c], c)
        ax.set_title(f"{task} task"); ax.set_xlabel("Training timestep")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Fraction of batch with nonzero raw reward")
    axes[1].legend(loc="best")
    fig.suptitle("F7: nonzero-reward batch fraction (P1 phase-dependence)",
                 fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "F7_nonzero_reward_fraction.png")); plt.close(fig)


CAPTIONS = """\
# Figure captions — sparse→dense study

**F1.** Final success rate (IQM over 5 seeds, ±95% stratified bootstrap CI, 10k
resamples) per condition, easy and hard tasks. Dotted line = sparse floor,
dashed = oracle ceiling.

**F2.** Evaluation success rate vs training timestep for sparse, oracle, coupled
(η=1), decoupled (η=1), and PPO-HER; mean ±1 SD across seeds.

**F3.** Final success vs η for coupled and decoupled BSRS, with sparse/oracle
reference lines. Tests whether decoupling escapes the coupled collapse (P3).

**F4.** P2: final success for sparse vs coupled-η1 with advantage normalization
ON, and their `_nonorm` counterparts with it OFF.

**F5.** P4: mean |critic value target| over training for coupled η∈{0,0.5,1,2}
(η=0 ≡ sparse). Growth with η indicates value-target inflation.

**F6.** PPO-HER diagnostics over training: relabeled fraction, clip fraction on
original vs relabeled transitions, and policy entropy (easy solid, hard dashed).

**F7.** Fraction of each update batch with nonzero raw reward over training —
the phase variable behind P1 (advantage-norm annihilation is exact only while
batches are all-zero-reward).
"""


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    ev = load_eval()
    made = []
    if ev is not None and len(ev):
        fig1(ev); made.append("F1")
        fig3(ev); made.append("F3")
        fig4(ev); made.append("F4")
    fig2(); made.append("F2")
    fig5(); made.append("F5")
    fig6(); made.append("F6")
    fig7(); made.append("F7")
    with open(os.path.join(FIG_DIR, "captions.md"), "w") as f:
        f.write(CAPTIONS)
    print(f"Wrote figures {made} + captions.md to {FIG_DIR}/")


if __name__ == "__main__":
    main()

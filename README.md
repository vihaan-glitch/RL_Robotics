# RL---Robotics — Sparse-Reward Reaching with PPO on a Simulated KUKA iiwa

Code and data for **"Bootstrapped Reward Shaping Does Not Transfer to PPO"**
(V. Kulkarni, 2026), plus the earlier reward-ablation study it builds on.

Everything here runs on **MuJoCo** (via `mujoco` ≥ 3.0) through a custom
Gymnasium environment. There is no PyBullet code in the current tree; the
project's first commit (Jan 2026) used PyBullet and was replaced in `3cec257`.

## Contents

| Study | Entry point | Results | Write-up |
|---|---|---|---|
| **Sparse→dense study** (the paper): 11 conditions × 2 tasks × 5 seeds = 110 runs | `run_s2d_study.py` | `results_s2d/`, `figures_s2d/` | `results_s2d_summary.md`, `review_report.md` |
| Post-review diagnostics, including the P4 fixed-point test | `diagnostics/*.py` | `results_p4_fixedpoint/` | `review_report.md` (with errata) |
| Earlier reward-ablation study: dense reward terms, easy + hard | `run_ablation_study.py` | `results/`, `results_hard/`, `figures/` | `results_summary.md` |
| Standalone single-run training (PPO / PPO-Lagrangian) | `train.py`, `evaluate.py` | — | — |

## Environment — `arm_reach_env.py`

| Property | Value |
|---|---|
| Robot | KUKA iiwa, 7 DoF, MuJoCo |
| Observation (21-d) | joint pos (7) · joint vel (7) · target (3) · end-effector (3) · distance (1) |
| Action (7-d) | continuous joint-position targets in `[-1, 1]` |
| Success / termination | end-effector within 0.05 m of target (terminates) |
| Truncation | 500 steps |
| `obstacle_mode=None` | **easy** task |
| `obstacle_mode="simple"` | **hard** task: 3 box obstacles, −10 per step with any arm–obstacle contact, applied in every reward mode |

`reward_mode` selects which terms are summed (`REWARD_MODES`):

| mode | terms |
|---|---|
| `sparse` | +100 on success only (plus −10 collision penalty on the hard task) |
| `full` | −distance + 10·(prev_d − d) + 100·success − 0.01 per step (the dense "oracle") |
| `no_progress`, `no_time_penalty` | ablations used by the earlier study |

## The sparse→dense study (paper)

### Algorithms

- `ppo_shaped.py` — `ShapedPPO`, an SB3 `PPO` subclass. It overrides
  `collect_rollouts` to keep raw rewards and an explicit bootstrap array
  (0 at true termination, V̂(terminal_obs) at truncation), applies
  potential-based shaping `r' = r + γΦ(s') − Φ(s)` at rollout time with the
  pre-update network, and computes GAE itself (`gae_advantages`).
  - `shaping_mode="none"` reproduces stock SB3 advantages exactly.
  - `bsrs_coupled`: Φ = η·V̂ (the live GAE critic).
  - `bsrs_decoupled`: Φ = η·V_tgt, a Polyak copy of the policy/value net,
    updated **once per rollout** with τ = 0.01.
- `ppo_her.py` — `HERPPO`, on-policy hindsight relabeling. Failed episodes are
  relabeled with their final achieved end-effector position; target and
  distance features are recomputed on raw observations, then normalized with
  the rollout's obs statistics; V̂ and log π_old are recomputed with the
  pre-update network. Augmentation is capped at 1× the original batch.
  A relabeled reward is the original reward plus +100 at the relabeled goal.
  That keeps the hard task's −10 collision penalty, which does not depend on
  the goal.
  - **Hard-task caveat:** the committed `results_s2d/hard/her_final_*` runs
    predate this fix and used the success term alone, dropping the penalty on
    relabeled transitions. `relabel_goal_independent_reward=False` reproduces
    them. They are to be re-run.
  - Easy-task results are identical under both settings (no collisions).
- `tests_shaping.py` — verification suite (≈5 s): T1 telescoping, T2 coupled
  identity A′ = (1+η)A − ηR^{γλ} including truncations, T3 truncation
  bootstrap, T4 decoupled residuals. All pass at ~1e-15.
- `tests_her.py` — relabeling-reward checks (≈10 s): the collision penalty
  survives relabeling, the legacy flag reproduces the pre-fix behaviour, and
  batches are identical when there are no collisions.

### Conditions → code path → settings

All runs: SB3 2.8.0 `PPO`/subclass, `MlpPolicy` (separate π and V MLPs,
64-64, no shared trainable parameters), `DummyVecEnv` with 1 env,
`VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10)`,
`lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, γ=0.99, λ=0.95,
clip_range=0.2, clip_range_vf=None, ent_coef=0.01, vf_coef=0.5,
max_grad_norm=0.5`, 500,000 steps, seeds `0 1 2 3 4`, easy and hard tasks.

| Condition | Class | `reward_mode` | Shaping | η | `normalize_advantage` |
|---|---|---|---|---|---|
| `sparse` | `ShapedPPO` | sparse | none | — | True |
| `oracle` | `ShapedPPO` | full | none | — | True |
| `bsrs_c_e05` / `_e1` / `_e2` | `ShapedPPO` | sparse | coupled (Φ = ηV̂) | 0.5 / 1 / 2 | True |
| `bsrs_d_e05` / `_e1` / `_e2` | `ShapedPPO` | sparse | decoupled (Φ = ηV_tgt, τ = 0.01/rollout) | 0.5 / 1 / 2 | True |
| `sparse_nonorm` | `ShapedPPO` | sparse | none | — | **False** |
| `bsrs_c_e1_nonorm` | `ShapedPPO` | sparse | coupled | 1 | **False** |
| `her_final` | `HERPPO` | sparse | — (final-state relabeling) | — | True |

The table is generated from `CONDITIONS` and `HP` in `run_s2d_study.py`;
that file is the source of truth.

### Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests_shaping.py                 # identity checks, ~5 s
python tests_her.py                     # HER relabeling-reward checks, ~10 s
python run_s2d_study.py --yes           # 110 runs, ~10 CPU-hours, resumable
python evaluate_s2d.py                  # 50 deterministic episodes/run -> results_s2d/eval_s2d*.csv
python figures_s2d.py                   # -> figures_s2d/F1..F7 + captions.md
```

Per-run outputs live in `results_s2d/{easy,hard}/{condition}_seed{N}/`:
`progress.csv` (per-update diagnostics: value-target magnitude,
pre-normalization advantage stats, nonzero-raw-reward fraction, clip fraction,
entropy, KL), `evaluations.npz`, `first_success.json`, `run_meta.json`.

### Statistics

`s2d_stats.py` computes the interquartile mean (25%-trimmed mean) with a
**percentile bootstrap over seeds** (10,000 resamples, one condition at a
time; there is no task axis to stratify). Cross-checked against `rliable`
1.2.0: all 22 IQM point estimates match exactly, and 21/22 CIs match to 4
decimals. The exception is easy/sparse, where the upper bound falls on a
discrete jump of the n = 5 IQM distribution (0.213 vs 0.233); the asymptotic
value is 0.233. `rliable` needs `pandas<3` to import (its `arch` dependency
calls a private pandas API that changed in pandas 3).

### Diagnostics (post-review analyses, no training unless noted)

Run from the repo root. The first two need the one checkpoint that is tracked,
`results_s2d/easy/sparse_seed0/{final_model.zip, vec_normalize.pkl}`; the
others read `eval_s2d.csv` / `progress.csv`, or the final checkpoints if you
regenerate them.

| Script | Question | Paper |
|---|---|---|
| `diagnostics/p2_update_probe.py` | Does scaling advantages by (1+η) act like a larger step size? Three PPO updates from identical weights, with fresh and with warm (checkpoint-restored) Adam state, plus the gradient decomposition. | §5.2 |
| `diagnostics/p4_fixed_point.py` | Does a coupled-shaping critic converge to V/(1+η)? Frozen policy, fresh critic, η ∈ {0, 0.5, 1, 2}. This one **trains** (20 critic-only runs, ~15 min on 4 cores). Output: `results_p4_fixedpoint/`. | §5.4 |
| `diagnostics/advantage_direction_probe.py` | How far does −ηR^{γλ} rotate the normalized advantage vector in batches that contain a success or collision? | §5.1 |
| `diagnostics/stats_vs_sparse.py` | Paired and unpaired bootstrap on the difference, Welch and Mann–Whitney, every condition vs. sparse. | §5.2, §5.5 |
| `diagnostics/rliable_check.py` | Cross-check of `s2d_stats.py` against rliable (run in an env with `pandas<3`). | §4.3 |

#### Result: the coupled-shaping critic converges to V/(1+η)

`results_p4_fixedpoint/p4_fixed_point_summary.csv`. The slope is a
through-origin least-squares fit of V̂_η on V̂_0 over a fixed set of states
visited by the frozen policy, measured after 300k steps.

| η | predicted 1/(1+η) | frozen policy A: `bsrs_c_e1_seed4`, 3 seeds | frozen policy B: `sparse_seed3`, 2 seeds |
|---|---|---|---|
| 0.5 | 0.667 | 0.667 ± 0.008 | 0.661 ± 0.003 |
| 1 | 0.500 | 0.492 ± 0.008 | 0.491 ± 0.002 |
| 2 | 0.333 | 0.325 ± 0.011 | 0.314 ± 0.001 |

The slopes have settled by 50k steps. The unshaped critic tracks Monte-Carlo
returns (slope 0.99 / 1.05). This explains the value-target deflation reported
in §5.4. The noisier ratios in the 110-run study (0.38 / 0.45 / 0.06) come
from each η having trained a different policy there.

## Earlier reward-ablation study

`run_ablation_study.py` trains PPO under each `reward_mode` (dense terms
ablated), with **`norm_reward=True`**, otherwise the same hyperparameters.
Its results are in `results/` (easy, 2 seeds) and `results_hard/` (hard,
5 seeds). Because reward normalization differs, its numbers are not directly
comparable to the sparse→dense study.

## Standalone training

`train.py --algo {ppo_lag,ppo}` trains a single PPO-Lagrangian or PPO agent on
the dense `full` reward (`norm_reward=True`). `evaluate.py` replays a trained
model, optionally with the MuJoCo viewer.

## Requirements

Python 3.11; see `requirements.txt` (stable-baselines3 ≥ 2.3, gymnasium ≥ 0.29,
mujoco ≥ 3.0). The study was run with SB3 2.8.0, torch 2.11.0, gymnasium 1.2.3,
mujoco 3.7.0, numpy 2.4.4 on an Apple-silicon CPU.

## License

MIT

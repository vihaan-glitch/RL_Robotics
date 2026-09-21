"""
run_s2d_study.py
================
Sparse→dense reward-generation study runner.

Trains, for each condition × task (easy/hard) × 5 seeds, a 500k-step agent with
the SAME PPO hyperparameters as the prior ablation EXCEPT norm_reward=False
(intentional — baselines are re-run under this normalization). All conditions
use reward_mode="sparse" except `oracle` (reward_mode="full", the ceiling).

Conditions (11):
  sparse             PPO, no shaping                     (floor)
  oracle             PPO, reward_mode="full"             (ceiling)
  bsrs_c_e05/_e1/_e2 coupled BSRS,   η ∈ {0.5, 1, 2}
  bsrs_c_e1_nonorm   coupled η=1, normalize_advantage=False
  sparse_nonorm      no shaping, normalize_advantage=False (P2 control)
  bsrs_d_e05/_e1/_e2 decoupled BSRS, η ∈ {0.5, 1, 2}
  her_final          PPO-HER, final relabeling

Full grid = 11 × 2 tasks × 5 seeds = 110 runs. With --trim, {bsrs_c_e2,
bsrs_d_e2} are dropped on HARD only → 100 runs.

Per-run outputs: results_s2d/{task}/{condition}_seed{N}/
  final_model.zip · vec_normalize.pkl · tensorboard/ · evaluations.npz
  progress.csv · first_success.json · run_meta.json

Sequential, resumable (skips runs whose final_model.zip exists), prints
progress + ETA. Usage:
  python run_s2d_study.py                 # prints config, asks to confirm
  python run_s2d_study.py --yes           # skip prompt
  python run_s2d_study.py --trim --yes    # 100-run trimmed grid
"""

import argparse
import json
import os
import time

import numpy as np
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecNormalize, sync_envs_normalization,
)

from arm_reach_env import ArmReachEnv
from ppo_shaped import ShapedPPO, ProgressCSVCallback
from ppo_her import HERPPO

# ── Configuration ─────────────────────────────────────────────────────────────
SUCCESS_DISTANCE = 0.05
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10
RESULTS_ROOT = "results_s2d"
SEEDS_DEFAULT = [0, 1, 2, 3, 4]
TIMESTEPS = 500_000

TASK_OBSTACLE = {"easy": None, "hard": "simple"}

# Hyperparameters identical to the prior study (norm_reward=False set on the env).
HP = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, verbose=0,
)

# cls ∈ {"shaped","her"}. `shaped` covers sparse/oracle/BSRS via shaping_mode.
CONDITIONS = {
    "sparse":           dict(cls="shaped", reward_mode="sparse", shaping_mode="none",          eta=0.0, normalize_advantage=True),
    "oracle":           dict(cls="shaped", reward_mode="full",   shaping_mode="none",          eta=0.0, normalize_advantage=True),
    "bsrs_c_e05":       dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_coupled",  eta=0.5, normalize_advantage=True),
    "bsrs_c_e1":        dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_coupled",  eta=1.0, normalize_advantage=True),
    "bsrs_c_e2":        dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_coupled",  eta=2.0, normalize_advantage=True),
    "bsrs_c_e1_nonorm": dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_coupled",  eta=1.0, normalize_advantage=False),
    "sparse_nonorm":    dict(cls="shaped", reward_mode="sparse", shaping_mode="none",          eta=0.0, normalize_advantage=False),
    "bsrs_d_e05":       dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_decoupled", eta=0.5, normalize_advantage=True),
    "bsrs_d_e1":        dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_decoupled", eta=1.0, normalize_advantage=True),
    "bsrs_d_e2":        dict(cls="shaped", reward_mode="sparse", shaping_mode="bsrs_decoupled", eta=2.0, normalize_advantage=True),
    "her_final":        dict(cls="her",    reward_mode="sparse", her_strategy="final",         normalize_advantage=True),
}
TRIM_DROP_ON_HARD = ["bsrs_c_e2", "bsrs_d_e2"]


# ── Env wrapper: expose is_success so EvalCallback records success-vs-steps ────
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_success"] = bool(info.get("distance", np.inf) < SUCCESS_DISTANCE)
        return obs, reward, terminated, truncated, info


# ── First-success tracker + eval-normalization sync ───────────────────────────
class FirstSuccessCallback(BaseCallback):
    def __init__(self, eval_env, sync_freq=1000, verbose=0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._sync_freq = sync_freq
        self.first_success_timestep = None

    def _on_step(self) -> bool:
        if self.first_success_timestep is None:
            for info in self.locals.get("infos", []):
                if info.get("distance", np.inf) < SUCCESS_DISTANCE:
                    self.first_success_timestep = int(self.num_timesteps)
                    break
        if self.num_timesteps % self._sync_freq == 0:
            try:
                sync_envs_normalization(self.training_env, self._eval_env)
            except Exception:
                if hasattr(self.training_env, "obs_rms"):
                    self._eval_env.obs_rms = self.training_env.obs_rms
        return True


# ── Single run ────────────────────────────────────────────────────────────────
def run_dir(task, condition, seed):
    return os.path.join(RESULTS_ROOT, task, f"{condition}_seed{seed}")


def is_complete(task, condition, seed):
    return os.path.exists(os.path.join(run_dir(task, condition, seed), "final_model.zip"))


def train_one(task, condition, seed):
    spec = CONDITIONS[condition]
    obstacle = TASK_OBSTACLE[task]
    out_dir = run_dir(task, condition, seed)
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    progress_csv = os.path.join(out_dir, "progress.csv")
    set_random_seed(seed)

    def make_env():
        return SuccessInfoWrapper(
            ArmReachEnv(render_mode=None, reward_mode=spec["reward_mode"],
                        obstacle_mode=obstacle))

    train_env = VecNormalize(DummyVecEnv([make_env]),
                             norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_env.seed(seed)
    eval_env = VecNormalize(DummyVecEnv([make_env]),
                            norm_obs=True, norm_reward=False, training=False)
    eval_env.seed(seed + 10_000)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=None, log_path=out_dir,
        eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True, render=False, verbose=0,
    )
    first_cb = FirstSuccessCallback(eval_env, sync_freq=1000)
    callbacks = [first_cb, eval_cb]

    if spec["cls"] == "her":
        model = HERPPO("MlpPolicy", train_env, her_strategy=spec["her_strategy"],
                       progress_csv=progress_csv, normalize_advantage=spec["normalize_advantage"],
                       tensorboard_log=tb_dir, seed=seed, **HP)
    else:
        model = ShapedPPO("MlpPolicy", train_env, shaping_mode=spec["shaping_mode"],
                          eta=spec["eta"], target_tau=0.01,
                          normalize_advantage=spec["normalize_advantage"],
                          tensorboard_log=tb_dir, seed=seed, **HP)
        callbacks.append(ProgressCSVCallback(progress_csv))

    model.learn(total_timesteps=TIMESTEPS, callback=callbacks, progress_bar=True)

    model.save(os.path.join(out_dir, "final_model"))
    train_env.save(os.path.join(out_dir, "vec_normalize.pkl"))
    with open(os.path.join(out_dir, "first_success.json"), "w") as f:
        json.dump({"task": task, "condition": condition, "seed": seed,
                   "success_distance": SUCCESS_DISTANCE,
                   "first_success_timestep": first_cb.first_success_timestep}, f, indent=2)
    train_env.close()
    eval_env.close()


# ── Orchestration ─────────────────────────────────────────────────────────────
def fmt_duration(s):
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def build_plan(tasks, conditions, seeds, trim):
    plan = []
    for task in tasks:
        for cond in conditions:
            if trim and task == "hard" and cond in TRIM_DROP_ON_HARD:
                continue
            for seed in seeds:
                plan.append((task, cond, seed))
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["easy", "hard"], choices=["easy", "hard"])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    p.add_argument("--trim", action="store_true",
                   help="drop bsrs_c_e2, bsrs_d_e2 on hard only (110 → 100 runs)")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    plan = build_plan(args.tasks, args.conditions, args.seeds, args.trim)
    total = len(plan)
    todo = [x for x in plan if not is_complete(*x)]
    done = total - len(todo)

    line = "=" * 66
    print(f"\n{line}\n  SPARSE→DENSE STUDY — configuration\n{line}")
    print(f"  Tasks       : {', '.join(args.tasks)}")
    print(f"  Conditions  : {len(args.conditions)}  ({', '.join(args.conditions)})")
    print(f"  Seeds       : {', '.join(map(str, args.seeds))}")
    print(f"  Timesteps   : {TIMESTEPS:,}/run   norm_reward=False")
    print(f"  Trim        : {'ON (drop %s on hard)' % TRIM_DROP_ON_HARD if args.trim else 'off'}")
    print(f"  Total runs  : {total}  ({done} complete, {len(todo)} to run)")
    print(f"  Output root : {RESULTS_ROOT}/{{task}}/{{condition}}_seed{{N}}/")
    print(line)
    if not todo:
        print("  Nothing to do — all requested runs complete.\n")
        return
    if not args.yes:
        try:
            ans = input(f"\n  Proceed with {len(todo)} runs (est. 6–8 h)? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("  Aborted — no training started.\n")
            return

    durations, started = [], time.time()
    for i, (task, cond, seed) in enumerate(plan, 1):
        if is_complete(task, cond, seed):
            print(f"\nRun {i}/{total}: {task}/{cond} seed{seed} — complete, skipping.")
            continue
        if durations:
            remaining = sum(1 for j, x in enumerate(plan, 1) if j >= i and not is_complete(*x))
            eta = f"  | ETA {fmt_duration(sum(durations)/len(durations)*remaining)}"
        else:
            eta = "  | ETA (after first run)"
        print(f"\n{'-'*66}\nRun {i}/{total}: {task}/{cond} seed{seed}{eta}\n{'-'*66}")
        t0 = time.time()
        train_one(task, cond, seed)
        dur = time.time() - t0
        durations.append(dur)
        with open(os.path.join(run_dir(task, cond, seed), "run_meta.json"), "w") as f:
            json.dump({"task": task, "condition": cond, "seed": seed,
                       "timesteps": TIMESTEPS, "duration_seconds": round(dur, 1),
                       **{k: v for k, v in CONDITIONS[cond].items() if k != "cls"}}, f, indent=2)
        print(f"  ✓ {task}/{cond} seed{seed} in {fmt_duration(dur)}")

    print(f"\n{line}\n  All runs complete. Wall-clock: {fmt_duration(time.time()-started)}\n{line}\n")


if __name__ == "__main__":
    main()

"""
evaluate_s2d.py
===============
Evaluate every trained sparse→dense run and aggregate with IQM + bootstrap CIs.

For each run under results_s2d/{task}/{condition}_seed{N}/ with a final_model.zip:
  * 50 deterministic eval episodes (matching evaluate.py protocol).
  * Records success rate, mean reward, mean length, mean final distance, and the
    first-success timestep (read from first_success.json).

Outputs:
  results_s2d/eval_s2d.csv             one row per run (seed-level).
  results_s2d/eval_s2d_aggregated.csv  per (task, condition): IQM ± 95% bootstrap
                                       CI of success rate (and IQM of the rest).

Usage:  python evaluate_s2d.py [--episodes 50]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv
from run_s2d_study import (CONDITIONS, TASK_OBSTACLE, RESULTS_ROOT,
                           SuccessInfoWrapper, SUCCESS_DISTANCE)
from s2d_stats import iqm, bootstrap_ci

EVAL_CSV = os.path.join(RESULTS_ROOT, "eval_s2d.csv")
AGG_CSV = os.path.join(RESULTS_ROOT, "eval_s2d_aggregated.csv")


def evaluate_run(task, condition, seed, n_episodes):
    spec = CONDITIONS[condition]
    out_dir = os.path.join(RESULTS_ROOT, task, f"{condition}_seed{seed}")
    model_path = os.path.join(out_dir, "final_model")
    if not os.path.exists(model_path + ".zip"):
        return None

    def make_env():
        return SuccessInfoWrapper(ArmReachEnv(
            render_mode=None, reward_mode=spec["reward_mode"],
            obstacle_mode=TASK_OBSTACLE[task]))

    env = DummyVecEnv([make_env])
    norm_path = os.path.join(out_dir, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    env.seed(seed + 50_000)
    model = PPO.load(model_path, env=env)

    rewards, lengths, final_d, succ = [], [], [], []
    obs = env.reset()
    for _ in range(n_episodes):
        ep_r, ep_l, done, last_d = 0.0, 0, False, np.inf
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, info = env.step(action)
            ep_r += float(r[0]); ep_l += 1
            last_d = float(info[0].get("distance", np.inf))
            if d[0]:
                done = True
                rewards.append(ep_r); lengths.append(ep_l); final_d.append(last_d)
                succ.append(last_d < SUCCESS_DISTANCE)
                obs = env.reset()
    env.close()

    fs = None
    fs_path = os.path.join(out_dir, "first_success.json")
    if os.path.exists(fs_path):
        fs = json.load(open(fs_path)).get("first_success_timestep")

    return {
        "task": task, "condition": condition, "seed": seed, "n_episodes": n_episodes,
        "success_rate": float(np.mean(succ)),
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_final_distance": float(np.mean(final_d)),
        "first_success_timestep": fs if fs is not None else np.nan,
    }


def discover_runs():
    runs = []
    for task in TASK_OBSTACLE:
        for d in sorted(glob.glob(os.path.join(RESULTS_ROOT, task, "*_seed*"))):
            if not os.path.exists(os.path.join(d, "final_model.zip")):
                continue
            base = os.path.basename(d)
            cond, seed = base.rsplit("_seed", 1)
            if cond in CONDITIONS and seed.isdigit():
                runs.append((task, cond, int(seed)))
    return runs


def aggregate(df):
    rows = []
    for (task, cond), g in df.groupby(["task", "condition"]):
        sr = g["success_rate"].values
        pt, lo, hi = bootstrap_ci(sr, agg=iqm)
        rows.append({
            "task": task, "condition": cond, "n_seeds": len(g),
            "success_iqm": pt, "success_ci_lo": lo, "success_ci_hi": hi,
            "success_mean": float(np.mean(sr)), "success_std": float(np.std(sr)),
            "reward_iqm": iqm(g["mean_reward"].values),
            "final_distance_iqm": iqm(g["mean_final_distance"].values),
            "first_success_iqm": iqm(g["first_success_timestep"].dropna().values)
            if g["first_success_timestep"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    runs = discover_runs()
    if not runs:
        print(f"No trained runs found under {RESULTS_ROOT}/. Run run_s2d_study.py first.")
        return
    print(f"Evaluating {len(runs)} runs ({args.episodes} episodes each)...")
    rows = []
    for i, (task, cond, seed) in enumerate(runs, 1):
        res = evaluate_run(task, cond, seed, args.episodes)
        if res is None:
            continue
        rows.append(res)
        print(f"  [{i}/{len(runs)}] {task}/{cond} seed{seed}: "
              f"success={res['success_rate']:.0%} reward={res['mean_reward']:.1f}")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    df.to_csv(EVAL_CSV, index=False)
    agg = aggregate(df)
    agg.to_csv(AGG_CSV, index=False)
    print(f"\nWrote {EVAL_CSV} ({len(df)} rows) and {AGG_CSV} ({len(agg)} groups).")


if __name__ == "__main__":
    main()

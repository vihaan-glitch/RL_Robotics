"""
p4_fixed_point.py — controlled test of the coupled-shaping critic fixed point.

Claim under test (P4, corrected): with coupled shaping Φ = η·V̂, the critic
regresses onto the shaped return, whose value telescopes to V^π(s) − η·V̂(s).
Self-consistency V̂ = V^π − η·V̂ gives the fixed point

        V̂_η(s) = V^π(s) / (1 + η).

The 110-run study can only compare value targets across η for *different*
policies (each η trained its own policy), so that comparison is confounded.
Here the policy is frozen and only a freshly initialised critic is trained:

  * policy-side parameters (log_std, mlp_extractor.policy_net.*, action_net.*)
    are copied from a trained easy-task checkpoint and frozen;
  * value-side parameters (mlp_extractor.value_net.*, value_net.*) start from
    the same fresh initialisation for every η at a given seed;
  * the potential is the critic being trained (bsrs_coupled), so the fixed
    point is the self-referential one above;
  * observation normalisation is the checkpoint's, frozen, so every critic
    sees identical inputs.

After 50k/100k/200k/300k steps each critic is evaluated on one fixed set of
states visited by the frozen policy. We report the through-origin regression
slope of V̂_η on V̂_0 (prediction 1/(1+η)) and on Monte-Carlo returns G
(prediction 1/(1+η) times the η=0 slope).

Usage (from the repo root):
    python diagnostics/p4_fixed_point.py run      # ~20 runs, parallel
    python diagnostics/p4_fixed_point.py analyze  # tables -> results_p4_fixedpoint/
"""
import copy
import itertools
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "results_p4_fixedpoint"
ETAS = (0.0, 0.5, 1.0, 2.0)
CHUNKS = (50_000, 100_000, 200_000, 300_000)      # cumulative evaluation points
POLICIES = {
    # name: (checkpoint dir, seeds)
    "c_e1_s4":  ("results_s2d/easy/bsrs_c_e1_seed4", (0, 1, 2)),   # 80% eval success
    "sparse_s3": ("results_s2d/easy/sparse_seed3",   (0, 1)),      # 26% eval success
}
N_STATE_EPISODES = 40
N_STATES = 5000
GAMMA = 0.99
MC_TRUNC_KEEP = 200          # truncated episodes: keep only t <= 200 (γ^300 ≈ 0.05)
N_WORKERS = 4


def _hp():
    from run_s2d_study import HP
    return {k: v for k, v in HP.items() if k != "verbose"}


def _env(ckpt_dir, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from arm_reach_env import ArmReachEnv
    from run_s2d_study import SuccessInfoWrapper
    env = DummyVecEnv([lambda: SuccessInfoWrapper(
        ArmReachEnv(render_mode=None, reward_mode="sparse", obstacle_mode=None))])
    env = VecNormalize.load(os.path.join(ckpt_dir, "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    env.seed(seed)
    return env


def _is_value_param(name):
    return "value_net" in name


def _build(ckpt_dir, eta, seed, env):
    """Frozen checkpoint policy + fresh (seeded) critic, coupled shaping at η."""
    import torch as th
    from ppo_shaped import ShapedPPO
    loaded = ShapedPPO.load(os.path.join(ckpt_dir, "final_model"), env=env)
    src = loaded.policy.state_dict()
    mode = "none" if eta == 0.0 else "bsrs_coupled"
    m = ShapedPPO("MlpPolicy", env, shaping_mode=mode, eta=eta, seed=seed,
                  verbose=0, **_hp())
    # Same fresh critic init for every η at this seed (the seed fixes the init).
    policy_side = {k: v for k, v in src.items() if not _is_value_param(k)}
    missing, unexpected = m.policy.load_state_dict(policy_side, strict=False)
    assert not unexpected and all(_is_value_param(k) for k in missing), (missing, unexpected)
    for name, p in m.policy.named_parameters():
        p.requires_grad_(_is_value_param(name))
    m.policy.optimizer = th.optim.Adam(
        [p for p in m.policy.parameters() if p.requires_grad], lr=_hp()["learning_rate"], eps=1e-5)
    return m


def _state_set(ckpt_dir):
    """Fixed evaluation states (normalised obs) + Monte-Carlo returns, frozen policy."""
    path = os.path.join(OUT, f"states_{os.path.basename(ckpt_dir)}.npz")
    if os.path.exists(path):
        return path
    import torch as th
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from arm_reach_env import ArmReachEnv
    from ppo_shaped import ShapedPPO
    vn = _env(ckpt_dir, 12345)
    model = ShapedPPO.load(os.path.join(ckpt_dir, "final_model"), env=vn)
    raw = ArmReachEnv(render_mode=None, reward_mode="sparse", obstacle_mode=None)
    rng = np.random.default_rng(0)
    obs_all, g_all = [], []
    n_succ = 0
    for ep in range(N_STATE_EPISODES):
        o, _ = raw.reset(seed=10_000 + ep)
        traj_o, traj_r, trunc = [], [], False
        while True:
            no = vn.normalize_obs(o[None].astype(np.float32))
            a, _ = model.predict(no, deterministic=False)
            o, r, term, tr, _ = raw.step(a[0])
            traj_o.append(no[0]); traj_r.append(r)
            if term:
                n_succ += 1
                break
            if tr:
                trunc = True
                break
        g, G = 0.0, np.zeros(len(traj_r))
        for t in reversed(range(len(traj_r))):
            g = traj_r[t] + GAMMA * g
            G[t] = g
        keep = np.arange(len(traj_r)) if not trunc else np.arange(min(len(traj_r), MC_TRUNC_KEEP + 1))
        obs_all.append(np.asarray(traj_o)[keep]); g_all.append(G[keep])
    obs_all = np.concatenate(obs_all); g_all = np.concatenate(g_all)
    idx = rng.choice(len(obs_all), size=min(N_STATES, len(obs_all)), replace=False)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, obs=obs_all[idx].astype(np.float32), mc_return=g_all[idx],
             episodes=N_STATE_EPISODES, successes=n_succ)
    vn.close()
    return path


def _one(job):
    import torch as th
    th.set_num_threads(1)
    pol, eta, seed = job
    ckpt_dir = POLICIES[pol][0]
    out = os.path.join(OUT, f"{pol}_eta{eta}_seed{seed}.npz")
    if os.path.exists(out):
        return out, 0.0
    t0 = time.time()
    states = np.load(os.path.join(OUT, f"states_{os.path.basename(ckpt_dir)}.npz"))
    obs_t = th.as_tensor(states["obs"])
    env = _env(ckpt_dir, seed)
    m = _build(ckpt_dir, eta, seed, env)
    frozen0 = {k: v.clone() for k, v in m.policy.state_dict().items() if not _is_value_param(k)}
    preds, done = [], 0
    for i, target in enumerate(CHUNKS):
        m.learn(total_timesteps=target - done, reset_num_timesteps=(i == 0))
        done = target
        with th.no_grad():
            preds.append(m.policy.predict_values(obs_t).cpu().numpy().ravel())
    for k, v in m.policy.state_dict().items():
        if not _is_value_param(k):
            assert th.equal(v, frozen0[k]), f"policy param {k} changed"
    np.savez(out, preds=np.asarray(preds), chunks=np.asarray(CHUNKS), eta=eta, seed=seed)
    env.close()
    return out, time.time() - t0


def run():
    os.makedirs(OUT, exist_ok=True)
    for pol, (ckpt, _) in POLICIES.items():
        p = _state_set(ckpt)
        s = np.load(p)
        print(f"[states] {pol}: {len(s['obs'])} states, {int(s['successes'])}/{int(s['episodes'])} "
              f"episodes succeeded, mean MC return {s['mc_return'].mean():.2f}", flush=True)
    jobs = [(pol, eta, seed) for pol, (_, seeds) in POLICIES.items()
            for seed in seeds for eta in ETAS]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(N_WORKERS) as pool:
        for k, (out, dt) in enumerate(pool.imap_unordered(_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {os.path.basename(out)}  {dt/60:.1f} min  "
                  f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


def _slope0(x, y):
    """Least-squares slope of y on x through the origin."""
    return float(np.dot(x, y) / np.dot(x, x))


def analyze():
    rows = []
    for pol, (ckpt, seeds) in POLICIES.items():
        st = np.load(os.path.join(OUT, f"states_{os.path.basename(ckpt)}.npz"))
        G = st["mc_return"]
        for seed in seeds:
            base = np.load(os.path.join(OUT, f"{pol}_eta0.0_seed{seed}.npz"))["preds"]
            for eta in ETAS:
                P = np.load(os.path.join(OUT, f"{pol}_eta{eta}_seed{seed}.npz"))["preds"]
                for ci, steps in enumerate(CHUNKS):
                    v, v0 = P[ci], base[ci]
                    rows.append(dict(policy=pol, seed=seed, eta=eta, steps=steps,
                                     slope_vs_v0=_slope0(v0, v),
                                     r2_vs_v0=float(np.corrcoef(v0, v)[0, 1] ** 2),
                                     ratio_mean_abs=float(np.mean(np.abs(v)) / np.mean(np.abs(v0))),
                                     slope_vs_mc=_slope0(G, v),
                                     predicted=1.0 / (1.0 + eta)))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p4_fixed_point_all.csv"), index=False)
    fin = df[df.steps == CHUNKS[-1]]
    summ = (fin.groupby(["policy", "eta"])
               .agg(n_seeds=("seed", "nunique"),
                    slope_vs_v0=("slope_vs_v0", "mean"), slope_vs_v0_sd=("slope_vs_v0", "std"),
                    ratio_mean_abs=("ratio_mean_abs", "mean"),
                    slope_vs_mc=("slope_vs_mc", "mean"), r2_vs_v0=("r2_vs_v0", "mean"),
                    predicted=("predicted", "first"))
               .reset_index())
    summ.to_csv(os.path.join(OUT, "p4_fixed_point_summary.csv"), index=False)
    conv = (df.groupby(["policy", "eta", "steps"]).slope_vs_v0.mean().unstack("steps"))
    pd.set_option("display.width", 160)
    print("Final (300k steps), mean over seeds:\n", summ.round(4).to_string(index=False))
    print("\nConvergence of slope_vs_v0 (mean over seeds):\n", conv.round(4).to_string())
    json.dump({"etas": ETAS, "chunks": CHUNKS, "policies": {k: list(v[1]) for k, v in POLICIES.items()}},
              open(os.path.join(OUT, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()

"""
p2_update_probe.py — does scaling advantages by (1+η) act like a larger step size?

Paper §5.2. Starting from the easy-task sparse seed-0 checkpoint, collect one
real 2048-step rollout, then apply PPO updates from three identical starting
states that differ only in:

  (a) baseline   advantages = A           lr = 3e-4
  (b) adv-scaled advantages = (1+η)·A     lr = 3e-4       (normalize_advantage=False)
  (c) lr-scaled  advantages = A           lr = (1+η)·3e-4

Each update replicates stock SB3 PPO.train() for one minibatch:
    loss = policy_loss + ent_coef·entropy_loss + vf_coef·value_loss,
    then clip_grad_norm_(0.5) and Adam.

Two optimizer regimes:
  fresh  — Adam moments zeroed (the first-step regime);
  warm   — Adam state restored from the checkpoint (SB3 saves `policy.optimizer`
           in the zip: 78,400 steps of accumulated moments at the end of training).

Reported: ‖Δθ‖ after 1 and 32 consecutive minibatch updates, the split between
policy- and value-side parameters, cosine similarities, and the gradient
decomposition that shows how much of the pre-clip gradient the policy loss owns.

Usage (from the repo root; needs results_s2d/easy/sparse_seed0/{final_model.zip,vec_normalize.pkl}):
    python diagnostics/p2_update_probe.py
"""
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv
from ppo_shaped import ShapedPPO
from run_s2d_study import SuccessInfoWrapper, HP

RUN = "results_s2d/easy/sparse_seed0"
ETA = 1.0
LR, ENT, VF, MAXG, CLIP, BS = (HP["learning_rate"], HP["ent_coef"], HP["vf_coef"],
                               HP["max_grad_norm"], HP["clip_range"], HP["batch_size"])
K = 32   # one epoch of minibatches


def make_env():
    env = DummyVecEnv([lambda: SuccessInfoWrapper(
        ArmReachEnv(render_mode=None, reward_mode="sparse", obstacle_mode=None))])
    env = VecNormalize.load(os.path.join(RUN, "vec_normalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    return env


def build(env, lr, sd, opt_sd):
    hp = {k: v for k, v in HP.items() if k not in ("verbose", "learning_rate")}
    m = ShapedPPO("MlpPolicy", env, shaping_mode="none", learning_rate=lr,
                  normalize_advantage=False, seed=0, verbose=0, **hp)
    m.policy.load_state_dict(copy.deepcopy(sd))
    m.policy.optimizer = th.optim.Adam(m.policy.parameters(), lr=lr, eps=1e-5)
    if opt_sd is not None:
        m.policy.optimizer.load_state_dict(copy.deepcopy(opt_sd))
        for g in m.policy.optimizer.param_groups:
            g["lr"] = lr
    return m


def flat(policy):
    return th.cat([p.detach().reshape(-1) for p in policy.parameters()])


def losses(m, obs, act, old_lp, adv, ret):
    values, log_prob, entropy = m.policy.evaluate_actions(obs, act)
    ratio = th.exp(log_prob - old_lp)
    pl = -th.min(adv * ratio, adv * th.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
    vl = F.mse_loss(ret, values.flatten())
    el = -th.mean(entropy)
    return pl, vl, el


def step(m, obs, act, old_lp, adv, ret):
    m.policy.set_training_mode(True)
    pl, vl, el = losses(m, obs, act, old_lp, adv, ret)
    m.policy.optimizer.zero_grad()
    (pl + ENT * el + VF * vl).backward()
    th.nn.utils.clip_grad_norm_(m.policy.parameters(), MAXG)
    m.policy.optimizer.step()


def grad_norm(m, loss):
    m.policy.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    return th.sqrt(sum((p.grad ** 2).sum() for p in m.policy.parameters() if p.grad is not None)).item()


def main():
    env = make_env()
    loaded = ShapedPPO.load(os.path.join(RUN, "final_model"), env=env)
    sd = copy.deepcopy(loaded.policy.state_dict())
    opt_sd = copy.deepcopy(loaded.policy.optimizer.state_dict())
    adam_steps = {int(v["step"]) for v in opt_sd["state"].values()}

    ref = build(env, LR, sd, None)
    _, cb = ref._setup_learn(HP["n_steps"])
    ref.collect_rollouts(ref.env, cb, ref.rollout_buffer, HP["n_steps"])
    buf = ref.rollout_buffer
    O = th.as_tensor(buf.observations.reshape(-1, buf.observations.shape[-1]))
    A_ = th.as_tensor(buf.actions.reshape(-1, buf.actions.shape[-1]))
    LP = th.as_tensor(buf.log_probs.reshape(-1))
    ADV = th.as_tensor(buf.advantages.reshape(-1))
    RET = th.as_tensor(buf.returns.reshape(-1))
    raw = ref._last_rollout_diag["raw_rewards"]
    vmask = th.cat([th.full((p.numel(),), "value_net" in n) for n, p in ref.policy.named_parameters()])
    print(f"rollout: nonzero raw-reward fraction = {np.mean(raw != 0):.6f} "
          f"(0 ⇒ R^γλ ≡ 0, so coupled A′ = (1+η)A exactly on this batch)")
    print(f"parameters: {vmask.numel()} total, {int(vmask.sum())} value-side, "
          f"{int((~vmask).sum())} policy-side, 0 shared")

    sl = slice(0, BS)
    m = build(env, LR, sd, None)
    pl, vl, el = losses(m, O[sl], A_[sl], LP[sl], ADV[sl], RET[sl])
    tot = grad_norm(m, pl + ENT * el + VF * vl)
    print(f"\ngradient norms, first minibatch: policy_loss {grad_norm(m, pl):.3f} | "
          f"vf_coef·value_loss {grad_norm(m, VF * vl):.3f} | ent_coef·entropy {grad_norm(m, ENT * el):.4f} | "
          f"total {tot:.3f} → clipped {tot / MAXG:.1f}× to {MAXG}")

    cos = lambda u, v: (th.dot(u, v) / (u.norm() * v.norm() + 1e-12)).item()
    for regime, o in (("fresh", None), ("warm", opt_sd)):
        label = "fresh Adam (zeroed moments)" if o is None else f"warm Adam (checkpoint state, step {adam_steps})"
        runs = {"a": (build(env, LR, sd, o), 1.0),
                "b": (build(env, LR, sd, o), 1.0 + ETA),
                "c": (build(env, LR * (1 + ETA), sd, o), 1.0)}
        th0 = flat(runs["a"][0].policy).clone()
        traj = {k: [] for k in runs}
        for k in range(K):
            s = slice(k * BS, (k + 1) * BS)
            for name, (mm, scale) in runs.items():
                step(mm, O[s], A_[s], LP[s], ADV[s] * scale, RET[s])
                traj[name].append(flat(mm.policy) - th0)
        print(f"\n=== {label} ===")
        print(f"{'steps':>6} {'‖Δ‖ a':>11} {'‖Δ‖ b':>11} {'‖Δ‖ c':>11} {'b/a':>7} {'c/a':>7} "
              f"{'‖b−a‖/‖a‖':>10} {'‖b−c‖/‖a‖':>10} {'cos(b,a)':>9} {'cos(c,a)':>9}")
        for k in (0, 3, 15, 31):
            a, b, c = traj["a"][k], traj["b"][k], traj["c"][k]
            print(f"{k + 1:>6} {a.norm():>11.4e} {b.norm():>11.4e} {c.norm():>11.4e} "
                  f"{(b.norm() / a.norm()).item():>7.4f} {(c.norm() / a.norm()).item():>7.4f} "
                  f"{((b - a).norm() / a.norm()).item():>10.4f} {((b - c).norm() / a.norm()).item():>10.4f} "
                  f"{cos(b, a):>9.5f} {cos(c, a):>9.5f}")
        a, b = traj["a"][-1], traj["b"][-1]
        print(f"  after {K}: policy-side ‖b−a‖/‖a‖ = {((b - a)[~vmask].norm() / a[~vmask].norm()).item():.4f}, "
              f"value-side = {((b - a)[vmask].norm() / a[vmask].norm()).item():.4f}")
    env.close()


if __name__ == "__main__":
    main()

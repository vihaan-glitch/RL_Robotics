"""
ppo_her.py
==========
On-policy PPO with Hindsight Experience Replay (final-state relabeling) for the
Kuka sparse reaching task, in the style of on-policy HER (Crowder et al., 2024).

The reaching goal is encoded in the observation itself:
    obs = [ joint_pos(7) | joint_vel(7) | target(3) | ee(3) | distance(1) ]
so relabeling a transition = overwriting the target dims with an achieved
end-effector position g′, recomputing the distance feature, and recomputing the
sparse reward (+100 & terminal when ‖ee − g′‖ < 0.05).

Why a custom collect_rollouts + train
-------------------------------------
On-policy HER augments each rollout with relabeled trajectories BEFORE the PPO
update, so GAE and the update run on the combined batch. That requires:
  * raw (un-normalized) observations to relabel on (the buffer stores normalized
    obs, and the target/distance features are derived — see _build_relabeled_batch);
  * recomputing V̂(s,g′) and π_old(a|s,g′) with the *pre-update* network so the
    PPO ratio starts at 1 on relabeled transitions;
  * a variable-size combined batch (original + relabeled), which SB3's fixed
    RolloutBuffer cannot hold — hence a reimplemented PPO update loop that also
    lets us split clip-fraction by transition origin.

her_strategy ∈ {"final", "future_k4"}:
  final      one relabeled trajectory per failed episode; goal = achieved ee at
             the last step.
  future_k4  up to 4 relabeled sub-trajectories per failed episode; each uses a
             goal sampled from a uniformly-chosen future step f and ends (success)
             at f. Sub-trajectory framing keeps GAE well-defined on-policy while
             realizing the "goal from a sampled later step" semantics.

Augmentation is capped at ≤ 1× the original batch size.
All runs use VecNormalize(norm_obs=True, norm_reward=False).
"""

from __future__ import annotations

import csv
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

SUCCESS_DISTANCE = 0.05
VALID_HER_STRATEGIES = ("final", "future_k4")


class HERPPO(PPO):
    """PPO with on-policy hindsight relabeling of failed episodes."""

    def __init__(self, *args, her_strategy: str = "final",
                 progress_csv: str | None = None, **kwargs):
        if her_strategy not in VALID_HER_STRATEGIES:
            raise ValueError(
                f"Unknown her_strategy={her_strategy!r}. "
                f"Expected one of {VALID_HER_STRATEGIES}."
            )
        self.her_strategy = her_strategy
        self.her_k = 4 if her_strategy == "future_k4" else 1
        self._progress_csv_path = progress_csv
        self._progress_file = None
        self._progress_writer = None
        # Filled by collect_rollouts, consumed by train().
        self._combined = None
        super().__init__(*args, **kwargs)
        # Observation-layout indices (see module docstring).
        n_act = self.action_space.shape[0]
        base = 2 * n_act
        self._tgt_sl = slice(base, base + 3)      # target dims
        self._ee_sl = slice(base + 3, base + 6)   # achieved ee dims
        self._dist_i = base + 6                    # distance feature

    def _excluded_save_params(self):
        # Open file handle / csv writer / large scratch batch — not picklable/needed.
        return super()._excluded_save_params() + [
            "_progress_file", "_progress_writer", "_combined",
        ]

    # ══════════════════════════════════════════════════════════════════════════
    #  Rollout collection (keeps raw obs; builds combined relabeled batch)
    # ══════════════════════════════════════════════════════════════════════════
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        T, n_envs = n_rollout_steps, env.num_envs
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        vecnorm = self.get_vec_normalize_env()

        obs_norm = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
        raw_obs = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
        actions_a = np.zeros((T, n_envs, act_dim), dtype=np.float32)
        rewards_a = np.zeros((T, n_envs), dtype=np.float64)
        values_a = np.zeros((T, n_envs), dtype=np.float64)
        logp_a = np.zeros((T, n_envs), dtype=np.float64)
        dones_a = np.zeros((T, n_envs), dtype=np.float64)
        trunc_a = np.zeros((T, n_envs), dtype=bool)
        term_val = {}         # (t,e) -> V̂(terminal_obs) for truncations
        term_raw = {}         # (t,e) -> raw terminal observation

        n_steps = 0
        while n_steps < T:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()
            clipped = actions_np
            if isinstance(self.action_space, spaces.Box):
                clipped = np.clip(actions_np, self.action_space.low, self.action_space.high)

            # Raw obs of the CURRENT state s_t (for relabeling), pre-step.
            raw_now = (vecnorm.get_original_obs() if vecnorm is not None
                       else self._last_obs)
            obs_norm[n_steps] = self._last_obs
            raw_obs[n_steps] = raw_now

            new_obs, rewards, dones, infos = env.step(clipped)
            self.num_timesteps += n_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)

            actions_a[n_steps] = actions_np
            rewards_a[n_steps] = rewards
            values_a[n_steps] = values.cpu().numpy().flatten()
            logp_a[n_steps] = log_probs.cpu().numpy().flatten()
            dones_a[n_steps] = dones.astype(np.float64)

            for e, done in enumerate(dones):
                if not done:
                    continue
                truncated = infos[e].get("TimeLimit.truncated", False)
                trunc_a[n_steps, e] = bool(truncated)
                tobs = infos[e].get("terminal_observation")
                if tobs is not None:
                    tobs = np.asarray(tobs, dtype=np.float32)
                    term_raw[(n_steps, e)] = (
                        vecnorm.unnormalize_obs(tobs) if vecnorm is not None else tobs
                    )
                    if truncated:
                        with th.no_grad():
                            tv = self.policy.predict_values(
                                self.policy.obs_to_tensor(tobs)[0]
                            ).flatten()[0].item()
                        term_val[(n_steps, e)] = tv

            # Standard buffer add (raw reward; used only for bookkeeping/callbacks).
            rollout_buffer.add(self._last_obs, actions_np, rewards,
                               self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        with th.no_grad():
            last_values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            ).cpu().numpy().flatten()

        # ── Original-batch GAE (bootstrap at truncation, 0 at success) ─────────
        from ppo_shaped import gae_advantages
        next_values = np.zeros((T, n_envs), dtype=np.float64)
        for t in range(T):
            for e in range(n_envs):
                if dones_a[t, e]:
                    next_values[t, e] = term_val.get((t, e), 0.0)  # 0 at success
                elif t < T - 1:
                    next_values[t, e] = values_a[t + 1, e]
                else:
                    next_values[t, e] = last_values[e]
        adv_orig = gae_advantages(rewards_a, values_a, next_values, dones_a,
                                  self.gamma, self.gae_lambda)
        ret_orig = adv_orig + values_a

        # ── Build the relabeled augmentation ───────────────────────────────────
        n_original = T * n_envs
        relabeled = self._build_relabeled_batch(
            raw_obs, actions_a, dones_a, trunc_a, term_raw, cap=n_original
        )

        # ── Flatten original + relabeled into one combined batch ───────────────
        flat = lambda a: a.reshape(-1, *a.shape[2:])
        combined = {
            "obs": np.concatenate([flat(obs_norm), relabeled["obs"]], axis=0),
            "actions": np.concatenate([flat(actions_a), relabeled["actions"]], axis=0),
            "old_values": np.concatenate([values_a.reshape(-1), relabeled["values"]]),
            "old_logp": np.concatenate([logp_a.reshape(-1), relabeled["logp"]]),
            "advantages": np.concatenate([adv_orig.reshape(-1), relabeled["adv"]]),
            "returns": np.concatenate([ret_orig.reshape(-1), relabeled["ret"]]),
            "is_relabeled": np.concatenate([
                np.zeros(n_original, dtype=bool),
                np.ones(len(relabeled["adv"]), dtype=bool),
            ]),
            "n_original": n_original,
            "raw_rewards_orig": rewards_a.reshape(-1),
        }
        self._combined = combined

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ══════════════════════════════════════════════════════════════════════════
    #  THE RELABELING CODE PATH
    # ══════════════════════════════════════════════════════════════════════════
    def _build_relabeled_batch(self, raw_obs, actions_a, dones_a, trunc_a,
                               term_raw, cap: int) -> dict:
        """
        For every COMPLETED, FAILED (truncated) episode, produce relabeled
        trajectories, recompute V̂/logp with the pre-update net, GAE each, and
        return flattened arrays. Augmentation stops once `cap` transitions built.
        """
        T, n_envs, obs_dim = raw_obs.shape
        out_obs, out_act = [], []
        out_val, out_logp, out_adv, out_ret = [], [], [], []
        built = 0

        for e in range(n_envs):
            start = 0
            for t in range(T):
                if not dones_a[t, e]:
                    continue
                end = t
                failed = bool(trunc_a[t, e])  # success = terminated, not truncated
                if failed and (end, e) in term_raw:
                    # Achieved ee AFTER each transition τ in [start..end]:
                    # next-state ee, i.e. raw_obs[τ+1] within the episode, and the
                    # terminal observation's ee for the last transition.
                    ee_next = {}
                    for tau in range(start, end):
                        ee_next[tau] = raw_obs[tau + 1, e, self._ee_sl].copy()
                    ee_next[end] = term_raw[(end, e)][self._ee_sl].copy()

                    if self.her_strategy == "final":
                        goals = [(end, ee_next[end])]  # goal = final achieved ee
                    else:  # future_k4: sample 4 future steps as goals
                        rng = np.random.default_rng()
                        goals = []
                        for _ in range(self.her_k):
                            f = int(rng.integers(start, end + 1))
                            goals.append((f, ee_next[f].copy()))

                    for f, g in goals:
                        if built >= cap:
                            break
                        traj = self._relabel_episode(raw_obs[:, e, :], actions_a[:, e, :],
                                                     ee_next, start, f, g)
                        if traj is None:
                            continue
                        L = len(traj["adv"])
                        if built + L > cap:      # respect the ≤1× cap exactly
                            L = cap - built
                            for k in ("obs", "actions", "values", "logp", "adv", "ret"):
                                traj[k] = traj[k][:L]
                        out_obs.append(traj["obs"]); out_act.append(traj["actions"])
                        out_val.append(traj["values"]); out_logp.append(traj["logp"])
                        out_adv.append(traj["adv"]); out_ret.append(traj["ret"])
                        built += L
                start = t + 1
                if built >= cap:
                    break

        empty = (obs_dim,)
        return {
            "obs": np.concatenate(out_obs, 0) if out_obs else np.zeros((0, *empty), np.float32),
            "actions": np.concatenate(out_act, 0) if out_act else np.zeros((0, actions_a.shape[2]), np.float32),
            "values": np.concatenate(out_val, 0) if out_val else np.zeros(0),
            "logp": np.concatenate(out_logp, 0) if out_logp else np.zeros(0),
            "adv": np.concatenate(out_adv, 0) if out_adv else np.zeros(0),
            "ret": np.concatenate(out_ret, 0) if out_ret else np.zeros(0),
        }

    def _relabel_episode(self, raw_e, act_e, ee_next, start, goal_step, g) -> dict | None:
        """
        Relabel transitions [start .. goal_step] with goal g, ending in success at
        goal_step. Returns a GAE'd trajectory (normalized obs, actions, old V̂/logp,
        advantages, returns) or None if degenerate.
        """
        rel_raw, rel_act, rel_rew, rel_done = [], [], [], []
        for tau in range(start, goal_step + 1):
            s = raw_e[tau].copy()
            ee_t = s[self._ee_sl].copy()          # current-state ee (obs feature)
            s[self._tgt_sl] = g                    # overwrite target with achieved goal
            s[self._dist_i] = np.linalg.norm(ee_t - g)   # recompute distance feature
            achieved = np.linalg.norm(ee_next[tau] - g) < SUCCESS_DISTANCE
            rel_raw.append(s)
            rel_act.append(act_e[tau])
            rel_rew.append(100.0 if achieved else 0.0)  # sparse reward under g
            rel_done.append(achieved)
            if achieved:                            # terminate at first achievement
                break
        L = len(rel_raw)
        if L == 0:
            return None
        rel_raw = np.asarray(rel_raw, dtype=np.float32)
        rel_act = np.asarray(rel_act, dtype=np.float32)
        rel_done = np.asarray(rel_done, dtype=np.float64)
        rel_done[-1] = 1.0                          # trajectory ends here

        # Normalize relabeled obs with the rollout's frozen obs-rms, then recompute
        # V̂(s,g) and π_old(a|s,g) with the PRE-UPDATE network (ratios start at 1).
        vecnorm = self.get_vec_normalize_env()
        norm = vecnorm.normalize_obs(rel_raw) if vecnorm is not None else rel_raw
        with th.no_grad():
            obs_t = obs_as_tensor(norm.astype(np.float32), self.device)
            act_t = obs_as_tensor(rel_act, self.device)
            values = self.policy.predict_values(obs_t).cpu().numpy().flatten()
            _, logp, _ = self.policy.evaluate_actions(obs_t, act_t)
            logp = logp.cpu().numpy().flatten()

        # GAE over the coherent single-goal trajectory (terminal success at end).
        from ppo_shaped import gae_advantages
        next_v = np.zeros(L, dtype=np.float64)
        next_v[:-1] = values[1:]                    # bootstrap from next relabeled state
        # next_v[-1] = 0 (terminal success)
        rewards = np.asarray(rel_rew, dtype=np.float64).reshape(L, 1)
        adv = gae_advantages(rewards, values.reshape(L, 1), next_v.reshape(L, 1),
                             rel_done.reshape(L, 1), self.gamma, self.gae_lambda).reshape(L)
        ret = adv + values
        return {"obs": norm.astype(np.float32), "actions": rel_act,
                "values": values, "logp": logp, "adv": adv, "ret": ret}

    # ══════════════════════════════════════════════════════════════════════════
    #  PPO update over the combined batch (split instrumentation → progress.csv)
    # ══════════════════════════════════════════════════════════════════════════
    def train(self) -> None:
        assert self._combined is not None, "collect_rollouts must run before train"
        c = self._combined
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (self.clip_range_vf(self._current_progress_remaining)
                         if self.clip_range_vf is not None else None)

        device = self.device
        obs = obs_as_tensor(c["obs"], device)
        acts = obs_as_tensor(c["actions"], device)
        old_values = th.as_tensor(c["old_values"], dtype=th.float32, device=device)
        old_logp = th.as_tensor(c["old_logp"], dtype=th.float32, device=device)
        advantages = th.as_tensor(c["advantages"], dtype=th.float32, device=device)
        returns = th.as_tensor(c["returns"], dtype=th.float32, device=device)
        is_rel = th.as_tensor(c["is_relabeled"], dtype=th.bool, device=device)
        N = obs.shape[0]

        ent_all, kl_all = [], []
        clipf_orig, clipf_rel = [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            for idx in th.randperm(N, device=device).split(self.batch_size):
                a = acts[idx]
                if isinstance(self.action_space, spaces.Discrete):
                    a = a.long().flatten()
                values, logp, entropy = self.policy.evaluate_actions(obs[idx], a)
                values = values.flatten()
                adv = advantages[idx]
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = th.exp(logp - old_logp[idx])
                pl1 = adv * ratio
                pl2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(pl1, pl2).mean()

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values[idx] + th.clamp(
                        values - old_values[idx], -clip_range_vf, clip_range_vf)
                value_loss = th.nn.functional.mse_loss(returns[idx], values_pred)
                entropy_loss = -th.mean(-logp) if entropy is None else -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = logp - old_logp[idx]
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    kl_all.append(approx_kl)
                    clipped = (th.abs(ratio - 1) > clip_range).float()
                    rmask = is_rel[idx]
                    if (~rmask).any():
                        clipf_orig.append(clipped[~rmask].mean().item())
                    if rmask.any():
                        clipf_rel.append(clipped[rmask].mean().item())
                    ent_all.append((-entropy_loss).item())

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        # ── Logging (TensorBoard + progress.csv) ───────────────────────────────
        rel_frac = float(c["is_relabeled"].mean())
        cf_orig = float(np.mean(clipf_orig)) if clipf_orig else float("nan")
        cf_rel = float(np.mean(clipf_rel)) if clipf_rel else float("nan")
        entropy_mean = float(np.mean(ent_all)) if ent_all else float("nan")
        approx_kl_mean = float(np.mean(kl_all)) if kl_all else float("nan")
        self.logger.record("her/relabeled_fraction", rel_frac)
        self.logger.record("her/clip_fraction_original", cf_orig)
        self.logger.record("her/clip_fraction_relabeled", cf_rel)
        self.logger.record("train/entropy_loss", -entropy_mean)
        self.logger.record("train/approx_kl", approx_kl_mean)

        self._write_progress({
            "timestep": int(self.num_timesteps),
            "relabeled_fraction": rel_frac,
            "clip_fraction_original": cf_orig,
            "clip_fraction_relabeled": cf_rel,
            "clip_fraction": float(np.mean(clipf_orig + clipf_rel)) if (clipf_orig or clipf_rel) else float("nan"),
            "entropy": entropy_mean,
            "approx_kl": approx_kl_mean,
            "value_target_mean_abs": float(np.mean(np.abs(c["returns"]))),
            "adv_mean_prenorm": float(np.mean(c["advantages"])),
            "adv_std_prenorm": float(np.std(c["advantages"])),
            "nonzero_raw_reward_frac": float(np.mean(c["raw_rewards_orig"] != 0.0)),
        })

    # ── progress.csv writer ───────────────────────────────────────────────────
    _PROGRESS_FIELDS = [
        "timestep", "relabeled_fraction", "clip_fraction_original",
        "clip_fraction_relabeled", "clip_fraction", "entropy", "approx_kl",
        "value_target_mean_abs", "adv_mean_prenorm", "adv_std_prenorm",
        "nonzero_raw_reward_frac",
    ]

    def _write_progress(self, row: dict) -> None:
        if self._progress_csv_path is None:
            return
        if self._progress_writer is None:
            self._progress_file = open(self._progress_csv_path, "w", newline="")
            self._progress_writer = csv.DictWriter(self._progress_file,
                                                   fieldnames=self._PROGRESS_FIELDS)
            self._progress_writer.writeheader()
        self._progress_writer.writerow(row)
        self._progress_file.flush()

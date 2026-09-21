"""
ppo_shaped.py
=============
PPO with belief-state reward shaping (BSRS) for the sparse→dense reward study.

Background
----------
Potential-based reward shaping (PBRS): r'_t = r_t + γΦ(s_{t+1}) − Φ(s_t).
GAE:  δ_t = r_t + γV̂(s_{t+1}) − V̂(s_t),  A_t = Σ_k (γλ)^k δ_{t+k}.

KEY IDENTITY implemented and verified here (tests_shaping.py):
    If Φ = η·V̂ where V̂ is the SAME critic used by GAE (computed at rollout
    time), then the shaped GAE advantages satisfy
        A'_t = (1 + η)·A_t − η·R^{γλ}_t,      R^{γλ}_t = Σ_k (γλ)^k r_{t+k},
    with the (γλ)-tail resetting at episode boundaries. This is what makes
    "coupled BSRS" collapse toward the sparse baseline under advantage
    normalization (prediction P1).

Decoupled BSRS uses Φ = η·V_pot where V_pot is a Polyak-averaged *target* copy
of the value net (τ = 0.01, updated once per rollout), NOT the GAE critic. Then
    A'_t = A_t + η·Σ_k (γλ)^k [γV_pot(s_{t+k+1}) − V_pot(s_{t+k})],
which does not telescope against the critic and so escapes the collapse (P3).

Design (why override collect_rollouts)
--------------------------------------
Stock SB3 folds the truncation bootstrap γ·V̂(s_term) directly into the stored
reward before buffering (on_policy_algorithm.py). That destroys the raw r_t we
need for R^{γλ} and for shaping. So `ShapedPPO.collect_rollouts` reimplements
collection to keep raw rewards and an explicit `next_value` array (0 at true
termination, V̂(terminal_obs) at truncation — never zero Φ at truncation), then
runs `gae_advantages` (a pure function, reused by the tests) instead of
`RolloutBuffer.compute_returns_and_advantage`. In `shaping_mode="none"` this
reproduces stock-SB3 advantages exactly (proven in tests_shaping.py), so the
sparse baseline is unchanged.

All runs use VecNormalize(norm_obs=True, norm_reward=False).
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, polyak_update
from stable_baselines3.common.vec_env import VecEnv

VALID_SHAPING_MODES = ("none", "bsrs_coupled", "bsrs_decoupled")


# ══════════════════════════════════════════════════════════════════════════════
#  Pure math core — shared by ShapedPPO and the unit tests
# ══════════════════════════════════════════════════════════════════════════════
def gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """
    Generalized Advantage Estimation with explicit next-state bootstrapping.

    All arrays are shape (T, n_envs). Boundary convention (differs from stock
    SB3, which folds truncation into the reward):
      * `next_values[t]` is the bootstrap value V̂(s_{t+1}) for transition t —
        0 at a true termination, V̂(terminal_obs) at a truncation, V̂(s_{t+1})
        otherwise. The bootstrap term γ·next_values[t] is ALWAYS added.
      * `dones[t]` ∈ {0,1} marks an episode boundary at step t and only masks
        the (γλ)-tail so advantages do not leak across episodes.

    In `none` shaping this yields advantages identical to
    RolloutBuffer.compute_returns_and_advantage (verified in tests).
    """
    T, n = rewards.shape
    adv = np.zeros((T, n), dtype=np.float64)
    last = np.zeros(n, dtype=np.float64)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * gae_lambda * (1.0 - dones[t]) * last
        adv[t] = last
    return adv


def discounted_reward_tail(rewards: np.ndarray, dones: np.ndarray, coef: float) -> np.ndarray:
    """
    R^{coef}_t = Σ_k coef^k · r_{t+k}, resetting at episode boundaries (dones[t]).
    With coef = γλ this is the R^{γλ}_t of the coupled-BSRS identity.
    """
    T, n = rewards.shape
    R = np.zeros((T, n), dtype=np.float64)
    last = np.zeros(n, dtype=np.float64)
    for t in reversed(range(T)):
        last = rewards[t] + coef * (1.0 - dones[t]) * last
        R[t] = last
    return R


# ══════════════════════════════════════════════════════════════════════════════
#  ShapedPPO
# ══════════════════════════════════════════════════════════════════════════════
class ShapedPPO(PPO):
    """
    PPO subclass supporting potential-based reward shaping at rollout time.

    Parameters
    ----------
    shaping_mode : {"none", "bsrs_coupled", "bsrs_decoupled"}
        none            standard PPO (custom GAE path, identical to stock SB3).
        bsrs_coupled    Φ = η·V̂ using the live critic computed during the rollout.
        bsrs_decoupled  Φ = η·V_pot using a Polyak target copy of the value net.
    eta : float
        Shaping strength η.
    target_tau : float
        Polyak coefficient for the decoupled target value net (updated once per
        rollout, in `train`). τ=0 freezes the target (used by test T4).

    After each rollout, per-transition diagnostics are stored in
    `self._last_rollout_diag` (used by the tests and the progress callback).
    """

    def __init__(self, *args, shaping_mode: str = "none", eta: float = 0.0,
                 target_tau: float = 0.01, **kwargs):
        if shaping_mode not in VALID_SHAPING_MODES:
            raise ValueError(
                f"Unknown shaping_mode={shaping_mode!r}. "
                f"Expected one of {VALID_SHAPING_MODES}."
            )
        # These must be set before super().__init__ triggers _setup_model.
        self.shaping_mode = shaping_mode
        self.eta = float(eta)
        self.target_tau = float(target_tau)
        self._pot_policy = None
        self._last_rollout_diag: dict[str, np.ndarray] = {}
        self._progress_callback = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.shaping_mode == "bsrs_decoupled":
            # Frozen-by-default target copy of the whole policy; only its value
            # head is queried. Polyak-updated once per rollout in `train`.
            self._pot_policy = copy.deepcopy(self.policy)
            self._pot_policy.set_training_mode(False)
            for p in self._pot_policy.parameters():
                p.requires_grad_(False)

    def _excluded_save_params(self):
        # These hold open file handles / large scratch arrays / a live module and
        # must not be pickled by model.save().
        return super()._excluded_save_params() + [
            "_progress_callback", "_last_rollout_diag", "_pot_policy",
        ]

    # ── Potential value V_pot(s) (η applied by caller) ────────────────────────
    def _potential_values(self, obs_tensor) -> th.Tensor:
        """V_pot(s): live critic for coupled, target net for decoupled."""
        if self.shaping_mode == "bsrs_decoupled":
            return self._pot_policy.predict_values(obs_tensor).flatten()
        return self.policy.predict_values(obs_tensor).flatten()

    # ── Rollout collection (raw rewards kept; custom GAE) ─────────────────────
    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        n_envs = env.num_envs
        shaping_on = self.shaping_mode != "none"
        # Per-transition scratch arrays (T, n_envs).
        raw_rewards = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)
        pot_values = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)   # η·V_pot(s_t)
        dones_arr = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)    # boundary flag
        next_values = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)  # V̂(s_{t+1}) bootstrap
        next_pot = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)     # η·V_pot(s_{t+1})
        bdry_next_val = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)
        bdry_next_pot = np.zeros((n_rollout_steps, n_envs), dtype=np.float64)
        # (t, env) -> normalized terminal_observation, for test verification.
        terminal_obs: dict[tuple[int, int], np.ndarray] = {}

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                if shaping_on:
                    pot_values[n_steps] = (
                        self.eta * self._potential_values(obs_tensor).cpu().numpy()
                    )
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            raw_rewards[n_steps - 1] = rewards
            dones_arr[n_steps - 1] = dones.astype(np.float64)

            # Boundary bootstrap: V̂(terminal_obs)/Φ at truncation; 0 at true
            # termination. terminal_observation is already VecNormalize-normalized.
            for idx, done in enumerate(dones):
                if not done:
                    continue
                if infos[idx].get("terminal_observation") is not None:
                    terminal_obs[(n_steps - 1, idx)] = np.asarray(
                        infos[idx]["terminal_observation"], dtype=np.float32
                    )
                truncated = (
                    infos[idx].get("TimeLimit.truncated", False)
                    and infos[idx].get("terminal_observation") is not None
                )
                if truncated:
                    term_t = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        v_term = self.policy.predict_values(term_t).flatten()[0].item()
                        p_term = (
                            self.eta * self._potential_values(term_t)[0].item()
                            if shaping_on else 0.0
                        )
                    bdry_next_val[n_steps - 1, idx] = v_term
                    bdry_next_pot[n_steps - 1, idx] = p_term
                # else true termination → bootstrap 0, Φ(s') = 0 (arrays already 0)

            rollout_buffer.add(
                self._last_obs, actions, rewards,
                self._last_episode_starts, values, log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Final bootstrap for the last (possibly non-terminal) step.
        with th.no_grad():
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            last_values = self.policy.predict_values(last_obs_tensor).cpu().numpy().flatten()
            last_pot = (
                self.eta * self._potential_values(last_obs_tensor).cpu().numpy()
                if shaping_on else np.zeros(n_envs)
            )

        # Reconstruct next_values / next_pot with boundary awareness.
        values_buf = rollout_buffer.values  # (T, n_envs) = V̂(s_t)
        for t in range(n_rollout_steps):
            for e in range(n_envs):
                if dones_arr[t, e]:
                    next_values[t, e] = bdry_next_val[t, e]
                    next_pot[t, e] = bdry_next_pot[t, e]
                elif t < n_rollout_steps - 1:
                    next_values[t, e] = values_buf[t + 1, e]
                    next_pot[t, e] = pot_values[t + 1, e]
                else:
                    next_values[t, e] = last_values[e]
                    next_pot[t, e] = last_pot[e]

        # Shaped rewards r'_t = r_t + γΦ(s_{t+1}) − Φ(s_t)   (Φ arrays include η).
        if shaping_on:
            shaped_rewards = raw_rewards + self.gamma * next_pot - pot_values
        else:
            shaped_rewards = raw_rewards.copy()

        # Custom GAE on shaped rewards; write advantages/returns into the buffer.
        advantages = gae_advantages(
            shaped_rewards, values_buf, next_values, dones_arr,
            self.gamma, self.gae_lambda,
        )
        rollout_buffer.advantages = advantages.astype(np.float32)
        rollout_buffer.returns = (advantages + values_buf).astype(np.float32)

        self._last_rollout_diag = {
            "raw_rewards": raw_rewards,
            "shaped_rewards": shaped_rewards,
            "values": values_buf.copy(),
            "next_values": next_values,
            "pot_values": pot_values,
            "next_pot": next_pot,
            "dones": dones_arr,
            "advantages": advantages,
            "terminal_obs": terminal_obs,
        }

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ── Training step: standard PPO + Polyak target + progress logging ────────
    def train(self) -> None:
        super().train()
        if self.shaping_mode == "bsrs_decoupled" and self.target_tau > 0.0:
            polyak_update(self.policy.parameters(),
                          self._pot_policy.parameters(), self.target_tau)
        if self._progress_callback is not None:
            self._progress_callback.write_row()


# ══════════════════════════════════════════════════════════════════════════════
#  Per-update diagnostics → progress.csv
# ══════════════════════════════════════════════════════════════════════════════
class ProgressCSVCallback(BaseCallback):
    """
    Logs one row per PPO update to `progress.csv` (alongside the TensorBoard log):
      timestep, value_target_mean_abs, value_target_max_abs,
      adv_mean_prenorm, adv_std_prenorm, nonzero_raw_reward_frac,
      clip_fraction, entropy, approx_kl

    The row is written from ShapedPPO.train() (via `write_row`), not on_step,
    because on_rollout_end fires before train() in SB3 — so buffer statistics
    (pre-normalization advantages, value targets) and the just-computed train
    metrics can only be paired for the same update after super().train() runs.
    """

    FIELDS = [
        "timestep", "value_target_mean_abs", "value_target_max_abs",
        "adv_mean_prenorm", "adv_std_prenorm", "nonzero_raw_reward_frac",
        "clip_fraction", "entropy", "approx_kl",
    ]

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self._csv_path = csv_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        import csv
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        # Register so ShapedPPO.train() can call write_row after each update.
        self.model._progress_callback = self

    def _on_step(self) -> bool:
        return True

    def write_row(self) -> None:
        if self._writer is None:
            return
        buf = self.model.rollout_buffer
        adv = np.asarray(buf.advantages, dtype=np.float64).ravel()
        returns = np.asarray(buf.returns, dtype=np.float64).ravel()
        diag = getattr(self.model, "_last_rollout_diag", {})
        raw = np.asarray(diag.get("raw_rewards", np.zeros(1)), dtype=np.float64).ravel()
        logs = self.model.logger.name_to_value
        row = {
            "timestep": int(self.model.num_timesteps),
            "value_target_mean_abs": float(np.mean(np.abs(returns))),
            "value_target_max_abs": float(np.max(np.abs(returns))),
            "adv_mean_prenorm": float(np.mean(adv)),
            "adv_std_prenorm": float(np.std(adv)),
            "nonzero_raw_reward_frac": float(np.mean(raw != 0.0)),
            "clip_fraction": float(logs.get("train/clip_fraction", np.nan)),
            "entropy": float(-logs.get("train/entropy_loss", np.nan)),
            "approx_kl": float(logs.get("train/approx_kl", np.nan)),
        }
        self._writer.writerow(row)
        self._file.flush()

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

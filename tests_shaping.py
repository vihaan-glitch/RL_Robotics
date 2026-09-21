"""
tests_shaping.py
================
Standalone numerical verification of the reward-shaping math in ppo_shaped.py.
Run directly:

    python tests_shaping.py

All four tests operate on a FIXED-SEED rollout of ~2048 steps on the easy sparse
env collected with an UNTRAINED policy (random init → nontrivial V̂). The tests
encode mathematical identities: a FAILURE means a real bug (or a real math
error), not something to "fix" by loosening the assertion.

T1 Telescoping   : static potential Φ ⇒ discounted shaped return
                   = raw return + γ^T Φ(s_T) − Φ(s_0), per episode segment.
T2 Coupled ident.: A'_t = (1+η)A_t − η R^{γλ}_t for η ∈ {1, 2}, across done and
                   truncation boundaries (≥1 truncation forced by 2048 > 500).
T3 Truncation    : the γΦ(s_{t+1}) term at a truncation boundary uses
                   V̂(terminal_observation), not 0 — checked against a hand
                   recomputation.
T4 Decoupled     : with a frozen (τ=0) target potential V_pot ≠ V̂, shaped
                   advantages equal A_t + η·Σ(γλ)^k[γV_pot(s') − V_pot(s)].
"""

import hashlib
import sys

import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv
from ppo_shaped import ShapedPPO, gae_advantages, discounted_reward_tail

GAMMA = 0.99
GAE_LAMBDA = 0.95
N_STEPS = 2048
TOL = 1e-5


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_env(seed: int):
    env = DummyVecEnv([lambda: ArmReachEnv(reward_mode="sparse", obstacle_mode=None)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.seed(seed)
    return env


def build_and_collect(shaping_mode: str, eta: float, seed: int = 0,
                      target_tau: float = 0.01, perturb_target: bool = False):
    """Fixed-seed rollout with an untrained policy; returns (model, diag, buffer)."""
    th.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(seed)
    model = ShapedPPO(
        "MlpPolicy", env, shaping_mode=shaping_mode, eta=eta, target_tau=target_tau,
        n_steps=N_STEPS, gamma=GAMMA, gae_lambda=GAE_LAMBDA, seed=seed, verbose=0,
    )
    if perturb_target and model._pot_policy is not None:
        # Make V_pot ≠ V̂ so the decoupled test genuinely exercises the target net.
        with th.no_grad():
            for p in model._pot_policy.parameters():
                p.add_(0.5 * th.randn_like(p))
    total_timesteps, callback = model._setup_learn(N_STEPS)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)
    return model, model._last_rollout_diag, model.rollout_buffer


def phi_static(obs: np.ndarray) -> float:
    """Arbitrary deterministic hash-based potential (used only by T1)."""
    b = np.asarray(obs, dtype=np.float32).tobytes()
    h = int.from_bytes(hashlib.sha256(b).digest()[:8], "little") / 2**64
    return (h - 0.5) * 10.0


def episode_segments(dones: np.ndarray):
    """Yield (start, end) index pairs for episodes ending at a boundary (env 0)."""
    d = dones[:, 0]
    start = 0
    for t in range(len(d)):
        if d[t]:
            yield start, t
            start = t + 1


# ── T1 — Telescoping of a static potential ────────────────────────────────────
def test_T1(diag, buffer):
    obs = np.asarray(buffer.observations)[:, 0, :]  # (T, obs_dim), normalized s_t
    raw = diag["raw_rewards"][:, 0]
    term_obs = diag["terminal_obs"]
    max_err, n_checked = 0.0, 0
    for start, end in episode_segments(diag["dones"]):
        if (end, 0) not in term_obs:
            continue  # need the terminal state to close the telescope
        states = [obs[t] for t in range(start, end + 1)] + [term_obs[(end, 0)]]
        rew = raw[start:end + 1]
        L = len(rew)
        phis = [phi_static(s) for s in states]
        shaped = [rew[k] + GAMMA * phis[k + 1] - phis[k] for k in range(L)]
        disc = np.array([GAMMA ** k for k in range(L)])
        disc_shaped = float(np.sum(disc * shaped))
        disc_raw = float(np.sum(disc * rew))
        telescoped = disc_raw + GAMMA ** L * phis[L] - phis[0]
        max_err = max(max_err, abs(disc_shaped - telescoped))
        n_checked += 1
    assert n_checked >= 1, "no closable episode segments found"
    ok = max_err < TOL
    return ok, max_err, f"{n_checked} episode segments"


# ── T2 — Coupled identity A'_t = (1+η)A_t − η R^{γλ}_t ────────────────────────
def test_T2(diag):
    raw = diag["raw_rewards"]
    values = diag["values"].astype(np.float64)
    next_values = diag["next_values"].astype(np.float64)
    dones = diag["dones"]
    n_trunc = sum(1 for k in diag["terminal_obs"])
    assert n_trunc >= 1, "no truncation boundary in rollout (need ≥1)"

    A_unshaped = gae_advantages(raw, values, next_values, dones, GAMMA, GAE_LAMBDA)
    R_tail = discounted_reward_tail(raw, dones, GAMMA * GAE_LAMBDA)

    max_err = 0.0
    detail = []
    # η=1: the model's OWN coupled advantages (real code path).
    lhs_model = diag["advantages"]
    rhs_1 = 2.0 * A_unshaped - 1.0 * R_tail
    e1 = float(np.max(np.abs(lhs_model - rhs_1)))
    max_err = max(max_err, e1)
    detail.append(f"η=1(model) {e1:.2e}")
    # η ∈ {1,2}: shaped rewards rebuilt from the SAME buffer V̂, re-run through GAE.
    for eta in (1.0, 2.0):
        shaped = raw + eta * (GAMMA * next_values - values)
        A_shaped = gae_advantages(shaped, values, next_values, dones, GAMMA, GAE_LAMBDA)
        rhs = (1.0 + eta) * A_unshaped - eta * R_tail
        e = float(np.max(np.abs(A_shaped - rhs)))
        max_err = max(max_err, e)
        detail.append(f"η={eta:g} {e:.2e}")
    return max_err < TOL, max_err, ", ".join(detail) + f" ({n_trunc} truncations)"


# ── T3 — Truncation uses V̂(terminal_obs), not 0 ──────────────────────────────
def test_T3(model, diag):
    dones = diag["dones"]
    next_values = diag["next_values"]
    next_pot = diag["next_pot"]
    term_obs = diag["terminal_obs"]
    # Pick a truncation boundary (env 0).
    t_star = None
    for (t, e) in term_obs:
        if e == 0 and dones[t, 0]:
            t_star = t
            break
    assert t_star is not None, "no truncation boundary found"

    # Hand recomputation of V̂(terminal_obs) with the same (untrained) critic.
    term = model.policy.obs_to_tensor(term_obs[(t_star, 0)])[0]
    with th.no_grad():
        v_hand = float(model.policy.predict_values(term).flatten()[0].item())

    err_val = abs(next_values[t_star, 0] - v_hand)
    err_pot = abs(next_pot[t_star, 0] - model.eta * v_hand)
    nonzero = abs(v_hand) > 1e-8 and abs(next_pot[t_star, 0]) > 1e-8
    ok = err_val < TOL and err_pot < TOL and nonzero
    detail = (f"t*={t_star} V̂(term)={v_hand:.4f} next_pot={next_pot[t_star,0]:.4f} "
              f"(η={model.eta:g}); nonzero={nonzero}")
    return ok, max(err_val, err_pot), detail


# ── T4 — Decoupled: A'_t = A_t + η·Σ(γλ)^k[γV_pot(s') − V_pot(s)] ─────────────
def test_T4(diag, eta):
    raw = diag["raw_rewards"]
    values = diag["values"].astype(np.float64)
    next_values = diag["next_values"].astype(np.float64)
    dones = diag["dones"]
    # pot arrays carry η already; divide it out to get the raw V_pot residuals.
    v_pot = diag["pot_values"] / eta
    v_pot_next = diag["next_pot"] / eta
    residual = GAMMA * v_pot_next - v_pot           # γV_pot(s') − V_pot(s)

    A_unshaped = gae_advantages(raw, values, next_values, dones, GAMMA, GAE_LAMBDA)
    tail = discounted_reward_tail(residual, dones, GAMMA * GAE_LAMBDA)
    rhs = A_unshaped + eta * tail

    lhs = diag["advantages"]                        # model's decoupled advantages
    # Confirm the target really differs from the critic (else test is vacuous).
    pot_differs = float(np.max(np.abs(v_pot - values))) > 1e-4
    err = float(np.max(np.abs(lhs - rhs)))
    ok = err < TOL and pot_differs
    return ok, err, f"η={eta:g}, max|V_pot−V̂|={np.max(np.abs(v_pot-values)):.3f}"


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    import warnings
    warnings.filterwarnings("ignore")

    print("Collecting fixed-seed rollouts (untrained policy)...")
    m_c, diag_c, buf_c = build_and_collect("bsrs_coupled", eta=1.0, seed=0)
    m_d, diag_d, buf_d = build_and_collect("bsrs_decoupled", eta=1.0, seed=0,
                                           target_tau=0.0, perturb_target=True)

    results = []
    results.append(("T1 telescoping", *test_T1(diag_c, buf_c)))
    results.append(("T2 coupled identity", *test_T2(diag_c)))
    results.append(("T3 truncation bootstrap", *test_T3(m_c, diag_c)))
    results.append(("T4 decoupled residuals", *test_T4(diag_d, eta=1.0)))

    print(f"\n{'='*74}\n  SHAPING UNIT TESTS\n{'='*74}")
    all_ok = True
    for name, ok, err, detail in results:
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"  [{status}] {name:26} max_err={err:.2e}   {detail}")
    print("=" * 74)
    if all_ok:
        print("  ALL TESTS PASSED\n")
        return 0
    print("  SOME TESTS FAILED — this indicates a real bug or math error.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

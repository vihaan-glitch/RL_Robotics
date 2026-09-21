"""
tests_her.py — checks for the on-policy HER relabeling reward (ppo_her.py).

H1  Relabeled reward = original (goal-independent) reward + 100 at the relabeled
    goal: an injected −10 collision penalty survives relabeling at exactly the
    steps where it occurred, and the final relabeled step carries +100 on top.
H2  relabel_goal_independent_reward=False reproduces the pre-fix behaviour
    (success term only), used for results_s2d/hard/her_final_* at commit 962dffb.
H3  With no collisions (easy task), both settings give identical relabeled
    batches, so the published easy-task HER results are unaffected by the fix.

Run:  python tests_her.py        (~10 s)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arm_reach_env import ArmReachEnv
from ppo_her import HERPPO

T = 1024  # two full 500-step failed episodes + a partial one


def make_env(obstacle=None):
    return VecNormalize(DummyVecEnv([lambda: ArmReachEnv(
        render_mode=None, reward_mode="sparse", obstacle_mode=obstacle)]),
        norm_obs=True, norm_reward=False)


def make(flag, env):
    return HERPPO("MlpPolicy", env, her_strategy="final", n_steps=T, batch_size=64,
                  seed=0, verbose=0, relabel_goal_independent_reward=flag)


def collect(m):
    """Run one rollout and capture the arrays _build_relabeled_batch receives."""
    captured = {}
    orig = m._build_relabeled_batch

    def spy(raw_obs, actions_a, rewards_a, dones_a, trunc_a, term_raw, cap):
        captured.update(raw_obs=raw_obs.copy(), actions_a=actions_a.copy(),
                        rewards_a=rewards_a.copy(), dones_a=dones_a.copy(),
                        trunc_a=trunc_a.copy(), term_raw=dict(term_raw), cap=cap)
        return orig(raw_obs, actions_a, rewards_a, dones_a, trunc_a, term_raw, cap)
    m._build_relabeled_batch = spy
    _, cb = m._setup_learn(T)
    m.collect_rollouts(m.env, cb, m.rollout_buffer, T)
    m._build_relabeled_batch = orig
    return captured


def build(m, c, rewards):
    return m._build_relabeled_batch(c["raw_obs"], c["actions_a"], rewards, c["dones_a"],
                                    c["trunc_a"], c["term_raw"], c["cap"])


def main():
    results = []

    # ── H1 / H2: inject collisions into the first failed episode ─────────────
    env = make_env()
    m_fix = make(True, env)
    c = collect(m_fix)
    # Legacy-behaviour model: same env (hence the same obs normaliser) and weights.
    m_old = make(False, env)
    m_old.policy.load_state_dict(m_fix.policy.state_dict())
    done_idx = np.flatnonzero(c["dones_a"][:, 0])
    assert len(done_idx) >= 1 and c["trunc_a"][done_idx[0], 0], "need a truncated first episode"
    end = done_idx[0]                                   # first episode = [0 .. end]
    assert np.all(c["rewards_a"][: end + 1, 0] == 0.0), "untrained policy should not succeed"
    # The relabeled trajectory stops at the FIRST step that reaches the final
    # achieved position (often before `end`, since the arm settles), so measure
    # its length L from the clean batch: the first +100 marks its last step.
    clean = build(m_old, c, c["rewards_a"])["rewards"]
    L = int(np.flatnonzero(clean == 100.0)[0]) + 1
    assert 60 < L <= end + 1, L
    inject = np.array([3, 57, 58, L - 1])               # includes the goal step itself
    rew = c["rewards_a"].copy()
    rew[inject, 0] = -10.0

    out = build(m_fix, c, rew)
    r = out["rewards"][:L]
    expect = np.zeros(L); expect[inject] = -10.0; expect[L - 1] += 100.0
    ok1 = r.shape == (L,) and np.array_equal(r, expect)
    results.append(("H1 collision penalty kept on relabeled steps (+100 at goal)", ok1,
                    f"L={L}; steps with −10: {np.flatnonzero(r == -10).tolist()}, "
                    f"goal step reward={r[-1]:.0f} (= −10 + 100)"))

    out_old = build(m_old, c, rew)
    r_old = out_old["rewards"][:L]
    expect_old = np.zeros(L); expect_old[L - 1] = 100.0
    ok2 = np.array_equal(r_old, expect_old)
    results.append(("H2 legacy flag drops the penalty (pre-fix behaviour)", ok2,
                    f"nonzero steps: {np.flatnonzero(r_old).tolist()}"))

    # ── H3: no collisions ⇒ both settings produce identical batches ───────────
    a = build(m_fix, c, c["rewards_a"])
    b = build(m_old, c, c["rewards_a"])
    ok3 = all(np.array_equal(a[k], b[k]) for k in ("obs", "actions", "values", "logp", "adv", "ret", "rewards"))
    results.append(("H3 identical batches without collisions (easy task unaffected)", ok3,
                    f"{len(a['adv'])} relabeled transitions compared"))
    env.close()

    print("=" * 74 + "\n  HER RELABELING TESTS\n" + "=" * 74)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    print("=" * 74)
    assert all(ok for _, ok, _ in results), "HER relabeling tests FAILED"
    print("  ALL TESTS PASSED")


if __name__ == "__main__":
    main()

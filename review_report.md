# Review Report — Ghosh Issues 1 & 2

## Errata (added 2026-09-21)

The report below is kept as originally sent. Later analysis found five errors.
Each is marked inline with **[Erratum En]**, and the scripts that establish the
corrections are in `diagnostics/`.

| # | Original statement | Correction | Evidence |
|---|---|---|---|
| E1 | "`model.save()` pickles only `policy.state_dict()` — optimizer moments are never persisted." | False. `OnPolicyAlgorithm._get_torch_save_params` returns `["policy", "policy.optimizer"]`; the checkpoint restores Adam's full state (78,400 steps). The original probe *chose* fresh Adam; it was not forced to. | `final_model.zip` contains `policy.optimizer.pth`; `p2_update_probe.py` runs both regimes |
| E2 | Gradient clipping and Adam are "two independent mechanisms" that each destroy the (1+η) scale. | Clipping multiplies the whole gradient by one scalar. The value-loss gradient (norm 52.4) is ~8× the policy-loss gradient (6.4) and dominates the total (52.8), so after clipping the policy-side gradient is still ≈ 2× larger. Only Adam's per-parameter √v̂ cancels it, and only once v̂ has adapted: β₂ = 0.999 means ~1,000 minibatch steps ≈ 3 PPO updates. With warm Adam and the scale switched on suddenly, the policy-side update differs by **73%** after 32 steps. In the actual runs the scale is applied from step 0, so it is cancelled. The residual channel in steady state is an effective `ent_coef/(1+η)`, because the entropy gradient is not scaled. | `diagnostics/p2_update_probe.py` |
| E3 | Scaling advantages changed the update by 0.15% (1 step) and 2.8% (32 steps). | The probe used PyTorch's default Adam `eps=1e-8`. With SB3's `eps=1e-5`: 3.7% and 3.5% overall, 8.1% on policy parameters. The conclusion (close to a no-op with fresh or consistently scaled Adam) stands. | `p2_update_probe.py`, fresh regime |
| E4 | `bsrs_c_e1_nonorm` and `sparse_nonorm` "should be near-identical optimization processes" because >99.7% of transitions have zero reward. | The identity is exact only for batches with **no** nonzero reward. 42% of `bsrs_c_e1_nonorm` updates (39% of `bsrs_c_e1`, 72% of its last 50) contain a success. In those batches −ηR^{γλ} rotates the normalized advantage vector: corr(A′, A) = 0.81–0.97 at η = 1 with the runs' own critics. Adam does not cancel direction. The P2 *statistical* retraction is unaffected. | `progress.csv` (fraction of updates with nonzero reward > 0); `advantage_direction_probe.py` |
| E5 | The algebra "would predict *growth* with η", so value-target suppression is "not explained by the identity alone". | Wrong. The critic regresses onto the shaped return, whose value is V − ηV̂; self-consistency gives **V̂ = V/(1+η)**. A controlled frozen-policy test confirms it: slopes 0.667/0.492/0.325 (policy A) and 0.661/0.491/0.314 (policy B) against predicted 0.667/0.500/0.333. The confounded 110-run ratios came from policy differences across η. | `p4_fixed_point.py`, `results_p4_fixedpoint/` |

The Issue 1 recommendation (a 4-condition × 15-seed run) is unchanged. It
needs 50 new runs, ~3.5 CPU-hours at the measured 4.1 min per easy run.

**Scope.** Report-only investigation. No changes were made to `ppo_shaped.py`,
`ppo_her.py`, `run_s2d_study.py`, or any data under `results_s2d/`. No training
runs were launched. Two throwaway diagnostic scripts were written to a scratch
directory (reproduced verbatim in the Appendix) and run against the existing
`results_s2d/easy/sparse_seed0/` checkpoint. Findings and open questions only —
no recommendations about paper framing.

---

# ISSUE 1 — Possible learning-rate confound in the P2 result

## 1a. How advantages enter the loss, and Adam's scale invariance

**Path of the advantage tensor** (`stable_baselines3/ppo/ppo.py`, SB3 2.8.0):

```python
216                advantages = rollout_data.advantages
218                if self.normalize_advantage and len(advantages) > 1:
219                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
225                policy_loss_1 = advantages * ratio
226                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
227                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
...
244                value_loss = F.mse_loss(rollout_data.returns, values_pred)
250                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
```

`policy_loss` is **exactly linear in the advantage tensor**. So multiplying every
advantage by a constant `c` multiplies `policy_loss` — and its gradient
contribution — by `c`. It does **not** touch `value_loss` or `entropy_loss`.

**Architecture note (measured, not assumed).** With `MlpPolicy` on a Box
observation space, `share_features_extractor=True` but the shared extractor is a
parameterless `FlattenExtractor`. The diagnostic enumerated all 11,663 trainable
parameters: **6,030 policy-side** (`log_std`, `mlp_extractor.policy_net.*`,
`action_net.*`) and **5,633 value-side** (`mlp_extractor.value_net.*`,
`value_net.*`), with **zero shared trainable parameters**. Therefore scaling the
advantages perturbs only the policy-side gradient; scaling the LR scales updates
for policy-side *and* value-side alike. The two interventions are not the same
operation even before any optimizer effects.

**Adam's normalization denominator.** Adam computes
`m_t = β₁m_{t-1} + (1-β₁)g_t`, `v_t = β₂v_{t-1} + (1-β₂)g_t²`, and steps by
`-lr · m̂_t / (√v̂_t + ε)`. Under a constant rescaling `g → c·g` applied
consistently from `t=0`: `m → c·m` and `v → c²·v`, so
`m̂/√v̂ → c·m̂/(c·√v̂) = m̂/√v̂` — **invariant**, exactly, up to the `ε` term
(which only matters when `√v̂ ≲ ε = 1e-8`). At the very first step from zeroed
moments the invariance is even stronger and holds for *any* positive scaling:
bias correction gives `m̂₁ = g₁` and `v̂₁ = g₁²`, so the update is
`-lr·g₁/(|g₁|+ε) ≈ -lr·sign(g₁)` — magnitude-independent, with
`‖Δθ‖ ≈ lr·√n_params` regardless of gradient scale.

> **[Erratum E2]** Clipping does *not* destroy the policy-side scale: it is one
> global scalar, and the value loss dominates the norm, so the policy share of
> the clipped gradient still roughly doubles. Only Adam cancels it. See the
> Errata section at the top.

**A second, independent scale-destroying mechanism: gradient clipping.**
`max_grad_norm=0.5` renormalizes the whole gradient vector to a fixed norm
before Adam sees it, discarding global scale outright. Measured on a real
minibatch from the checkpoint:

| gradient component | norm |
|---|---|
| `∇ policy_loss` | 6.4446 |
| `∇ vf_coef·value_loss` | **52.3671** |
| `∇ ent_coef·entropy_loss` | 0.0265 |
| `∇ TOTAL` | 52.7620 → clipped by **105.5×** to reach 0.5 |

The total gradient is **value-loss dominated (≈8× the policy-loss gradient)**, so
even doubling the policy-loss contribution moves the total norm only from 52.76
to 53.93 (+2.2%) — and clipping then renormalizes both to exactly 0.5 anyway.

## 1b. Diagnostic: three single PPO steps from identical starting parameters

Loaded `results_s2d/easy/sparse_seed0/final_model.zip` + its `vec_normalize.pkl`,
collected one real 2048-step rollout with those exact weights, took the first
64-transition minibatch, and ran one PPO update from three identical copies
(identical weights, fresh Adam, same minibatch; only the advantage tensor or the
LR differs). η=1, so (1+η)=2.

**Important precondition, measured:** this rollout had
`nonzero_raw_reward_frac = 0.000000` — i.e. `R^{γλ} ≡ 0` across all 2048
transitions, so on this data the coupled-BSRS identity gives
`A′ = (1+η)A` **exactly**, not approximately. Scenario (b) is therefore a
faithful proxy for coupled BSRS with `normalize_advantage=False`, not a loose one.

### Single step (fresh Adam)

| scenario | lr | ‖Δθ‖ total | value-side | policy-side | ratio vs (a) |
|---|---|---|---|---|---|
| (a) baseline, `A` | 3e-4 | 3.236300e-02 | 2.247076e-02 | 2.329030e-02 | 1.0000 |
| (b) `(1+η)·A` | 3e-4 | 3.236391e-02 | 2.246997e-02 | 2.329233e-02 | **1.0000** |
| (c) `A`, lr×(1+η) | 6e-4 | 6.472631e-02 | 4.494151e-02 | 4.658064e-02 | **2.0000** |

`cos(b,a)=1.000006`, `cos(c,a)=1.000004`, `‖b−a‖/‖a‖ = 0.0015`,
`‖b−c‖/‖a‖ = 1.0000`.

Note `‖Δθ‖_a = 3.2363e-02` matches the predicted `lr·√n = 3e-4·√11663 = 3.236e-02`
to four significant figures — confirming the first Adam step is `≈ lr·sign(g)`.

### 32 consecutive updates (Adam moments warmed up)

The single-step test necessarily starts from zeroed Adam state (SB3's
`_get_torch_save_params()` returns `["policy"]`, so `model.save()` pickles only
`policy.state_dict()` — **optimizer moments are never persisted** *[Erratum E1:
false — the on-policy override saves `policy.optimizer`; see Errata]*). To check that
the result is not an artifact of that regime, the same comparison was run for a
full epoch of 32 real minibatches:

| step | (a) | (b) | (c) | b/a | c/a |
|---|---|---|---|---|---|
| 1 | 3.2363e-02 | 3.2364e-02 | 6.4726e-02 | 1.0000 | 2.0000 |
| 4 | 8.3128e-02 | 8.3119e-02 | 1.6419e-01 | 0.9999 | 1.9751 |
| 16 | 2.0773e-01 | 2.0775e-01 | 3.9717e-01 | 1.0001 | 1.9120 |
| 32 | 3.1831e-01 | 3.1729e-01 | 5.3895e-01 | **0.9968** | **1.6932** |

After 32 steps: `cos(b,a)=0.999615`, `cos(c,a)=0.878107`,
`‖b−a‖/‖a‖ = 0.0279`, `‖b−c‖/‖a‖ = 0.9426`.

### Plain statement of the result

- **(b) and (c) are NOT close.** They differ by roughly a full baseline-update
  magnitude (`‖b−c‖/‖a‖ ≈ 0.94–1.00`) and by ~12° in direction after warm-up.
- **(b) is essentially identical to (a)** — the unscaled baseline — in both
  magnitude (ratio 1.0000 → 0.9968) and direction (cos 0.9996–1.0000). Scaling
  every advantage by (1+η) changed the parameter update by **0.15% at step 1 and
  2.8% after 32 steps**. *[Erratum E3: these used PyTorch's default Adam
  eps=1e-8; with SB3's eps=1e-5 they are 3.7% and 3.5% (policy-side 8.1%).]*
- **(c) is far from (a)**, as expected: exactly 2× at step 1, decaying to 1.69×
  as Adam's second moments equilibrate.

## 1c. Assessment

**Is P2 a pure step-size artifact? No — but the actual finding is more damaging
to P2 than the confound Ghosh proposed.**

Ghosh's hypothesis was that "coupled BSRS, norm-off" ≈ "unshaped PPO with a
larger step size." The diagnostic refutes the *equivalence*: scaling advantages
and scaling the LR are demonstrably different interventions (b ≢ c). But it does
so by showing that scaling the advantages is **very nearly a no-op**, not by
showing that it does something distinctive. Two independent mechanisms —
grad-norm clipping (105× here) and Adam's `√v̂` denominator — each destroy the
global scale of the advantage tensor before it can influence the update.

Chaining this with the measured data:

1. Over the actual `bsrs_c_e1_nonorm` training runs,
   `nonzero_raw_reward_frac` peaked at **0.00313** (mean across seeds), i.e.
   **>99.7% of transitions carry zero raw reward**, so `R^{γλ}=0` and
   `A′ = (1+η)A` exactly for essentially the entire run.
2. A pure `(1+η)` rescaling perturbs the update by ~0.15–2.8%.
3. *[Erratum E4: overstated — 42% of `bsrs_c_e1_nonorm` updates contain a
   success, where −ηR^{γλ} rotates the advantage direction; see Errata.]*
   Therefore `bsrs_c_e1_nonorm` and `sparse_nonorm` should be **near-identical
   optimization processes**, differing by a perturbation comparable in size to
   numerical noise — and certainly far smaller than seed-to-seed variation.

And indeed the observed "separation" does not survive inspection:

| condition (easy) | per-seed success | IQM | mean ± std |
|---|---|---|---|
| `sparse_nonorm` | 0.02, 0.00, **0.44**, 0.02, 0.04 | 0.027 | 0.104 ± 0.168 |
| `bsrs_c_e1_nonorm` | **0.34**, **0.38**, 0.00, 0.02, 0.10 | 0.153 | 0.168 ± 0.161 |

The IQM gap (2.7% vs 15.3%) is an artifact of 25%-trimming on n=5:
`sparse_nonorm`'s single strong seed (0.44) is trimmed away entirely, while
`bsrs_c_e1_nonorm`'s two strong seeds (0.34, 0.38) both survive trimming. The
means are 0.104 vs 0.168.

Bootstrap CI on the **difference** (10k resamples, same methodology as the paper):

| comparison | metric | difference | 95% CI | P(diff ≤ 0) |
|---|---|---|---|---|
| P2 norm-OFF | IQM | +0.127 | **[−0.180, +0.340]** | 0.238 |
| P2 norm-OFF | mean | +0.064 | **[−0.156, +0.264]** | 0.286 |

Supplementary: Welch t = +0.550 (p=0.598); Mann–Whitney p=0.750.

The CI straddles zero comfortably, and ~24–29% of resamples put the shaped
condition *behind* the control. **The P2 separation is not statistically
supported by these 5 seeds**, and the mechanism P2 invokes to explain it is
shown above to be inert.

One nuance worth preserving: the ~3% residual direction change after 32 steps is
real (it arises because scaling the policy-loss gradient slightly rotates the
total gradient before clipping). But a 3% perturbation in a chaotic optimization
landscape is functionally **equivalent to changing the seed**, not a directed
improvement in learning signal — and within-condition seed spread here is
enormous (0.00 → 0.80 for `bsrs_c_e1`).

### Exactly what would settle it definitively

A controlled 3-condition comparison at higher seed count. All conditions:
easy task (`obstacle_mode=None`), `reward_mode="sparse"`, 500k steps,
`VecNormalize(norm_obs=True, norm_reward=False)`, `normalize_advantage=False`,
and otherwise the study's existing hyperparameters
(`lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, γ=0.99, λ=0.95,
clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5`).

| condition | shaping | learning rate | purpose |
|---|---|---|---|
| `sparse_nonorm` | none | 3e-4 | control (already exists; re-run to the new seed count) |
| `bsrs_c_e1_nonorm` | coupled, η=1 | 3e-4 | the P2 claim (already exists; extend seeds) |
| **`sparse_nonorm_lr2x`** | none | **6e-4** | **NEW** — isolates the step-size hypothesis directly |
| **`sparse_nonorm_advx2`** | none, advantages ×2 | 3e-4 | **NEW** — isolates the pure-rescaling hypothesis |

**Seeds: 0–14 (15 seeds per condition), not 5.** Justification: the observed
between-condition difference in means is 0.064 with pooled per-seed SD ≈ 0.165.
Detecting an effect that size at 80% power / α=0.05 needs roughly
`n ≈ 16·(0.165/0.064)² ≈ 106` seeds per arm — which is infeasible. 15 seeds is
the practical compromise: it will not establish a small true effect, but it is
enough to determine whether the *apparent* 12.6-point IQM gap replicates at all,
which is the actual question. Cost: 4 conditions × 15 seeds × ~3.3 min ≈ **3.3 h**.

The decisive prediction: if the analysis above is right,
`bsrs_c_e1_nonorm ≈ sparse_nonorm ≈ sparse_nonorm_advx2` (all within noise), and
only `sparse_nonorm_lr2x` will differ systematically.

### Issue 1 bottom line

**(i) What the evidence shows.** Scaling advantages by (1+η) is not equivalent to
scaling the LR by (1+η) — so P2 is not the specific confound Ghosh described.
But scaling advantages by (1+η) is very nearly a *no-op* under grad-clipping +
Adam (0.15% single-step, 2.8% over an epoch), and >99.7% of the training data has
`R^{γλ}=0` so coupled BSRS reduces to exactly that no-op rescaling. The observed
P2 gap has a bootstrap difference CI of [−0.180, +0.340] and is not
statistically supported.

**(ii) Confidence.** *High* on the mechanism (the invariance is analytic and the
measurements match closed-form predictions to 4 significant figures; the
`R^{γλ}=0` precondition is measured, not assumed). *High* that the reported P2
separation is not statistically supported at n=5. *Moderate* on the stronger
claim that the true effect is exactly zero — a ~3% per-step perturbation could
in principle compound, and n=5 cannot exclude a small real effect.

**(iii) What's needed.** The 4-condition × 15-seed run specified above (~3.3 h).
Nothing else is required; the mechanism question is settled analytically and
empirically.

---

# ISSUE 2 — Reconciling P1 ("indistinguishable") with P4 ("suppresses")

## 2a. Does `normalize_advantage` touch the value-regression targets?

**No. It is provably confined to the policy loss.** Two separate tensors are
involved, and they are set in different places at different times.

**Where both are written** — `ppo_shaped.py`, inside `collect_rollouts`, *before*
any training step:

```python
288        advantages = gae_advantages(
289            shaped_rewards, values_buf, next_values, dones_arr,
290            self.gamma, self.gae_lambda,
291        )
292        rollout_buffer.advantages = advantages.astype(np.float32)
293        rollout_buffer.returns = (advantages + values_buf).astype(np.float32)
```

Line 293 fixes the value-regression target as `advantages + values_buf` using the
**raw, unnormalized** GAE advantages, at rollout time. `ShapedPPO.train()`
(lines 312–318) delegates the update entirely to stock SB3 via `super().train()`
and never touches `.returns`.

**Where normalization happens** — `stable_baselines3/ppo/ppo.py`:

```python
216                advantages = rollout_data.advantages
218                if self.normalize_advantage and len(advantages) > 1:
219                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
225                policy_loss_1 = advantages * ratio
226                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
...
244                value_loss = F.mse_loss(rollout_data.returns, values_pred)
```

Line 219 **rebinds a local Python name**. That local `advantages` is consumed
only by lines 225–226 (the surrogate/policy loss). Line 244's value loss reads
`rollout_data.returns` — a different field, carrying the line-293 tensor,
untouched by the flag.

**Conclusion:** value targets are *never* normalized, under either setting of
`normalize_advantage`. "Success rate unaffected" (policy-side, normalized) and
"value targets affected" (critic-side, never normalized) are statements about
**two mechanically independent quantities**. They are compatible by construction;
there is no inconsistency in the implementation.

## 2b. Recomputed P1 statistics from `results_s2d/eval_s2d.csv`

Easy task:

| condition | per-seed success | IQM | 95% CI | mean |
|---|---|---|---|---|
| `sparse` | 0.02, 0.18, 0.04, 0.26, 0.12 | 0.113 | [0.027, 0.213] | 0.124 |
| `bsrs_c_e05` | 0.16, 0.06, 0.20, 0.02, 0.00 | 0.080 | [0.007, 0.187] | 0.088 |
| `bsrs_c_e1` | 0.06, 0.00, 0.00, 0.00, **0.80** | 0.020 | [0.000, 0.553] | 0.172 |
| `bsrs_c_e2` | 0.02, 0.00, 0.12, 0.08, 0.10 | 0.067 | [0.007, 0.113] | 0.064 |

Hard task: `sparse`, `bsrs_c_e05`, `bsrs_c_e1`, `bsrs_c_e2` are **all exactly
0.000** with CI [0.000, 0.000] — every seed of every condition scored 0/50.

Bootstrap CI on the difference vs `sparse` (easy):

| comparison | IQM diff | 95% CI | mean diff | 95% CI |
|---|---|---|---|---|
| η=0.5 − sparse | −0.033 | [−0.173, +0.100] | −0.036 | [−0.140, +0.068] |
| η=1 − sparse | −0.093 | [−0.207, +0.447] | +0.048 | [−0.172, +0.384] |
| η=2 − sparse | −0.047 | [−0.173, +0.053] | −0.060 | [−0.148, +0.024] |

Welch p-values: 0.561 (η=0.5), 0.782 (η=1), 0.276 (η=2).

**Is "indistinguishable" fair?** On the easy task, **yes for the aggregate
claim** — every difference CI straddles zero and no test approaches
significance. On the hard task it is fair but for a degenerate reason: *all*
conditions are identically zero, so it is a floor effect, not a demonstration of
equivalence. The framing is defensible but it **understates one thing**: it is
not that the conditions produce similar per-seed outcomes; it is that seed
variance is so large that nothing is resolvable (see 2c).

## 2c. Dose–response check

**There is no monotonic trend in success rate with η.** Easy-task IQM as η goes
0 → 0.5 → 1 → 2: **0.113 → 0.080 → 0.020 → 0.067** — non-monotonic, dipping at
η=1 and partially recovering at η=2. By mean: 0.124 → 0.088 → 0.172 → 0.064 —
also non-monotonic, with the ordering *reversed* at η=1 relative to IQM. Hard
task: 0 → 0 → 0 → 0, no trend possible.

**The one thing "indistinguishable" does gloss over is not a trend but an
outlier.** `bsrs_c_e1` has per-seed values `[0.06, 0.00, 0.00, 0.00, 0.80]` — one
seed reached **80% success** while the other four were ≤6%. IQM's 25% trim on
n=5 discards exactly that seed, which is why `bsrs_c_e1` reports the *lowest*
IQM (0.020) of any coupled condition while having the *highest* mean (0.172).
This single seed drives the anomalously wide CI [0.000, 0.553].

That outlier is worth flagging on its own terms: it shows that under coupled
shaping at η=1 the task is *sometimes* solvable to 80%, which is not visible
anywhere in the aggregate statistics. Whether that is a shaping effect or the
same seed-lottery visible in `her_final` (which had seeds at 0.98 and 0.76
alongside three near-zeros) cannot be determined from n=5.

## 2d. How P1 and P4 coexist

**They are not contradictory, and the reconciliation is mechanical rather than
interpretive.** Three facts compose:

1. **They describe different tensors.** P1 is about success rate, driven by the
   policy loss, which consumes the *normalized* advantage (ppo.py:219→225-226).
   P4 is about `mean |value target|`, i.e. `buffer.returns`, which is written
   once at ppo_shaped.py:293 from *unnormalized* advantages and is never
   normalized. No setting of `normalize_advantage` connects them.

2. **Normalization is exactly the operation that erases η from the policy
   side.** On sparse data `R^{γλ}=0`, so `A′=(1+η)A`. Mean/std normalization is
   invariant to positive scaling:
   `((1+η)A − mean((1+η)A)) / std((1+η)A) = (A − mean(A))/std(A)`. The policy
   loss therefore receives a tensor that is **bitwise independent of η** — P1's
   null result is not merely an empirical finding but an algebraic necessity
   under `normalize_advantage=True` on all-zero-reward batches.

3. **Nothing erases η from the critic side.** The value target is
   `returns′ = A′ + V̂ = (1+η)A − ηR^{γλ} + V̂`, regressed directly by
   `F.mse_loss` at ppo.py:244. η survives here at full strength.

So the coexistence is expected: **η is algebraically cancelled on the policy path
and algebraically retained on the critic path.** A study observing "no change in
success rate" and "measurable change in value targets" is observing exactly what
this implementation must produce.

> **[Erratum E5]** The paragraph below is wrong: the critic's self-consistent
> fixed point under coupled shaping is V̂ = V/(1+η), so *deflation* is the
> predicted direction. A controlled frozen-policy test now confirms it; see the
> Errata section at the top.

**One caveat on P4's stated direction.** The summary reports P4 as *refuted with
the effect reversed* — measured final `mean |value target|` on easy was 33.3
(η=0), 12.5 (η=0.5), 14.9 (η=1), 1.9 (η=2), i.e. shaping **suppresses** rather
than inflates. That is consistent with fact 3 above only in the sense that η has
*some* effect; the sign is a separate question. The algebra
`returns′ = (1+η)A − ηR^{γλ} + V̂` with `R^{γλ}≈0` predicts
`returns′ ≈ (1+η)A + V̂`, which for `|A|` large relative to `|V̂|` would predict
*growth* with η, not suppression. **The observed suppression is therefore not
explained by the identity alone**, and I could not close that gap from the
existing data. Two candidate explanations I could not distinguish without new
runs: (a) the critic is trained on these targets, so `V̂` adapts and `A = r + γV̂′ − V̂`
shrinks in a feedback loop that larger η accelerates — a self-consistency effect,
not a static rescaling; or (b) the η=2 run's near-zero targets (1.9) reflect a
policy that stopped reaching the goal at all, making it a downstream consequence
of behavior rather than a direct shaping effect. Distinguishing these needs the
per-update `adv_mean_prenorm` / `adv_std_prenorm` trajectories cross-referenced
against `V̂` magnitude, which is not currently logged.

### Issue 2 bottom line

**(i) What the evidence shows.** P1 and P4 are mechanically compatible and there
is no implementation inconsistency: `normalize_advantage` rebinds a local
consumed only by the policy loss (ppo.py:219, 225–226), while value targets are
fixed at ppo_shaped.py:293 from unnormalized advantages and never renormalized.
Under `normalize_advantage=True` on zero-reward batches, the policy loss is
provably independent of η, so P1's null is an algebraic necessity, not just an
empirical observation. "Indistinguishable" is a fair characterization of the
easy-task CIs (all difference CIs straddle zero; p ≥ 0.28) and is technically
true but degenerate on the hard task (universal floor at 0.000). There is **no
monotonic dose–response** in success rate with η. What the framing does hide is
a single `bsrs_c_e1` seed at 80% success that IQM's trim removes entirely.

**(ii) Confidence.** *Very high* that P1/P4 are not contradictory — this is
settled by reading the code, and the invariance argument in fact 2 is exact.
*High* on the recomputed statistics. *Low* on the explanation for P4's observed
*direction* (suppression rather than inflation), which the identity does not
predict and which I could not resolve from existing logs.

**(iii) What's needed.** Nothing further for the P1/P4 compatibility question —
it is resolved. For the unexplained *sign* of the P4 effect, the cheapest
discriminating evidence would be a re-analysis (no new training) of the existing
per-update logs correlating `value_target_mean_abs` against `adv_std_prenorm`
and mean `|V̂|` per condition — but mean `|V̂|` is not currently in `progress.csv`,
so it would require either re-deriving it from saved checkpoints at intervals
(only final checkpoints were kept, so this is not possible retroactively) or
adding one logged column and re-running a small subset. A 4-condition × 3-seed
easy-task run at η ∈ {0, 0.5, 1, 2} with `|V̂|` logged (~40 min) would settle it.

---

# Appendix — diagnostic scripts and raw output

Both scripts were written to the session scratch directory (not the repo) and
run with `PYTHONPATH=<repo>` against `results_s2d/easy/sparse_seed0/`. Neither
modifies any pipeline file or any data under `results_s2d/`.

## A1. `diag_lr_confound.py` — key excerpt

```python
def one_ppo_step(m, obs, act, old_v, old_lp, adv, ret):
    m.policy.set_training_mode(True)
    values, log_prob, entropy = m.policy.evaluate_actions(obs, act)
    values = values.flatten()
    ratio = th.exp(log_prob - old_lp)
    pl1 = adv * ratio
    pl2 = adv * th.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
    policy_loss = -th.min(pl1, pl2).mean()
    value_loss = F.mse_loss(ret, values)
    entropy_loss = -th.mean(entropy)
    loss = policy_loss + ENT_COEF * entropy_loss + VF_COEF * value_loss
    m.policy.optimizer.zero_grad(); loss.backward()
    th.nn.utils.clip_grad_norm_(m.policy.parameters(), MAX_GRAD_NORM)
    m.policy.optimizer.step()

runs["a_baseline"]   = (build(env, BASE_LR,           start_sd), adv.clone(),             BASE_LR)
runs["b_adv_scaled"] = (build(env, BASE_LR,           start_sd), adv.clone() * (1 + ETA), BASE_LR)
runs["c_lr_scaled"]  = (build(env, BASE_LR*(1 + ETA), start_sd), adv.clone(),             BASE_LR*(1+ETA))
```

Raw output:

```
   TOTAL trainable params = 11663   value-side=5633  policy-side=6030
rollout: nonzero_raw_reward_frac=0.00000  (whole 2048-step rollout)
minibatch(64): |A| mean=2.68372 std=2.94770 nonzero=64/64
[a_baseline   ] lr=3.0e-04  policy_loss=+0.855862  value_loss=9.285669  |grad|preclip=52.7620
                |dtheta| total=3.236300e-02  value-side=2.247076e-02  policy-side=2.329030e-02
[b_adv_scaled ] lr=3.0e-04  policy_loss=+1.711724  value_loss=9.285669  |grad|preclip=53.9297
                |dtheta| total=3.236391e-02  value-side=2.246997e-02  policy-side=2.329233e-02
[c_lr_scaled  ] lr=6.0e-04  policy_loss=+0.855862  value_loss=9.285669  |grad|preclip=52.7620
                |dtheta| total=6.472631e-02  value-side=4.494151e-02  policy-side=4.658064e-02
|b|/|a| = 1.0000    |c|/|a| = 2.0000
cos(b,a)=1.000006   cos(c,a)=1.000004   cos(b,c)=1.000002
|b-c|/|a| = 1.0000   |b-a|/|a| = 0.0015
```

## A2. `diag_multistep.py` — raw output

```
rollout nonzero_raw_reward_frac = 0.000000  -> R^(gamma*lambda) is identically 0,
                                              so A' = (1+eta)*A EXACTLY here
=== (A) gradient decomposition, one minibatch ===
  |grad policy_loss|           = 6.4446
  |grad vf_coef*value_loss|    = 52.3671
  |grad ent_coef*entropy_loss| = 0.0265
  |grad TOTAL|                 = 52.7620    max_grad_norm=0.5  -> clipped by 105.5x
=== (B) cumulative ||theta_k - theta_0|| over 32 consecutive updates ===
   step     a_baseline   b_adv_scaled    c_lr_scaled     b/a     c/a
      1   3.236300e-02   3.236391e-02   6.472631e-02  1.0000  2.0000
      2   5.223807e-02   5.222597e-02   1.043690e-01  0.9998  1.9979
      4   8.312792e-02   8.311922e-02   1.641893e-01  0.9999  1.9751
      8   1.350983e-01   1.351237e-01   2.582125e-01  1.0002  1.9113
     16   2.077282e-01   2.077535e-01   3.971683e-01  1.0001  1.9120
     24   2.751540e-01   2.748965e-01   4.801016e-01  0.9991  1.7448
     32   3.183074e-01   3.172935e-01   5.389513e-01  0.9968  1.6932
  cos(b,a)=0.999615  cos(c,a)=0.878107  cos(b,c)=0.880450
  |b-a|/|a| = 0.0279   |b-c|/|a| = 0.9426
```

## A3. Statistical re-analysis command

Bootstrap CIs on differences used the study's own `s2d_stats.iqm` with 10,000
resamples, resampling seeds independently within each condition. Welch t-tests
and Mann–Whitney U reported as supplementary checks only.

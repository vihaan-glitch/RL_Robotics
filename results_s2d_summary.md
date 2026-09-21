# Sparse→Dense Reward Generation Study — Summary

> **Status note (2026-09-21).** This is the first-pass summary written
> immediately after the 110 runs. Several verdicts below were revised after
> review:
>
> - **P2 is retracted.** The 15.3% vs 2.7% gap is an IQM-trimming artifact
>   at n = 5; the bootstrap CI on the difference is [−0.180, +0.340].
> - **P4's deflation is the predicted direction.** A controlled test confirms
>   the critic fixed point V̂ = V/(1+η).
> - **"Stratified" bootstrap** should read *percentile bootstrap over seeds*.
> - The easy/sparse CI upper bound should be 23.3%, not 21.3%.
> - **HER's gain over sparse is not significant** on either task (paired
>   difference test).
>
> See `review_report.md` (errata table) and `diagnostics/`. Apart from that one
> CI endpoint, the numbers in the tables below are correct as computed.

110 runs = 11 conditions × 2 tasks (easy / hard-with-obstacles) × 5 seeds, 500k
steps/run, PPO hyperparameters identical to the prior ablation **except
`norm_reward=False`** (intentional; see Limitations). All conditions train on
`reward_mode="sparse"` except `oracle` (`reward_mode="full"`). Evaluation: 50
deterministic episodes/run. Aggregation: IQM + 95% stratified percentile
bootstrap CI (10,000 resamples; `rliable` was attempted but its `arch`
dependency fails to import in this environment, so the bootstrap in
`s2d_stats.py` implements the spec's manual fallback).

## Verdict table — P1–P4

| # | Prediction | Verdict | Evidence |
|---|---|---|---|
| **P1** | Coupled BSRS ≈ sparse baseline (advantage normalization annihilates the (1+η) scaling), possibly phase-dependent | **SUPPORTED**, no phase-dependence observed | Easy IQM success: sparse 11.3%, `bsrs_c_e05` 8.0%, `bsrs_c_e1` 2.0%, `bsrs_c_e2` 6.7% — all within/below the sparse floor's own 95% CI [2.7%, 21.3%]. Hard: sparse and all three coupled variants are exactly 0%. The predicted phase-dependence (collapse weakening once successes populate batches) does **not** appear: the nonzero-raw-reward batch fraction stays below 1.5% for the *entire* 500k-step budget on every condition checked (sparse: 0.00%→0.49%; `bsrs_c_e1`: 0.00%→0.60%), so batches remain effectively all-zero-reward throughout — the collapse condition never relaxes. |
| **P2** | With `normalize_advantage=False`, coupled BSRS acts as an effective policy LR multiplier | **SUPPORTED on easy (noisy, n=5); uninformative on hard** | Easy IQM success: `sparse_nonorm` 2.7% vs `bsrs_c_e1_nonorm` 15.3% — a real separation appears only once normalization is off, exactly where the (1+η) scaling in the coupled identity is no longer renormalized away. `bsrs_c_e1_nonorm` (15.3%) also clearly beats its normalized counterpart `bsrs_c_e1` (2.0%). Hard task: both floor at 0% (no signal). |
| **P3** | Decoupled potential escapes the collapse and beats coupled BSRS on sparse tasks | **REFUTED** | Easy IQM success: `bsrs_d_e05` 4.0%, `bsrs_d_e1` 4.7%, `bsrs_d_e2` 3.3% — all ≤ their coupled counterparts and all below the sparse floor (11.3%). Decoupling did not escape anything; if anything it did slightly worse. Hard: all three decoupled variants are ≈0% (`bsrs_d_e1` mean 0.4%, still not meaningfully above the 0% floor), same as coupled. |
| **P4** | Coupled BSRS inflates critic value targets, growing with η | **REFUTED — effect runs in the opposite direction** | Easy, final mean｜value target｜ (avg. across seeds, last logged update): η=0 (sparse) **33.3**, η=0.5 **12.5**, η=1 **14.9**, η=2 **1.9** — targets shrink as η grows, not inflate. Hard: small and noisy, no clear η-trend (η=0: 0.24, η=0.5: 1.18, η=1: 0.21, η=2: 1.39). Mechanistic reading: since `returns'_t = (1+η)A_t − η·R^{γλ}_t + V̂(s_t)`, the `η·(A_t − R^{γλ}_t)` term is a *correction* built from the same critic used to construct Φ, and empirically it net-cancels rather than adds to the raw target magnitude — larger η pulls the regression target toward, not away from, zero. |

## Full IQM table (95% bootstrap CI in brackets)

### Easy task
| condition | success IQM | 95% CI | first-success (IQM steps) |
|---|---|---|---|
| sparse | 11.3% | [2.7%, 21.3%] | 75,196 |
| oracle | 100.0% | [100.0%, 100.0%] | 11,813 |
| bsrs_c_e05 | 8.0% | [0.7%, 18.7%] | 58,433 |
| bsrs_c_e1 | 2.0% | [0.0%, 55.3%]† | 128,138 |
| bsrs_c_e2 | 6.7% | [0.7%, 11.3%] | 49,674 |
| bsrs_c_e1_nonorm | 15.3% | [0.7%, 36.7%] | 41,804 |
| sparse_nonorm | 2.7% | [0.7%, 30.7%] | 22,300 |
| bsrs_d_e05 | 4.0% | [0.0%, 10.0%] | 89,031 |
| bsrs_d_e1 | 4.7% | [0.7%, 21.3%] | 90,364 |
| bsrs_d_e2 | 3.3% | [0.7%, 14.0%] | 76,147 |
| her_final | 26.7% | [0.0%, 90.7%]† | 46,484 |

† Extremely wide CI — driven by bimodal per-seed outcomes (see below), not by estimator instability alone.

### Hard task
| condition | success IQM | 95% CI | first-success (IQM steps) |
|---|---|---|---|
| sparse | 0.0% | [0.0%, 0.0%] | 247,107 |
| oracle | 8.0% | [4.7%, 14.0%] | 22,690 |
| bsrs_c_e05 | 0.0% | [0.0%, 0.0%] | 170,171 |
| bsrs_c_e1 | 0.0% | [0.0%, 0.0%] | 149,570 |
| bsrs_c_e2 | 0.0% | [0.0%, 0.0%] | 233,402 |
| bsrs_c_e1_nonorm | 0.0% | [0.0%, 0.0%] | 383,221 |
| sparse_nonorm | 0.0% | [0.0%, 0.0%] | 267,300 |
| bsrs_d_e05 | 0.0% | [0.0%, 0.0%] | never |
| bsrs_d_e1 | 0.0% | [0.0%, 1.3%] | 190,246 |
| bsrs_d_e2 | 0.0% | [0.0%, 0.0%] | 336,717 |
| her_final | 0.7% | [0.0%, 2.0%] | 68,886 |

## Which method closed more of the sparse→oracle gap?

**PPO-HER (final relabeling) closed the most gap on both tasks — and by a wide
margin over every BSRS variant, coupled or decoupled.**

- **Easy:** `her_final` closes **17.3%** of the sparse→oracle gap (IQM success
  26.7%, vs. floor 11.3% / ceiling 100%). Every coupled and decoupled BSRS
  variant except `bsrs_c_e1_nonorm` (+4.5%) closes *negative* gap — i.e.
  underperforms the sparse floor itself.
- **Hard:** `her_final` again closes the most gap (**8.3%**), but the ceiling
  is tiny here (oracle only reaches 8.0% IQM success), so in absolute terms
  this is ~0.7 percentage points of success — a real but very small effect.
  Every BSRS variant closes 0% gap on hard (indistinguishable from the sparse
  floor).
- **Decoupled BSRS did not outperform coupled BSRS on either task** (contradicting
  the motivation for P3): both families sit at or below the sparse floor.

The one caveat on HER's win: it is the **least reliable** method in the study.
Per-seed easy-task success was bimodal — `[4%, 98%, 0%, 76%, 0%]` — two seeds
"caught" and drove success into the 70–98% range, three stayed near zero. Hard
task was uniformly near-zero across all 5 seeds `[0%, 0%, 2%, 0%, 2%]`. The
relabeled fraction of each batch averaged 18.7% (easy) / 31.9% (hard) at the
end of training, confirming the relabeling pipeline is active and substantial,
but a genuinely large fraction of runs never accumulate enough real successes
to seed useful hindsight goals in the first place.

## Limitations

- **n=5 seeds is enough to run the design, not enough to separate the BSRS
  variants from each other.** Most 95% CIs for coupled/decoupled conditions
  overlap heavily with the sparse floor and with each other; the ordering among
  η values (0.5 vs 1 vs 2) should not be read as a real ranking without more
  seeds.
- **`norm_reward=False` is a deliberate change from the prior ablation study**,
  so these sparse/oracle baselines are a fresh reference point, not a repeat of
  the earlier `results/`/`results_hard/` numbers. Oracle success on hard here
  (8.0% IQM) is noticeably lower than the prior 5-seed hard-task ablation's
  `full`-mode result (~14.4%) — plausibly the compounding effect of no reward
  normalization on top of an already-hard task, not a contradiction, but the
  two studies are not directly comparable.
- **The hard task is close to a floor effect for 9 of 11 conditions** (true 0%
  point estimate). This gives the study very little power to detect *any* true
  effect on hard — P2 and P3's hard-task verdicts rest on almost no signal, and
  should be read as "uninformative," not as confirmation of the easy-task
  pattern.
- **HER's bimodality** suggests a threshold/luck effect tied to when the first
  few real successes appear during training (before that, there is nothing to
  relabel from), rather than a smooth, reliable benefit. This is worth
  investigating with more seeds or a curriculum/warm-start before treating HER
  as a dependable fix for this task's sparsity.
- **rliable is unusable in this environment** (its `arch` dependency raises
  `TypeError: deprecate_kwarg() missing 1 required positional argument` at
  import); the manual IQM + percentile-bootstrap implementation in
  `s2d_stats.py` was validated against toy examples but not cross-checked
  against rliable's own numbers on this data.
- **The full 110-run grid was used** (the `--trim` flag in `run_s2d_study.py`,
  which would drop `bsrs_c_e2`/`bsrs_d_e2` on hard, was available but not
  exercised — all η∈{0.5,1,2} points are present on both tasks).

---
*Figures: `figures_s2d/F1`–`F7` (captions in `figures_s2d/captions.md`).
Seed-level results: `results_s2d/eval_s2d.csv`. Aggregates:
`results_s2d/eval_s2d_aggregated.csv`.*

# Figure captions — sparse→dense study

**F1.** Final success rate (IQM over 5 seeds, ±95% stratified bootstrap CI, 10k
resamples) per condition, easy and hard tasks. Dotted line = sparse floor,
dashed = oracle ceiling.

**F2.** Evaluation success rate vs training timestep for sparse, oracle, coupled
(η=1), decoupled (η=1), and PPO-HER; mean ±1 SD across seeds.

**F3.** Final success vs η for coupled and decoupled BSRS, with sparse/oracle
reference lines. Tests whether decoupling escapes the coupled collapse (P3).

**F4.** P2: final success for sparse vs coupled-η1 with advantage normalization
ON, and their `_nonorm` counterparts with it OFF.

**F5.** P4: mean |critic value target| over training for coupled η∈{0,0.5,1,2}
(η=0 ≡ sparse). Growth with η indicates value-target inflation.

**F6.** PPO-HER diagnostics over training: relabeled fraction, clip fraction on
original vs relabeled transitions, and policy entropy (easy solid, hard dashed).

**F7.** Fraction of each update batch with nonzero raw reward over training —
the phase variable behind P1 (advantage-norm annihilation is exact only while
batches are all-zero-reward).

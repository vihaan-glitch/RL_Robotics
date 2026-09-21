"""Every condition vs sparse, per task: IQM/mean difference bootstrap CIs
(unpaired and seed-paired), Welch t-test, Mann–Whitney U. No training.
Usage (repo root): python diagnostics/stats_vs_sparse.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats as sps
from s2d_stats import iqm
df = pd.read_csv("results_s2d/eval_s2d.csv")
R = 10_000
def get(t, c):
    s = df[(df.task == t) & (df.condition == c)].sort_values("seed")
    return s.seed.values, s.success_rate.values
def unpaired(x, y, agg, seed=0):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (R, len(x))); iy = rng.integers(0, len(y), (R, len(y)))
    d = np.array([agg(y[b]) - agg(x[a]) for a, b in zip(ix, iy)])
    return agg(y) - agg(x), *np.percentile(d, [2.5, 97.5]), (d <= 0).mean()
def paired(x, y, seed=0):  # resample seed indices jointly (seed i shares init weights + env seed)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (R, len(x)))
    dd = y - x; d = dd[idx].mean(1)
    return dd.mean(), *np.percentile(d, [2.5, 97.5]), (d <= 0).mean()
conds = ["oracle","bsrs_c_e05","bsrs_c_e1","bsrs_c_e2","sparse_nonorm","bsrs_c_e1_nonorm",
         "bsrs_d_e05","bsrs_d_e1","bsrs_d_e2","her_final"]
rows = []
for t in ["easy", "hard"]:
    sx, x = get(t, "sparse")
    for c in conds:
        sy, y = get(t, c); assert (sx == sy).all()
        di, lo_i, hi_i, p_i = unpaired(x, y, iqm)
        dm, lo_m, hi_m, p_m = unpaired(x, y, np.mean)
        dp, lo_p, hi_p, p_p = paired(x, y)
        if np.all(x == x[0]) and np.all(y == y[0]):
            pw = pu = float("nan")
        else:
            pw = sps.ttest_ind(y, x, equal_var=False).pvalue
            pu = sps.mannwhitneyu(y, x, alternative="two-sided").pvalue
        excl = "EXCLUDES 0" if (lo_i > 0 or hi_i < 0) else ""
        rows.append((t, c, di, lo_i, hi_i, dm, lo_m, hi_m, lo_p, hi_p, pw, pu, excl))
print(f"{'task':<5}{'condition':<18}{'ΔIQM':>7}{'IQM-diff 95% CI':>18}{'Δmean':>7}{'mean-diff CI (unpaired)':>25}{'paired CI':>18}{'Welch p':>9}{'MWU p':>8}")
for t, c, di, a, b, dm, e, f, g, h, pw, pu, excl in rows:
    print(f"{t:<5}{c:<18}{di:>+7.3f}  [{a:+.3f},{b:+.3f}]{dm:>+7.3f}     [{e:+.3f},{f:+.3f}]    [{g:+.3f},{h:+.3f}]{pw:>9.3f}{pu:>8.3f}  {excl}")

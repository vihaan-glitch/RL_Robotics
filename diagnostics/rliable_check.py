"""Recompute IQM + 95% CI with rliable and compare to s2d_stats.py's published
numbers. rliable's `arch` dependency needs `pandas<3` to import.
Usage (repo root): python diagnostics/rliable_check.py [repo_dir]"""
import sys, numpy as np, pandas as pd
from rliable import library as rly, metrics
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
df = pd.read_csv(f"{REPO}/results_s2d/eval_s2d.csv")
agg = pd.read_csv(f"{REPO}/results_s2d/eval_s2d_aggregated.csv")
print("aggregated cols:", list(agg.columns))
scores = {}
for (t, c), g in df.groupby(["task", "condition"]):
    scores[f"{t}|{c}"] = g.sort_values("seed").success_rate.values.reshape(-1, 1)  # (5 runs, 1 task)
f = lambda x: np.array([metrics.aggregate_iqm(x)])
pts, cis = rly.get_interval_estimates(scores, f, reps=10000)
print(f"\n{'task|condition':<24}{'rliable IQM':>12}{'rliable CI':>20}{'s2d IQM':>9}{'s2d CI':>20}{'max|Δ| CI':>10}")
worst = 0
for k in sorted(scores):
    t, c = k.split("|")
    row = agg[(agg.task == t) & (agg.condition == c)].iloc[0]
    lo, hi = cis[k][0][0], cis[k][1][0]
    s_pt = row[[x for x in agg.columns if "iqm" in x.lower() and "lo" not in x.lower() and "hi" not in x.lower() and "success" in x.lower()][0]]
    s_lo = row[[x for x in agg.columns if "success" in x.lower() and "lo" in x.lower()][0]]
    s_hi = row[[x for x in agg.columns if "success" in x.lower() and "hi" in x.lower()][0]]
    d = max(abs(lo - s_lo), abs(hi - s_hi)); worst = max(worst, d)
    print(f"{k:<24}{pts[k][0]:>12.4f}   [{lo:.4f},{hi:.4f}]{s_pt:>9.4f}   [{s_lo:.4f},{s_hi:.4f}]{d:>10.4f}")
print(f"\nmax |Δ| in any CI endpoint: {worst:.4f}   (point estimates: exact match expected)")

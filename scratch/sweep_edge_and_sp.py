import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scratch')
sys.path.insert(0, 'feature engineering scripts')
from optimize_peak_bidirectional_system import evaluate_params

print("Sweep across Edge and ShortMinPct:")
for edge in [7.0, 7.5, 8.0]:
    for sp in [0.33, 0.40, 0.50, 0.57]:
        evaluate_params(edge=edge, trend_th=12.0, short_pct=sp, cooldown=45.0)

import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scratch')
sys.path.insert(0, 'feature engineering scripts')
from optimize_peak_bidirectional_system import evaluate_params

for sp in [0.33, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    print(f"Testing short_min_pct = {sp}")
    evaluate_params(edge=7.5, trend_th=12.0, short_pct=sp, cooldown=45.0)

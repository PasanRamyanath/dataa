import sys
sys.path.insert(0, 'feature engineering scripts/scratch')
from simulate_hierarchical_reentry import simulate
tds = simulate(180, 10.0, 5.0)
for t in tds:
    print(f"{t['cycle']:10s} | Bar {t['entry_b']:3d} | {t['side']:4s} | gross={t['gross']:+6.2f}b | net={t['net']:+6.2f}b | {t['reason']}")

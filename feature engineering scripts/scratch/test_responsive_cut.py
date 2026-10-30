import sys
from pathlib import Path

# Replace invalidation condition with responsive model-driven cut:
p = Path("feature engineering scripts/scratch/simulate_hierarchical_reentry.py")
code = p.read_text(encoding="utf-8")
code_mod = code.replace(
    "if unreal <= -7.0 and pdir == opp_dir and edge >= 12.0 and (d_meso * sign) < -0.15 and (d_tact * sign) < -0.15:",
    "if unreal <= -6.0 and (((d_tact * sign) < -0.08 and (d_micro * sign) < -0.05) or (d_meso * sign) < -0.10):"
)
Path("feature engineering scripts/scratch/sim_responsive_cut.py").write_text(code_mod, encoding="utf-8")

sys.path.insert(0, "feature engineering scripts/scratch")
import sim_responsive_cut
tds = sim_responsive_cut.simulate(180, 8.0, 5.0)
n = len(tds)
w = sum(1 for t in tds if t['net'] > 0)
net = sum(t['net'] for t in tds)
gw = sum(t['net'] for t in tds if t['net'] > 0)
gl = abs(sum(t['net'] for t in tds if t['net'] < 0))
pf = (gw / gl) if gl > 0 else 999.0
print(f"Trades: {n} | Wins: {w} ({w/n*100.1:.1f}%) | Net PnL: {net:+.2f} bps | PF: {pf:.2f}")
for t in tds:
    print(f"{t['cycle']:10s} | Bar {t['entry_b']:3d} | {t['side']:4s} | gross={t['gross']:+6.2f}b | net={t['net']:+6.2f}b | {t['reason']}")

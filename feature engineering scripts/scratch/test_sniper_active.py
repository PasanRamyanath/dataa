import sys
from pathlib import Path

# Modify sim to enforce Tactical & Micro active confirmation on entry:
p = Path("feature engineering scripts/scratch/simulate_hierarchical_reentry.py")
code = p.read_text(encoding="utf-8")

# 1. Responsive Invalidation:
code = code.replace(
    "if unreal <= -7.0 and pdir == opp_dir and edge >= 12.0 and (d_meso * sign) < -0.15 and (d_tact * sign) < -0.15:",
    "if unreal <= -6.0 and (((d_tact * sign) < -0.08 and (d_micro * sign) < -0.05) or (d_meso * sign) < -0.10):"
)

# 2. Strict Sniper Entry Confirmation (Tactical & Micro must agree):
entry_target = """if pdir in (0, 2) and edge >= min_edge_bps and mode in ("PASSIVE_MAKER_SNIPER", "URGENT_HYBRID", "WAIT_FOR_MICRO_SNIPER"):"""
entry_replacement = """sign_target = 1.0 if pdir == 2 else -1.0
                tact_confirms = (sign_target * d_tact) >= 0.05
                micro_confirms = (sign_target * d_micro) >= 0.0
                if pdir in (0, 2) and edge >= min_edge_bps and tact_confirms and micro_confirms:"""
code = code.replace(entry_target, entry_replacement)

Path("feature engineering scripts/scratch/sim_sniper_active.py").write_text(code, encoding="utf-8")

sys.path.insert(0, "feature engineering scripts/scratch")
import sim_sniper_active
tds = sim_sniper_active.simulate(180, 8.0, 5.0)
n = len(tds)
w = sum(1 for t in tds if t['net'] > 0)
net = sum(t['net'] for t in tds)
gw = sum(t['net'] for t in tds if t['net'] > 0)
gl = abs(sum(t['net'] for t in tds if t['net'] < 0))
pf = (gw / gl) if gl > 0 else 999.0
print(f"TRADES: {n} | WINS: {w} ({w/n*100.1:.1f}%) | NET PNL: {net:+.2f} bps | PF: {pf:.2f}")
for t in tds:
    print(f"{t['cycle']:10s} | Bar {t['entry_b']:3d} | {t['side']:4s} | gross={t['gross']:+6.2f}b | net={t['net']:+6.2f}b | {t['reason']}")

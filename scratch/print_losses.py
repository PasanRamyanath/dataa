import sys
sys.path.append('.')
from scratch.test_cycle65_with_quantile_and_wide_sl import all_closed_trades

losses = [t for t in all_closed_trades if t['net'] <= 0]
print(f"Total losses: {len(losses)}")
for l in losses:
    print(f"{l['cycle']:<20} | Side: {l['side']} | Net: {l['net']:+.2f} bps | Dur: {l['dur']:.1f}s | Reason: {l['reason']}")

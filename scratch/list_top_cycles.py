import sys
sys.path.insert(0, 'scratch')
from deep_cycle_analysis import parse_changelog

cycles = parse_changelog()
top = sorted([c for c in cycles if c.get('net_pnl', 0) > 0], key=lambda x: x['net_pnl'], reverse=True)
print(f"Total profitable cycles documented in changelog: {len(top)}")
for c in top:
    print(f"Cycle {c['cycle']:2d} | Net: {c['net_pnl']:+6.2f} bps | WR: {c['win_rate']:5.1f}% | Trades: {c['trades']:2d} | Winner: {c['winner']}")

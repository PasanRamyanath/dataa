import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Find trade 3 entry in Cycle 59
# Challenger 3 Trade 3: entry_price: 80674.2, entry_ts_ms: 1793296320000
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if abs(d.get('bbo_mid', 0) - 80674.2) < 5.0:
            print(f"Bar #{d.get('bar_idx')}: Mid={d.get('bbo_mid')}, Tact={d.get('tactical_probs')}, Meso={d.get('meso_probs')}, Dir={d.get('primary_direction')}, Edge={d.get('expected_net_edge_bps')}")

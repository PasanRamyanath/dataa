import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total bars in ledger: {len(lines)}")
for b in lines[-12:]:
    print(f"Bar {b.get('bar_idx')}: Price={b.get('bbo_mid')} Dir={b.get('primary_direction')} Edge={b.get('expected_net_edge_bps')}")
    print(f"   Macro: {b.get('macro_probs')}")
    print(f"   Meso:  {b.get('meso_probs')}")
    print(f"   Tact:  {b.get('tactical_probs')}")
    print(f"   Micro: {b.get('micro_probs')}")

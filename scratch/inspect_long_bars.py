import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

long_bars = [b for b in bars if b.get('primary_direction') == 2]
print(f"Total LONG bars: {len(long_bars)}")
for b in long_bars[:10]:
    print(f"Bar {b.get('bar_idx')}: Price={b.get('bbo_mid')} Edge={b.get('expected_net_edge_bps')}")
    print(f"   Macro: {b.get('macro_probs')}")
    print(f"   Meso:  {b.get('meso_probs')}")
    print(f"   Tact:  {b.get('tactical_probs')}")
    print(f"   Micro: {b.get('micro_probs')}")

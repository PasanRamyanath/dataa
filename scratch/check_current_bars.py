import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total bars in ledger: {len(lines)}")
# Let's inspect the last 15 bars
for d in lines[-15:]:
    bar = d.get('bar_idx')
    price = d.get('bbo_mid')
    macro = d.get('macro_probs')
    meso = d.get('meso_probs')
    tact = d.get('tactical_probs')
    micro = d.get('micro_probs')
    edge = d.get('expected_net_edge_bps')
    direction = d.get('primary_direction')
    fsm = d.get('fsm_state')
    print(f"Bar {bar}: Price={price} | Dir={direction} | Edge={edge} | FSM={fsm}")
    print(f"   Macro: {macro}")
    print(f"   Meso:  {meso}")
    print(f"   Tact:  {tact}")
    print(f"   Micro: {micro}")

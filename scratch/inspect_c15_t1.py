import json

with open("scratch/archive_cycle15/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

for b in bars[6:12]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    fsm = b["fsm_state"]
    macro = b["macro_probs"]
    meso = b["meso_probs"]
    tact = b["tactical_probs"]
    micro = b["micro_probs"]
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | FSM: {fsm}")
    print(f"       Macro: {macro} | Meso: {meso}")

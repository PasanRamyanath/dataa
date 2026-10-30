import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

print("=== BARS 185 TO 222 ===")
for b in bars[184:222]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    fsm = b["fsm_state"]
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | FSM: {fsm}")

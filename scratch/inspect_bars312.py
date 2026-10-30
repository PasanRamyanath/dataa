import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

print("=== BARS 308 TO 325 ===")
for b in bars[307:325]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    fsm = b["fsm_state"]
    pos = b.get("active_position")
    unreal = 0.0
    if pos:
        entry = pos["entry_price"]
        unreal = (mid - entry) / entry * 10000.0 if pos["side"] == "BUY" else (entry - mid) / entry * 10000.0
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | Unreal: {unreal:+.2f} bps | FSM: {fsm}")

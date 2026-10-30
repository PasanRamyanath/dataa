import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

for b in bars:
    idx = b["bar_idx"]
    if idx in [124, 125, 126, 127, 134, 135, 136]:
        mid = b["bbo_mid"]
        direction = b["primary_direction"]
        edge = b["expected_net_edge_bps"]
        fsm = b["fsm_state"]
        pos = b.get("active_position")
        closed = b["closed_trades_count"]
        print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | FSM: {fsm} | Closed: {closed}")
        if pos:
            print(f"       Pos: side={pos['side']}, entry={pos['entry_price']}, peak={pos['peak_favorable_excursion_bps']:.2f} bps, armed={pos['trailing_maker_armed']}")

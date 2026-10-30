import json

with open("scratch/archive_cycle11/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

print("=== BARS 215 TO 226 ===")
for b in bars[214:226]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"] # 0: SHORT, 1: FLAT, 2: LONG
    edge = b["expected_net_edge_bps"]
    mode = b["execution_mode"]
    fsm = b["fsm_state"]
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | Mode: {mode} | FSM: {fsm}")

print("\n=== BARS 222 TO 346 (TRADE 2 LIFETIME) ===")
# Sample every 10 bars
for b in bars[221:348:10]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    macro = b["macro_probs"]
    meso = b["meso_probs"]
    tact = b["tactical_probs"]
    micro = b["micro_probs"]
    pos = b.get("active_position")
    unreal = 0.0
    if pos:
        entry = pos["entry_price"]
        unreal = (mid - entry) / entry * 10000.0 if pos["side"] == "BUY" else (entry - mid) / entry * 10000.0
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | Unreal: {unreal:+.2f} bps | Macro: {macro} | Meso: {meso}")

print("\n=== BARS AROUND SL EXIT (340 TO 348) ===")
for b in bars[339:349]:
    idx = b["bar_idx"]
    mid = b["bbo_mid"]
    direction = b["primary_direction"]
    edge = b["expected_net_edge_bps"]
    fsm = b["fsm_state"]
    trades_cnt = b["closed_trades_count"]
    print(f"Bar {idx:03d} | Mid: {mid:.2f} | Dir: {direction} | Edge: {edge:+.2f} | FSM: {fsm} | ClosedTrades: {trades_cnt}")

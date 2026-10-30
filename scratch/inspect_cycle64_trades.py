import json

with open("D:/PrivateProjects/BTC Futures/scratch/live_hierarchical_session_1h_ledger.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

print(f"Total bars logged: {len(lines)}")
prev_cnt = 0
for idx, b in enumerate(lines):
    cnt = b.get("closed_trades_count", 0)
    pos = b.get("active_position")
    prev_pos = lines[idx-1].get("active_position") if idx > 0 else None
    
    if pos is not None and prev_pos is None:
        print(f"\n[ENTRY] Bar #{b['bar_idx']} ({b['datetime']}) Mid: {b['bbo_mid']}")
        print(f"  Side: {pos['side']} | Entry Px: {pos['entry_price']} | SL: {pos.get('sl_price')} | TP: {pos.get('tp_price')}")
    
    if cnt > prev_cnt:
        print(f"[EXIT] Bar #{b['bar_idx']} ({b['datetime']}) Mid: {b['bbo_mid']}")
        print(f"  Cum PnL: {b['cumulative_realized_pnl_bps']:.2f} bps | Exited Position: {prev_pos}")
        prev_cnt = cnt

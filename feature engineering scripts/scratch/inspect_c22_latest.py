import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print(f"Total rows recorded: {len(rows)}")
if rows:
    last = rows[-1]
    print(f"Last bar: #{last.get('bar_idx')} | Mid: ${last.get('bbo_mid', 0):.2f} | FSM: {last.get('fsm_state')} | Closed Trades: {last.get('closed_trades_count')} | Realized PnL: {last.get('cumulative_realized_pnl_bps', 0.0):+.2f} bps")
    print(f"Active Pos: {last.get('active_position')}")
    print(f"Edge: {last.get('expected_net_edge_bps')} | Mode: {last.get('execution_mode')}")
    
    # Check all closed trades
    for i in range(1, len(rows)):
        if rows[i].get("closed_trades_count", 0) > rows[i-1].get("closed_trades_count", 0):
            pos = rows[i-1].get("active_position")
            print(f"Trade #{rows[i]['closed_trades_count']}: Bar {rows[i]['bar_idx']} | Pos: {pos}")

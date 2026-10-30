import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get('active_position') is not None or d.get('closed_trades_count', 0) > 0:
            print(f"Bar {d.get('bar_idx')}: pos={d.get('active_position')} closed={d.get('closed_trades_count')} pnl={d.get('cumulative_realized_pnl_bps')}")

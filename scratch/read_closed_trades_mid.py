import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    trades_seen = 0
    for line in f:
        d = json.loads(line)
        c = d.get('closed_trades_count', 0)
        if c > trades_seen:
            trades_seen = c
            print(f"Bar {d['bar_idx']:03d} | PnL: {d.get('cumulative_realized_pnl_bps', 0):+.2f}b | Mid: {d['bbo_mid']} | Total Trades: {c}")

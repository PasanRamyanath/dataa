import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

trades_seen = 0
with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('closed_trades_count', 0) > trades_seen:
            trades_seen = d.get('closed_trades_count')
            print(f"Bar #{d.get('bar_idx')}: Closed trades count={trades_seen} | Realized PnL={d.get('cumulative_realized_pnl_bps'):+.2f} bps | State={d.get('fsm_state')} | Pos={d.get('active_position')} | Mid={d.get('bbo_mid')}")

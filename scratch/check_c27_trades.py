import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    prev_closed = 0
    for line in f:
        d = json.loads(line)
        closed = d.get('closed_trades_count', 0)
        pos = d.get('active_position')
        fsm = d.get('fsm_state')
        bar = d.get('bar_idx')
        pnl = d.get('cumulative_realized_pnl_bps', 0.0)
        mid = d.get('bbo_mid')
        if pos is not None or closed != prev_closed or fsm not in ('IDLE', 'SNIPER_ENTRY_STALKING'):
            print(f"Bar {bar:03d} | mid={mid:.2f} | fsm={fsm} | pos={pos is not None} | closed={closed} | cum_pnl={pnl:+.2f}")
            prev_closed = closed

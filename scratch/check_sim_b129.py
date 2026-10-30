import sys, os, json
sys.path.insert(0, os.path.abspath('feature engineering scripts'))
sys.path.insert(0, os.path.abspath('.'))
from scratch.sim_cycle60_candidates import Challenger3_ExhaustionAware, DummyDecision

with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = f.readlines()

eng = Challenger3_ExhaustionAware(symbol='BTCUSDT')
for i, l in enumerate(lines[:135]):
    r = json.loads(l)
    dec = DummyDecision(r)
    eng.on_market_micro_tick(ts_ms=r['ts_ms'], bbo_bid=r['bbo_bid'], bbo_ask=r['bbo_ask'], bbo_bid_qty=1.0, bbo_ask_qty=1.0)
    st = eng.on_hierarchical_decision(decision=dec, bbo_bid=r['bbo_bid'], bbo_ask=r['bbo_ask'], bbo_bid_qty=1.0, bbo_ask_qty=1.0, equity_usdt=10000.0, intra_bar_open=r['bbo_mid'])
    if r['bar_idx'] in (128, 129, 130):
        print("Bar", r['bar_idx'], "state=", eng.state, "pos=", eng.active_position.side if eng.active_position else 'NONE')
        sess_high = max(eng.rolling_15m_prices) if eng.rolling_15m_prices else 0
        sess_low = min(eng.rolling_15m_prices) if eng.rolling_15m_prices else 0
        print("  high=", sess_high, "low=", sess_low, "len=", len(eng.rolling_15m_prices))
        print("  is_violation=", eng._is_range_boundary_violation(0, r['bbo_mid']))

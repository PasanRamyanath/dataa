import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from execution.champion_challenger_framework import ChampionChallengerTournamentManager
from execution.hierarchical_execution_engine import HierarchicalSignalDecision

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")

print(f"Simulating {len(bars)} recorded bars through all 6 tournament competitors...")

# We will trace why competitors do or do not enter
for bar in bars:
    ts_ms = bar.get('ts_ms')
    mid = bar.get('bbo_mid')
    bid = bar.get('bbo_bid', mid - 0.05)
    ask = bar.get('bbo_ask', mid + 0.05)
    bid_qty = 1.0
    ask_qty = 1.0
    
    dec = HierarchicalSignalDecision(
        bar_idx=bar.get('bar_idx'),
        ts_ms=ts_ms,
        macro_probs=tuple(bar.get('macro_probs')),
        meso_probs=tuple(bar.get('meso_probs')),
        tactical_probs=tuple(bar.get('tactical_probs')),
        micro_probs=tuple(bar.get('micro_probs')),
        primary_direction=bar.get('primary_direction'),
        expected_net_edge_bps=bar.get('expected_net_edge_bps'),
        execution_mode=bar.get('execution_mode'),
        kelly_fraction=1.0,
        suggested_limit_price=bid if bar.get('primary_direction') == 2 else ask,
        fsm_action="NONE"
    )
    
    mgr.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
    mgr.on_hierarchical_decision(decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)

print("\n=== SIMULATION RESULTS ===")
for name, comp in mgr.competitors.items():
    eng = comp.engine
    print(f"{name}: State={eng.state} | Trades={len(eng.closed_trades)} | ActivePos={eng.active_position is not None}")
    if eng.active_orders:
        print(f"   Active Orders: {eng.active_orders}")

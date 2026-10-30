import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('feature engineering scripts'))

from execution.champion_challenger_framework import ChampionChallengerTournamentManager
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

import json

# Read all ledger lines up to bar 60
records = []
with open('scratch/archive_cycle59/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")

print("Replaying bars 1 to 65...")
for r in records[:65]:
    bar_idx = r["bar_idx"]
    ts_ms = r["ts_ms"]
    bid = r["bbo_bid"]
    ask = r["bbo_ask"]
    mid = r["bbo_mid"]
    
    dec = HierarchicalDecisionOutput(
        ts_ms=ts_ms,
        macro_probs=r["macro_probs"],
        meso_probs=r["meso_probs"],
        tactical_probs=r["tactical_probs"],
        micro_probs=r["micro_probs"],
        primary_direction=r["primary_direction"],
        expected_net_edge_bps=r["expected_net_edge_bps"],
        target_allocation_usd=1000.0,
        kelly_fraction=0.10,
        conformal_quantiles=(0.0, 0.0),
        regime_label="TEST",
        regime_volatility=0.001
    )
    
    # Tick simulation
    mgr.on_market_micro_tick(ts_ms=ts_ms, price=mid, volume=1.0)
    mgr.on_hierarchical_decision(
        decision=dec,
        bbo_bid=bid,
        bbo_ask=ask,
        bbo_bid_qty=1.0,
        bbo_ask_qty=1.0,
        equity_usdt=10000.0,
        intra_bar_open=mid
    )
    
    # Check positions at bar 56
    if bar_idx in (55, 56, 57):
        print(f"\n--- Bar #{bar_idx} ---")
        for name, comp in mgr.competitors.items():
            pos = comp.engine.active_position
            print(f"[{name}] state={comp.engine.state}, pos={pos.side if pos else 'NONE'} @ {pos.entry_price if pos else 0}")

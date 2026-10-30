import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")

for name, comp in mgr.competitors.items():
    comp.engine.ticks_seen = 20  # Set warmed up

print(f"Total bars to evaluate: {len(bars)}")

trade_triggers = {name: [] for name in mgr.competitors}

for b_idx, bar in enumerate(bars):
    ts_ms = bar.get('ts_ms')
    mid = bar.get('bbo_mid')
    bid = bar.get('bbo_bid', mid - 0.05)
    ask = bar.get('bbo_ask', mid + 0.05)
    bid_qty = 5.0
    ask_qty = 5.0
    
    macro = tuple(bar.get('macro_probs'))
    meso = tuple(bar.get('meso_probs'))
    tact = tuple(bar.get('tactical_probs'))
    micro = tuple(bar.get('micro_probs'))
    prim_dir = bar.get('primary_direction')
    edge = bar.get('expected_net_edge_bps')
    
    dec = HierarchicalDecisionOutput(
        ts_ms=ts_ms,
        macro_probs=macro,
        meso_probs=meso,
        tactical_probs=tact,
        micro_probs=micro,
        fused_swing_probs=(0.33, 0.34, 0.33),
        primary_direction=prim_dir,
        cross_horizon_alignment=0.8,
        expected_gross_move_bps=20.0,
        expected_net_edge_bps=edge,
        meta_act_probability=0.8,
        micro_sniper_trigger_prob=0.8,
        target_holding_seconds=120,
        take_profit_bps=15.0,
        stop_loss_bps=12.0,
        kelly_position_fraction=1.0,
        execution_mode="MAKER_STALKING"
    )
    
    mgr.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
    mgr.on_hierarchical_decision(decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
    
    for name, comp in mgr.competitors.items():
        if comp.engine.state != "IDLE":
            trade_triggers[name].append((b_idx, comp.engine.state, mid, edge))

print("=== TRIGGERED TRADES / STALKING STATES ===")
for name, trigs in trade_triggers.items():
    print(f"{name}: {len(trigs)} events | {trigs[:5]}")

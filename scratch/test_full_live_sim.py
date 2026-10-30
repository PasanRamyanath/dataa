import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
champ = mgr.competitors["Champion"].engine

# EXACT initial state as live session:
print("Initial champ.ticks_seen:", champ.ticks_seen)

for i, bar in enumerate(bars):
    mid = bar.get('bbo_mid')
    dec = HierarchicalDecisionOutput(
        ts_ms=bar.get('ts_ms'),
        macro_probs=tuple(bar.get('macro_probs')),
        meso_probs=tuple(bar.get('meso_probs')),
        tactical_probs=tuple(bar.get('tactical_probs')),
        micro_probs=tuple(bar.get('micro_probs')),
        fused_swing_probs=(0.33, 0.34, 0.33),
        primary_direction=bar.get('primary_direction'),
        cross_horizon_alignment=0.8,
        expected_gross_move_bps=20.0,
        expected_net_edge_bps=bar.get('expected_net_edge_bps'),
        meta_act_probability=0.8,
        micro_sniper_trigger_prob=0.8,
        target_holding_seconds=120,
        take_profit_bps=15.0,
        stop_loss_bps=12.0,
        kelly_position_fraction=1.0,
        execution_mode="MAKER_STALKING"
    )
    
    # 1. on_market_micro_tick
    champ.on_market_micro_tick(
        ts_ms=bar.get('ts_ms'),
        bbo_bid=mid-0.05,
        bbo_ask=mid+0.05,
        bbo_bid_qty=10.0,
        bbo_ask_qty=5.0,
        bar_high=mid+1.0,
        bar_low=mid-1.0,
        current_decision=dec
    )
    
    # 2. on_hierarchical_decision
    champ.on_hierarchical_decision(
        decision=dec,
        bbo_bid=mid-0.05,
        bbo_ask=mid+0.05,
        bbo_bid_qty=10.0,
        bbo_ask_qty=5.0
    )
    
    if i in (0, 1, 2, 5, 9, 10, 11, 20, 50, 100, 200):
        print(f"Bar {i:3d}: State={champ.state:20s} | ticks_seen={champ.ticks_seen:3d} | Pos={champ.active_position is not None} | Trades={len(champ.closed_trades)}")

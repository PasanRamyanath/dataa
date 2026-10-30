import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
champ = mgr.competitors["Champion"].engine
champ.ticks_seen = 50

# Bar 0
bar0 = bars[0]
mid0 = bar0.get('bbo_mid')
dec0 = HierarchicalDecisionOutput(
    ts_ms=bar0.get('ts_ms'),
    macro_probs=tuple(bar0.get('macro_probs')),
    meso_probs=tuple(bar0.get('meso_probs')),
    tactical_probs=tuple(bar0.get('tactical_probs')),
    micro_probs=tuple(bar0.get('micro_probs')),
    fused_swing_probs=(0.33, 0.34, 0.33),
    primary_direction=bar0.get('primary_direction'),
    cross_horizon_alignment=0.8,
    expected_gross_move_bps=20.0,
    expected_net_edge_bps=bar0.get('expected_net_edge_bps'),
    meta_act_probability=0.8,
    micro_sniper_trigger_prob=0.8,
    target_holding_seconds=120,
    take_profit_bps=15.0,
    stop_loss_bps=12.0,
    kelly_position_fraction=1.0,
    execution_mode="MAKER_STALKING"
)
champ.on_market_micro_tick(ts_ms=bar0.get('ts_ms'), bbo_bid=mid0-0.05, bbo_ask=mid0+0.05, bbo_bid_qty=10.0, bbo_ask_qty=5.0)
champ.on_hierarchical_decision(decision=dec0, bbo_bid=mid0-0.05, bbo_ask=mid0+0.05, bbo_bid_qty=10.0, bbo_ask_qty=5.0)
print("After Bar 0 Decision: State =", champ.state)

# Bar 1 arrives (10 seconds later)
bar1 = bars[1]
mid1 = bar1.get('bbo_mid')
dec1 = HierarchicalDecisionOutput(
    ts_ms=bar1.get('ts_ms'),
    macro_probs=tuple(bar1.get('macro_probs')),
    meso_probs=tuple(bar1.get('meso_probs')),
    tactical_probs=tuple(bar1.get('tactical_probs')),
    micro_probs=tuple(bar1.get('micro_probs')),
    fused_swing_probs=(0.33, 0.34, 0.33),
    primary_direction=bar1.get('primary_direction'),
    cross_horizon_alignment=0.8,
    expected_gross_move_bps=20.0,
    expected_net_edge_bps=bar1.get('expected_net_edge_bps'),
    meta_act_probability=0.8,
    micro_sniper_trigger_prob=0.8,
    target_holding_seconds=120,
    take_profit_bps=15.0,
    stop_loss_bps=12.0,
    kelly_position_fraction=1.0,
    execution_mode="MAKER_STALKING"
)

# In run_live_session:
# It calls on_market_micro_tick with current_decision=dec1
champ.on_market_micro_tick(
    ts_ms=bar1.get('ts_ms'),
    bbo_bid=mid1-0.05,
    bbo_ask=mid1+0.05,
    bbo_bid_qty=10.0,
    bbo_ask_qty=5.0,
    bar_high=mid1+1.0,
    bar_low=mid1-1.0,
    current_decision=dec1
)
print("After Bar 1 on_market_micro_tick: State =", champ.state)
if champ.active_position is not None:
    print("ACTIVE POSITION FILLED!", champ.active_position)

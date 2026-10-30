import sys
sys.path.insert(0, 'feature engineering scripts')
from execution.champion_challenger_framework import ChampionChallengerTournamentManager
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
champ = mgr.competitors["Champion"].engine

dec = HierarchicalDecisionOutput(
    ts_ms=1000,
    macro_probs=(0.2, 0.4, 0.4),
    meso_probs=(0.2, 0.4, 0.4),
    tactical_probs=(0.2, 0.4, 0.4),
    micro_probs=(0.2, 0.4, 0.4),
    fused_swing_probs=(0.33, 0.34, 0.33),
    primary_direction=2,
    cross_horizon_alignment=0.8,
    expected_gross_move_bps=20.0,
    expected_net_edge_bps=15.0,
    meta_act_probability=0.8,
    micro_sniper_trigger_prob=0.8,
    target_holding_seconds=120,
    take_profit_bps=15.0,
    stop_loss_bps=12.0,
    kelly_position_fraction=1.0,
    execution_mode="MAKER_STALKING"
)

for i in range(15):
    print(f"Before call {i}: ticks_seen = {champ.ticks_seen}")
    champ.on_hierarchical_decision(dec, 80000.0, 80000.1, 1.0, 1.0)
    print(f"After call {i}:  ticks_seen = {champ.ticks_seen}")

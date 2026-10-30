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

bar = bars[0]
ts_ms = bar.get('ts_ms')
mid = bar.get('bbo_mid')
bid = bar.get('bbo_bid', mid - 0.05)
ask = bar.get('bbo_ask', mid + 0.05)
bid_qty = 10.0
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

# Step 1: on_market_micro_tick
champ.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
# Step 2: on_hierarchical_decision
res = champ.on_hierarchical_decision(decision=dec, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
print("Result of on_hierarchical_decision:", res)
print("State after decision:", champ.state)
print("Resting entry side:", champ.resting_entry_side)
print("Resting entry price:", champ.resting_entry_price)

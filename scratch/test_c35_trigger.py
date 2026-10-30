import sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import Cycle35_ReversionPureScalper

eng = Cycle35_ReversionPureScalper(symbol="BTCUSDT")
eng.ticks_seen = 20

# Simulate a legitimate Long breakout (like Bar 300 in Cycle 57)
dec = HierarchicalDecisionOutput(
    ts_ms=1000,
    macro_probs=(0.15, 0.35, 0.50),  # d_macro = +0.35
    meso_probs=(0.15, 0.35, 0.50),   # d_meso = +0.35
    tactical_probs=(0.10, 0.30, 0.60),# d_tact = +0.50
    micro_probs=(0.15, 0.35, 0.50),  # d_micro = +0.35
    fused_swing_probs=(0.15, 0.35, 0.50),
    primary_direction=2,
    cross_horizon_alignment=0.85,
    expected_gross_move_bps=25.0,
    expected_net_edge_bps=20.0,
    meta_act_probability=0.85,
    micro_sniper_trigger_prob=0.85,
    target_holding_seconds=120,
    take_profit_bps=15.0,
    stop_loss_bps=12.0,
    kelly_position_fraction=1.0,
    execution_mode="MAKER_STALKING"
)

# Test 1: Seed price history
for p in [81000.0, 81020.0, 81040.0, 81060.0]:
    eng.rolling_15m_prices.append(p)
    eng.recent_mid_prices.append((1000, p))

print("Initial State:", eng.state)
eng.on_market_micro_tick(ts_ms=1000, bbo_bid=81050.0, bbo_ask=81050.1, bbo_bid_qty=10.0, bbo_ask_qty=5.0)
res = eng.on_hierarchical_decision(dec, bbo_bid=81050.0, bbo_ask=81050.1, bbo_bid_qty=10.0, bbo_ask_qty=5.0)
print("Result of on_hierarchical_decision:", res)
print("State after decision:", eng.state)
print("Resting entry side:", eng.resting_entry_side)
print("Resting entry price:", eng.resting_entry_price)

# Next tick (fill)
eng.on_market_micro_tick(ts_ms=1010, bbo_bid=81050.0, bbo_ask=81050.1, bbo_bid_qty=10.0, bbo_ask_qty=5.0, bar_high=81055.0, bar_low=81049.0, current_decision=dec)
print("State after next tick:", eng.state)
print("Active pos:", eng.active_position is not None)

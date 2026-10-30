import sys
import json
from pathlib import Path

sys.path.insert(0, "feature engineering scripts")
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

ledger_path = Path("scratch/archive_cycle19/live_hierarchical_session_1h_ledger.jsonl")
with open(ledger_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

engine = HierarchicalSniperExecutionEngine(symbol="BTCUSDT")

for r in rows:
    b = r["bar_idx"]
    now_ms = r["ts_ms"]
    mid = r["bbo_mid"]
    bid = r.get("bbo_bid", mid - 0.05)
    ask = r.get("bbo_ask", mid + 0.05)
    
    dec = HierarchicalDecisionOutput(
        ts_ms=now_ms,
        macro_probs=tuple(r["macro_probs"]),
        meso_probs=tuple(r["meso_probs"]),
        tactical_probs=tuple(r["tactical_probs"]),
        micro_probs=tuple(r["micro_probs"]),
        fused_swing_probs=(0.33, 0.33, 0.34),
        primary_direction=r["primary_direction"],
        cross_horizon_alignment=0.5,
        expected_gross_move_bps=20.0,
        expected_net_edge_bps=r["expected_net_edge_bps"],
        meta_act_probability=0.65,
        micro_sniper_trigger_prob=0.75,
        target_holding_seconds=1800,
        take_profit_bps=45.0,
        stop_loss_bps=14.5,
        kelly_position_fraction=0.5,
        execution_mode=r["execution_mode"],
    )
    
    # 1. on_market_micro_tick
    engine.on_market_micro_tick(
        ts_ms=now_ms,
        bbo_bid=bid,
        bbo_ask=ask,
        bbo_bid_qty=1.0,
        bbo_ask_qty=1.0,
        aggressive_sell_vol_at_bid=0.0,
        aggressive_buy_vol_at_ask=0.0,
        eth_1s_ret_bps=0.0,
        bar_high=ask,
        bar_low=bid,
        current_decision=dec,
    )
    
    # 2. on_hierarchical_decision
    engine.on_hierarchical_decision(
        decision=dec,
        bbo_bid=bid,
        bbo_ask=ask,
        bbo_bid_qty=1.0,
        bbo_ask_qty=1.0,
        equity_usdt=10000.0,
        intra_bar_open=mid,
    )

print(f"REPLAY OF CYCLE 19 WITH UPGRADED ENGINE:")
print(f"Total Trades: {len(engine.closed_trades)}")
for i, t in enumerate(engine.closed_trades, 1):
    print(f"Trade {i}: {t.side} | Entry: {t.entry_price:.2f} | Exit: {t.exit_price:.2f} | Gross: {t.gross_return_bps:+6.2f} bps | Net: {t.net_return_bps:+6.2f} bps | MFE: {t.peak_mfe_bps:+6.2f} bps | Reason: {t.exit_reason}")

total_net = sum(t.net_return_bps for t in engine.closed_trades)
wins = sum(1 for t in engine.closed_trades if t.net_return_bps > 0)
wr = (wins / len(engine.closed_trades) * 100.0) if engine.closed_trades else 0.0
print(f"Summary: Wins: {wins}/{len(engine.closed_trades)} ({wr:.1f}%) | Total Net PnL: {total_net:+.2f} bps")

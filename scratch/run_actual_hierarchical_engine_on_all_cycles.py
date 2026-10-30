"""
scratch/run_actual_hierarchical_engine_on_all_cycles.py
Run the real HierarchicalSniperExecutionEngine directly on all 5 cycles.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path("feature engineering scripts").resolve()))

from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
import numpy as np

cycles = [25, 26, 27, 28, 29]

for c in cycles:
    ledger_path = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        continue
    
    with open(ledger_path) as f:
        bars = [json.loads(line) for line in f]
        
    engine = HierarchicalSniperExecutionEngine(symbol="BTCUSDT")
    
    for b in bars:
        dec = HierarchicalDecisionOutput(
            ts_ms=b["ts_ms"],
            primary_direction=b["primary_direction"],
            macro_probs=np.array(b["macro_probs"]),
            meso_probs=np.array(b["meso_probs"]),
            tactical_probs=np.array(b["tactical_probs"]),
            micro_probs=np.array(b["micro_probs"]),
            fused_swing_probs=np.array(b["tactical_probs"]),
            cross_horizon_alignment=0.8,
            expected_gross_move_bps=15.0,
            expected_net_edge_bps=b["expected_net_edge_bps"],
            meta_act_probability=0.75,
            micro_sniper_trigger_prob=0.6,
            target_holding_seconds=300,
            take_profit_bps=25.0,
            stop_loss_bps=15.0,
            kelly_position_fraction=0.1,
            execution_mode=b.get("execution_mode", "NONE"),
        )
        bbo_bid = b["bbo_bid"]
        bbo_ask = b["bbo_ask"]
        mid = b["bbo_mid"]
        
        # Micro tick first
        engine.on_market_micro_tick(
            ts_ms=b["ts_ms"],
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            bar_high=mid + 0.5,
            bar_low=mid - 0.5,
            current_decision=dec,
        )
        # Decision second
        engine.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=1.0,
            bbo_ask_qty=1.0,
            equity_usdt=10000.0,
            intra_bar_open=mid,
        )
        
    trades = engine.closed_trades
    n = len(trades)
    w = sum(1 for t in trades if t.net_return_bps > 0)
    net = sum(t.net_return_bps for t in trades)
    wr = (w / n * 100.0) if n > 0 else 0.0
    print(f"\n==================== CYCLE {c} BASE ENGINE ====================")
    print(f"Trades: {n} | Wins: {w} | WR: {wr:.1f}% | Net: {net:+.2f} bps")
    for i, t in enumerate(trades, 1):
        print(f"  Trade {i}: {t.side:<4} @ {t.entry_price:.1f} -> {t.exit_price:.1f} | Net: {t.net_return_bps:>+6.2f} bps | MFE: {t.peak_mfe_bps:>+5.2f}b | {t.exit_reason}")

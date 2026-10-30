"""
scratch/test_cycle35_tournament_simulation.py
Simulates the Cycle 35 Tournament across Cycle 33 and Cycle 34 ledgers
to verify that all 6 competitors execute properly and produce superior results.
"""
import sys
sys.path.insert(0, "feature engineering scripts")
import json
from pathlib import Path
from execution.champion_challenger_framework import ChampionChallengerTournamentManager
from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput

def simulate_cycle(cycle_num: int):
    ledger_path = Path(f"scratch/archive_cycle{cycle_num}/live_hierarchical_session_1h_ledger.jsonl")
    if not ledger_path.exists():
        print(f"Cycle {cycle_num} ledger not found: {ledger_path}")
        return

    bars = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))

    mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")

    for b in bars:
        dec = HierarchicalDecisionOutput(
            macro_probs=b["macro_probs"],
            meso_probs=b["meso_probs"],
            tactical_probs=b["tactical_probs"],
            micro_probs=b["micro_probs"],
            fused_swing_probs=b.get("fused_swing_probs", b["tactical_probs"]),
            primary_direction=b["primary_direction"],
            cross_horizon_alignment=b.get("cross_horizon_alignment", 1.0),
            expected_gross_move_bps=b.get("expected_gross_move_bps", 12.0),
            expected_net_edge_bps=b.get("expected_net_edge_bps", 6.0),
            meta_act_probability=b.get("meta_act_probability", 0.8),
            micro_sniper_trigger_prob=b.get("micro_sniper_trigger_prob", 0.7),
            target_holding_seconds=b.get("target_holding_seconds", 300),
            take_profit_bps=b.get("take_profit_bps", 30.0),
            stop_loss_bps=b.get("stop_loss_bps", 14.5),
            kelly_position_fraction=b.get("kelly_position_fraction", 0.5),
            execution_mode="PASSIVE_MAKER_SNIPER",
            ts_ms=b["ts_ms"],
        )

        bbo_bid = b["bbo_bid"]
        bbo_ask = b["bbo_ask"]
        bbo_bid_qty = b.get("bbo_bid_qty", 1.5)
        bbo_ask_qty = b.get("bbo_ask_qty", 1.5)
        bar_high = b.get("bar_high", bbo_ask)
        bar_low = b.get("bar_low", bbo_bid)

        mgr.on_market_micro_tick(
            ts_ms=b["ts_ms"],
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            bar_high=bar_high,
            bar_low=bar_low,
            current_decision=dec,
        )

        mgr.on_hierarchical_decision(
            decision=dec,
            bbo_bid=bbo_bid,
            bbo_ask=bbo_ask,
            bbo_bid_qty=bbo_bid_qty,
            bbo_ask_qty=bbo_ask_qty,
            equity_usdt=10000.0,
            intra_bar_open=b.get("intra_bar_open", bbo_bid),
        )

    print("=" * 90)
    print(f"TOURNAMENT SIMULATION RESULTS FOR CYCLE {cycle_num}")
    print("=" * 90)
    for name, c in mgr.competitors.items():
        tr = c.engine.closed_trades
        n_tr = len(tr)
        wins = sum(1 for t in tr if t.net_return_bps > 0)
        wr = (wins / n_tr * 100) if n_tr > 0 else 0.0
        tot_net = sum(t.net_return_bps for t in tr)
        print(f"  {name:15s} | Trades: {n_tr:2d} | Wins: {wins:2d} | WR: {wr:5.1f}% | Net PnL: {tot_net:+6.2f} bps | Profile: {c.strategy_profile}")
        for i, t in enumerate(tr, 1):
            print(f"      T{i}: {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} | Net: {t.net_return_bps:+6.2f} bps | MFE: {t.peak_mfe_bps:+5.2f}b | Hold: {t.holding_duration_sec:.0f}s | Reason: {t.exit_reason}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "feature engineering scripts")
    simulate_cycle(34)
    simulate_cycle(33)

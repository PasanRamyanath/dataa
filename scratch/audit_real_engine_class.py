import os
import glob
import json
import collections
import io
import sys
import types

sys.path.append("feature engineering scripts")
from execution.champion_challenger_framework import Cycle65_UnifiedPeakReversionEngine

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archives = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

def run_real_engine_audit():
    total_trades = []
    active_sessions = 0
    wr100_sessions = 0
    profitable_sessions = 0
    
    for arch in archives:
        ledgers = glob.glob(os.path.join(base_dir, arch, "*ledger*.jsonl"))
        if not ledgers: continue
        
        engine = Cycle65_UnifiedPeakReversionEngine(symbol="BTCUSDT")
        session_trades = []
        session_net = 0.0
        
        with open(ledgers[0], "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip().startswith("{"): continue
                bar = json.loads(line)
                
                mid = bar.get("bbo_mid", bar.get("mid", 0.0))
                bid = bar.get("bbo_bid", mid - 0.05)
                ask = bar.get("bbo_ask", mid + 0.05)
                ts_ms = bar.get("ts_ms", idx * 10000)
                
                # Mock HierarchicalDecision
                decision = types.SimpleNamespace(
                    ts_ms=ts_ms,
                    primary_direction=bar.get("primary_direction", bar.get("direction", 1)),
                    conformal_confidence=bar.get("conformal_confidence", 0.6),
                    expected_net_edge_bps=bar.get("expected_net_edge_bps", bar.get("edge", 0.0)),
                    tactical_probs=tuple(bar.get("tactical_probs", [0.33, 0.33, 0.33])),
                    meso_probs=tuple(bar.get("meso_probs", [0.33, 0.33, 0.33])),
                    macro_probs=tuple(bar.get("macro_probs", [0.33, 0.33, 0.33])),
                    micro_probs=tuple(bar.get("micro_probs", [0.33, 0.33, 0.33])),
                    ofi_imbalance=bar.get("ofi_imbalance", 0.0),
                    ofi_z_score=bar.get("ofi_z_score", 0.0),
                    execution_mode=bar.get("execution_mode", "PASSIVE_MAKER_SNIPER"),
                    kelly_position_fraction=bar.get("kelly_position_fraction", 0.25)
                )
                
                engine.on_hierarchical_decision(
                    decision=decision,
                    bbo_bid=bid,
                    bbo_ask=ask,
                    bbo_bid_qty=bar.get("bbo_bid_qty", 10.0),
                    bbo_ask_qty=bar.get("bbo_ask_qty", 10.0),
                    equity_usdt=10000.0,
                    intra_bar_open=bar.get("intra_bar_open", mid)
                )
                
        # Check closed trades
        for t in engine.closed_trades:
            rec = {
                "cycle": arch,
                "side": t.side,
                "net": t.net_return_bps,
                "gross": t.gross_return_bps,
                "dur": t.holding_duration_sec,
                "mfe": t.peak_mfe_bps,
                "mae": t.max_mae_bps,
                "reason": t.exit_reason
            }
            session_trades.append(rec)
            total_trades.append(rec)
            session_net += t.net_return_bps
            
        if session_trades:
            active_sessions += 1
            w = sum(1 for t in session_trades if t["net"] > 0)
            if w == len(session_trades):
                wr100_sessions += 1
            if session_net > 0:
                profitable_sessions += 1
                
    total_w = sum(1 for t in total_trades if t["net"] > 0)
    wr = (total_w / len(total_trades) * 100.0) if total_trades else 0.0
    tot_pnl = sum(t["net"] for t in total_trades)
    
    print("="*80)
    print("CYCLE 65 UNIFIED PEAK REVERSION ENGINE - PRODUCTION CLASS SIMULATION")
    print("="*80)
    print(f"Total Trades: {len(total_trades)} across {active_sessions} active sessions")
    print(f"Wins: {total_w} | Losses: {len(total_trades) - total_w}")
    print(f"Win Rate: {wr:.2f}%")
    print(f"Total Net Realized PnL: {tot_pnl:+.2f} bps")
    print(f"Average Profit per Trade: {(tot_pnl/len(total_trades)):+.2f} bps" if total_trades else "N/A")
    print(f"100% Win Rate Sessions: {wr100_sessions} / {active_sessions} ({wr100_sessions/max(1,active_sessions)*100:.1f}%)")
    print(f"Profitable Sessions: {profitable_sessions} / {active_sessions} ({profitable_sessions/max(1,active_sessions)*100:.1f}%)")
    
    # Analyze MAE and drops
    maes = [t["mae"] for t in total_trades]
    losses = [t for t in total_trades if t["net"] <= 0]
    print("\nAdverse Excursion (Drop) Audit:")
    print(f"  Worst Adverse Excursion (Drop): {min(maes):.2f} bps" if maes else "N/A")
    print(f"  Median Adverse Excursion: {sorted(maes)[len(maes)//2]:.2f} bps" if maes else "N/A")
    print(f"  Total Losing Trades (Stopped out or timed out): {len(losses)}")
    for l in losses:
        print(f"    Cycle: {l['cycle']:<18} Net: {l['net']:>7.2f}b Dur: {l['dur']:>6.1f}s MAE: {l['mae']:>7.2f}b Reason: {l['reason']}")

    reasons = collections.Counter(t["reason"] for t in total_trades)
    print(f"\nExit Reasons Breakdown:")
    for r, cnt in reasons.items():
        print(f"  {r:<35}: {cnt} trades")

if __name__ == "__main__":
    run_real_engine_audit()

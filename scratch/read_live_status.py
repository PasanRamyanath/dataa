import json
from pathlib import Path

hb_path = Path("D:/PrivateProjects/BTC Futures/scratch/live_session_1h_heartbeat.json")
lb_path = Path("D:/PrivateProjects/BTC Futures/scratch/champion_challenger_leaderboard.json")
ledger_path = Path("D:/PrivateProjects/BTC Futures/scratch/live_hierarchical_session_1h_ledger.jsonl")

print("="*80)
print("LIVE CYCLE 64 REAL-TIME TELEMETRY STATUS")
print("="*80)

if hb_path.exists():
    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            hb = json.load(f)
        print(f"Elapsed Time:   {hb.get('elapsed_minutes')} min / {hb.get('target_minutes')} min ({hb.get('progress_pct')}%)")
        print(f"Bars Evaluated: {hb.get('bars_evaluated')} | Current BTC Mid: ${hb.get('current_price', 0):.2f}")
        print(f"Champion Pos:   {hb.get('active_position')}")
        print(f"Champion Trade: {hb.get('trades_count')} trades | {hb.get('wins_count')} wins | WR: {hb.get('win_rate_pct')}% | Net: {hb.get('net_pnl_bps', 0):+.2f} bps")
        print("\nMulti-Horizon Prediction Accuracy:")
        for tf, data in hb.get("accuracy_summary", {}).items():
            print(f"  - {tf:>3s}: Overall Acc = {data.get('overall_accuracy_pct', 0):.1f}% | Active Prec = {data.get('active_precision_pct', 0):.1f}% ({data.get('evaluated_samples', 0)} samples)")
    except Exception as e:
        print(f"Error reading heartbeat: {e}")

if lb_path.exists():
    try:
        with open(lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)
        print("\nTournament Leaderboard:")
        print(f"{'Rank':4s} | {'Competitor':14s} | {'Trades':6s} | {'Wins':4s} | {'Win Rate':8s} | {'Net PnL':10s} | {'PF':5s} | {'Active Pos':20s}")
        print("-" * 80)
        for r in lb.get("leaderboard", []):
            print(f"#{r.get('rank', 0):2d}   | {r.get('name', ''):14s} | {r.get('trades', 0):6d} | {r.get('wins', 0):4d} | {r.get('win_rate_pct', 0):7.1f}% | {r.get('net_pnl_bps', 0):+9.2f}b | {r.get('profit_factor', 1.0):5.2f} | {r.get('active_pos', 'NONE'):20s}")
            print(f"      Avg Trade: {r.get('avg_trade_bps', 0):+.2f}b | Avg Duration: {r.get('avg_duration_sec', 0):.1f}s | Profile: {r.get('strategy_profile', '')}")
    except Exception as e:
        print(f"Error reading leaderboard: {e}")

if ledger_path.exists():
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        print(f"\nLedger Records: {len(lines)} total bars logged.")
        # Find where trade closed
        for i, l in enumerate(lines):
            if l.get("closed_trades_count", 0) > 0 and (i == 0 or lines[i-1].get("closed_trades_count", 0) == 0):
                print(f"Trade 1 closed at Bar {l.get('bar_idx')} ({l.get('datetime')})")
                print(f"  BBO Mid at Close: ${l.get('bbo_mid'):.2f} | Net Realized PnL: {l.get('cumulative_realized_pnl_bps'):+.2f} bps")
                break
    except Exception as e:
        print(f"Error reading ledger: {e}")

print("="*80)

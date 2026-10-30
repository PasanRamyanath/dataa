import os
import json
import glob

base = "D:/PrivateProjects/BTC Futures/scratch"

cycles_to_audit = [
    ("archive_cycle31", "Cycle 31 (Highest Frequency 100% WR: 5 trades, 100% WR)"),
    ("archive_cycle35", "Cycle 35 (Highest Profit/Trade 100% WR: +22.32 bps/tr, +44.64 bps net)"),
    ("archive_v3_2_session2", "V3.2 Session 2 (100% WR: 3 trades, +11.45 bps net, 0 taker stops)"),
    ("archive_cycle43", "Cycle 43 (Stepped Ratchet 100% WR: 3 trades, +7.60 bps net)"),
    ("archive_cycle40", "Cycle 40 (100% WR: 2 trades, +6.03 bps net)"),
    ("archive_cycle60", "Cycle 60 (Highest Total Net Surge: 5 trades, 80% WR, +62.20 bps net)")
]

for folder, title in cycles_to_audit:
    p = os.path.join(base, folder)
    print("\n" + "="*100)
    print(f"AUDITING: {title} [{folder}]")
    print("="*100)
    
    # Check for metrics json
    metrics_files = glob.glob(os.path.join(p, "*metrics*.json")) + glob.glob(os.path.join(p, "*final*.json")) + glob.glob(os.path.join(p, "*heartbeat*.json"))
    leaderboard_files = glob.glob(os.path.join(p, "*leaderboard*.json"))
    ledger_files = glob.glob(os.path.join(p, "*ledger*.jsonl"))
    
    # 1. Print Leaderboard if exists
    if leaderboard_files:
        with open(leaderboard_files[0], "r", encoding="utf-8") as f:
            lb = json.load(f)
            print("Leaderboard Standings:")
            for item in lb.get("leaderboard", [])[:3]:
                print(f"  Rank {item.get('rank')}: {item.get('name')} | Trades: {item.get('trades')} | WR: {item.get('win_rate_pct')}% | PnL: {item.get('net_pnl_bps')} bps | Profile: {item.get('strategy_profile', '')}")
    
    # 2. Print Accuracy & Closed Trades from metrics
    for mf in metrics_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "accuracy_summary" in data:
                    acc = data["accuracy_summary"]
                    print("\nPrediction Accuracy Summary:")
                    for tier, vals in acc.items():
                        if tier == "tiers":
                            for k, v in vals.items():
                                print(f"  Tier {k:<10}: {v.get('accuracy_pct')}%")
                        elif isinstance(vals, dict):
                            print(f"  Horizon {tier:<6}: Acc: {vals.get('overall_accuracy_pct')}% | Prec: {vals.get('active_precision_pct')}% | Active Signals: {vals.get('active_signals_count')}")
                if "closed_trades" in data and isinstance(data["closed_trades"], list) and len(data["closed_trades"]) > 0:
                    print("\nClosed Trades Details:")
                    for idx, tr in enumerate(data["closed_trades"]):
                        print(f"  Trade #{idx+1}: {tr.get('side')} @ {tr.get('entry_price')} -> {tr.get('exit_price')} | Net: {tr.get('net_return_bps'):+6.2f} bps | MFE: {tr.get('peak_mfe_bps', tr.get('peak_favorable_excursion_bps'))} bps | MAE: {tr.get('max_mae_bps', tr.get('max_adverse_excursion_bps'))} bps | Dur: {tr.get('holding_duration_sec')}s | Exit Reason: {tr.get('exit_reason')}")
                    break
        except Exception as e:
            pass
            
    # 3. If closed trades not in metrics, parse ledger
    if ledger_files:
        try:
            with open(ledger_files[0], "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip().startswith("{")]
            trades_logged = []
            prev_cnt = 0
            for idx, b in enumerate(lines):
                cnt = b.get("closed_trades_count", 0)
                if cnt > prev_cnt:
                    pos = lines[idx-1].get("active_position", {})
                    trades_logged.append({
                        "bar": b.get("bar_idx"),
                        "cum_pnl": b.get("cumulative_realized_pnl_bps"),
                        "pos": pos
                    })
                    prev_cnt = cnt
            if trades_logged and not ("closed_trades" in data if 'data' in locals() else False):
                print(f"\nLedger Trades ({len(trades_logged)} closed):")
                prev_p = 0.0
                for idx, t in enumerate(trades_logged):
                    diff = t["cum_pnl"] - prev_p
                    prev_p = t["cum_pnl"]
                    print(f"  Trade #{idx+1} at bar {t['bar']}: Net: {diff:+6.2f} bps | Cum: {t['cum_pnl']:+6.2f} bps | Pos: {t['pos']}")
        except Exception as e:
            pass

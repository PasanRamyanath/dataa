import os
import glob
import json
import re

base_dir = "D:/PrivateProjects/BTC Futures/scratch"
archive_dirs = sorted([d for d in os.listdir(base_dir) if (d.startswith("archive_cycle") or d.startswith("archive_v3")) and os.path.isdir(os.path.join(base_dir, d))])

print(f"Found {len(archive_dirs)} archives to audit.\n")

results = []

for ad in archive_dirs:
    full_path = os.path.join(base_dir, ad)
    files = os.listdir(full_path)
    
    # Look for ledgers
    ledger_files = [f for f in files if f.endswith(".jsonl") and ("ledger" in f or "session" in f or "log" in f)]
    summary_files = [f for f in files if f.endswith(".json") and ("final" in f or "results" in f or "summary" in f or "heartbeat" in f or "leaderboard" in f)]
    
    trades = []
    win_count = 0
    loss_count = 0
    total_net_pnl = 0.0
    accuracy_data = {}
    
    # 1. Try parsing summary files first
    parsed_summary = False
    for sf in summary_files:
        try:
            with open(os.path.join(full_path, sf), "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Check if leaderboard
                    if "leaderboard" in data:
                        top = data["leaderboard"][0]
                        trades_cnt = top.get("trades", 0)
                        wr = top.get("win_rate_pct", 0.0)
                        pnl = top.get("net_pnl_bps", 0.0)
                        results.append({
                            "archive": ad,
                            "source": sf,
                            "trades": trades_cnt,
                            "wins": top.get("wins", 0),
                            "losses": top.get("losses", 0),
                            "win_rate": wr,
                            "net_pnl": pnl,
                            "avg_trade_pnl": (pnl / trades_cnt) if trades_cnt > 0 else 0.0,
                            "strategy": top.get("name", "top"),
                            "profile": top.get("strategy_profile", "")
                        })
                        parsed_summary = True
                        break
                    elif "trades_count" in data or "closed_trades" in data or "net_pnl_bps" in data:
                        cnt = data.get("trades_count", data.get("closed_trades", 0))
                        pnl = data.get("net_pnl_bps", data.get("cumulative_realized_pnl_bps", 0.0))
                        wr = data.get("win_rate_pct", data.get("win_rate", 0.0))
                        wins = data.get("wins_count", data.get("wins", 0))
                        losses = data.get("losses_count", data.get("losses", 0))
                        results.append({
                            "archive": ad,
                            "source": sf,
                            "trades": cnt,
                            "wins": wins,
                            "losses": losses,
                            "win_rate": wr,
                            "net_pnl": pnl,
                            "avg_trade_pnl": (pnl / cnt) if cnt > 0 else 0.0,
                            "strategy": "summary_file",
                            "profile": sf
                        })
                        parsed_summary = True
                        break
        except Exception as e:
            pass
            
    # 2. If no summary file or to verify ledger, check ledger files
    if not parsed_summary and ledger_files:
        for lf in ledger_files:
            try:
                with open(os.path.join(full_path, lf), "r", encoding="utf-8", errors="ignore") as f:
                    lines = [json.loads(line) for line in f if line.strip().startswith("{")]
                
                prev_cnt = 0
                trade_pnls = []
                for b in lines:
                    cnt = b.get("closed_trades_count", 0)
                    if cnt > prev_cnt:
                        cum_pnl = b.get("cumulative_realized_pnl_bps", 0.0)
                        trade_pnls.append(cum_pnl)
                        prev_cnt = cnt
                
                if trade_pnls:
                    realized_diffs = []
                    prev = 0.0
                    for p in trade_pnls:
                        diff = p - prev
                        realized_diffs.append(diff)
                        prev = p
                    
                    wins = len([d for d in realized_diffs if d > 0])
                    losses = len([d for d in realized_diffs if d < 0])
                    cnt = len(realized_diffs)
                    wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
                    pnl = trade_pnls[-1]
                    
                    results.append({
                        "archive": ad,
                        "source": lf,
                        "trades": cnt,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": wr,
                        "net_pnl": pnl,
                        "avg_trade_pnl": (pnl / cnt) if cnt > 0 else 0.0,
                        "strategy": "ledger_parsed",
                        "profile": lf
                    })
                    parsed_summary = True
                    break
            except Exception as e:
                pass

print(f"Successfully audited {len(results)} archives.")

# Filter and Sort by various peak criteria
print("\n" + "="*90)
print("--- ALL 100% WIN RATE CYCLES (Sorted by Trade Count Descending, then Net PnL Descending) ---")
print("="*90)

wr100 = [r for r in results if r["win_rate"] == 100.0 and r["trades"] > 0]
wr100.sort(key=lambda x: (x["trades"], x["net_pnl"]), reverse=True)

for r in wr100:
    print(f"{r['archive']:<20} | Trades: {r['trades']} | WR: {r['win_rate']:.1f}% | Net PnL: {r['net_pnl']:+6.2f} bps | Avg/Tr: {r['avg_trade_pnl']:+6.2f} bps | Strategy: {r['strategy']} ({r['source']})")

print("\n" + "="*90)
print("--- TOP 10 HIGHEST PROFIT PER TRADE CYCLES (Min 2 trades) ---")
print("="*90)
top_avg = sorted([r for r in results if r["trades"] >= 2], key=lambda x: x["avg_trade_pnl"], reverse=True)[:10]
for r in top_avg:
    print(f"{r['archive']:<20} | Trades: {r['trades']} | WR: {r['win_rate']:.1f}% | Net PnL: {r['net_pnl']:+6.2f} bps | Avg/Tr: {r['avg_trade_pnl']:+6.2f} bps | Strategy: {r['strategy']}")

print("\n" + "="*90)
print("--- TOP 10 HIGHEST NET PNL CYCLES ---")
print("="*90)
top_net = sorted(results, key=lambda x: x["net_pnl"], reverse=True)[:10]
for r in top_net:
    print(f"{r['archive']:<20} | Trades: {r['trades']} | WR: {r['win_rate']:.1f}% | Net PnL: {r['net_pnl']:+6.2f} bps | Avg/Tr: {r['avg_trade_pnl']:+6.2f} bps | Strategy: {r['strategy']}")

print("\n" + "="*90)
print("--- TOP 10 HIGHEST TRADE COUNT CYCLES ---")
print("="*90)
top_trades = sorted(results, key=lambda x: (x["trades"], x["win_rate"], x["net_pnl"]), reverse=True)[:10]
for r in top_trades:
    print(f"{r['archive']:<20} | Trades: {r['trades']} | WR: {r['win_rate']:.1f}% | Net PnL: {r['net_pnl']:+6.2f} bps | Avg/Tr: {r['avg_trade_pnl']:+6.2f} bps | Strategy: {r['strategy']}")

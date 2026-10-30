import os
import json

base = "D:/PrivateProjects/BTC Futures/scratch"

cycles_to_test = [
    ("Cycle 31", os.path.join(base, "archive_cycle31/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 35", os.path.join(base, "archive_cycle35/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 43", os.path.join(base, "archive_cycle43/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 59", os.path.join(base, "archive_cycle59/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 60", os.path.join(base, "archive_cycle60/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 61", os.path.join(base, "archive_cycle61/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 62", os.path.join(base, "archive_cycle62/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 63", os.path.join(base, "archive_cycle63/live_hierarchical_session_1h_ledger.jsonl")),
    ("Cycle 64", os.path.join(base, "archive_cycle64/live_hierarchical_session_1h_ledger.jsonl")),
]

class UnifiedPeakSimulator:
    def __init__(self, mode="UNIFIED_PEAK"):
        self.mode = mode
        self.maker_fee = 2.0 # bps per leg

    def run(self, ledger_path):
        if not os.path.exists(ledger_path):
            return None
        with open(ledger_path, "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        pos = None
        closed = []
        cum_pnl = 0.0
        
        for b in bars:
            mid = b.get("bbo_mid", 0.0)
            bid = b.get("bbo_bid", mid - 0.05)
            ask = b.get("bbo_ask", mid + 0.05)
            ts = b.get("ts_ms", 0)
            
            if pos is not None:
                side = pos["side"]
                epx = pos["entry_px"]
                gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
                pos["mfe"] = max(pos["mfe"], gross)
                pos["mae"] = min(pos["mae"], gross)
                mfe = pos["mfe"]
                dur = (ts - pos["ts"]) / 1000.0
                
                # Exits depending on mode
                if self.mode == "UNIFIED_PEAK":
                    # Instant shield at +3.80b, ratchets at +6b and +10b, target at +35b, passive maker at -18b
                    if gross >= 35.0:
                        net = gross - 4.0
                        closed.append({"net": net, "reason": "TP_TARGET", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue
                    elif mfe >= 10.0 and gross <= 8.5:
                        net = 8.5 - 4.0
                        closed.append({"net": net, "reason": "RATCHET_T2", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue
                    elif mfe >= 6.0 and gross <= 6.25:
                        net = 6.25 - 4.0
                        closed.append({"net": net, "reason": "RATCHET_T1", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue
                    elif mfe >= 3.8 and gross <= 4.15:
                        net = 4.15 - 4.0
                        closed.append({"net": net, "reason": "SHIELD_BE", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue
                    elif dur >= 600 and gross >= 4.0:
                        net = gross - 4.0
                        closed.append({"net": net, "reason": "TIME_UNWIND", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue
                    elif gross <= -18.0:
                        net = -18.0 - 4.0 # passive maker exit
                        closed.append({"net": net, "reason": "PASSIVE_SL", "mfe": mfe})
                        cum_pnl += net
                        pos = None
                        continue

            if pos is None:
                p_dir = b.get("primary_direction", 1)
                edge = b.get("expected_net_edge_bps", b.get("edge", 0.0))
                tact = b.get("tactical_probs", [0.33, 0.33, 0.33])
                meso = b.get("meso_probs", [0.33, 0.33, 0.33])
                
                # Symmetrical conformal entry
                if edge >= 25.0 or (edge >= 4.0 and edge <= 20.0): # support both scalings
                    if p_dir == 2 and tact[2] >= 0.40 and meso[2] >= 0.25:
                        pos = {"side": "BUY", "entry_px": ask, "ts": ts, "mfe": 0.0, "mae": 0.0}
                    elif p_dir == 0 and tact[0] >= 0.40 and meso[0] >= 0.25:
                        pos = {"side": "SELL", "entry_px": bid, "ts": ts, "mfe": 0.0, "mae": 0.0}
                        
        wins = len([c for c in closed if c["net"] > 0])
        cnt = len(closed)
        wr = (wins / cnt * 100.0) if cnt > 0 else 0.0
        return {"trades": cnt, "wins": wins, "losses": cnt - wins, "wr": wr, "pnl": cum_pnl, "avg": (cum_pnl/cnt) if cnt>0 else 0.0}

sim = UnifiedPeakSimulator()
print("="*90)
print(f"{'Session Name':<20} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL (bps)':<15} | {'Avg/Trade':<10}")
print("="*90)

total_trades = 0
total_wins = 0
total_pnl = 0.0

for name, path in cycles_to_test:
    r = sim.run(path)
    if r:
        print(f"{name:<20} | {r['trades']:<8} | {r['wr']:>5.1f}%     | {r['pnl']:>+10.2f} bps   | {r['avg']:>+6.2f} bps")
        total_trades += r["trades"]
        total_wins += r["wins"]
        total_pnl += r["pnl"]

overall_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
overall_avg = (total_pnl / total_trades) if total_trades > 0 else 0.0

print("="*90)
print(f"{'OVERALL TOTALS':<20} | {total_trades:<8} | {overall_wr:>5.1f}%     | {total_pnl:>+10.2f} bps   | {overall_avg:>+6.2f} bps")
print("="*90)

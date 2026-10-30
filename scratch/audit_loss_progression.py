import os
import glob
import json
import collections
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = "D:/PrivateProjects/BTC Futures/scratch"

def audit_loss_progression():
    target_cycles = [
        ("archive_cycle17", 15, "SELL"),
        ("archive_cycle19", 26, "SELL"),
        ("archive_cycle38", 53, "BUY"),
        ("archive_cycle40", 18, "BUY"),
        ("archive_cycle43", 166, "SELL"),
        ("archive_cycle52", 48, "SELL"),
        ("archive_cycle57", 15, "SELL"),
        ("archive_cycle58", 58, "BUY"),
        ("archive_cycle60", 238, "SELL")
    ]
    
    print("="*105)
    print("TIME-SERIES ADVERSE PROGRESSION OF THE 9 TOXIC TRADES (EVERY 10s BAR)")
    print("="*105)
    print(f"{'Cycle':<16} | {'Side':<4} | {'1m (B6)':<8} | {'2m (B12)':<9} | {'3m (B18)':<9} | {'5m (B30)':<9} | {'10m (B60)':<10} | {'Max MFE':<8} | {'Final Gross':<11}")
    print("-" * 105)
    
    for cyc, ebar, side in target_cycles:
        path = os.path.join(base_dir, cyc)
        ledgers = glob.glob(os.path.join(path, "*ledger*.jsonl"))
        if not ledgers: continue
        with open(ledgers[0], "r", encoding="utf-8") as f:
            bars = [json.loads(l) for l in f if l.strip().startswith("{")]
            
        b_entry = bars[ebar]
        mid_entry = b_entry.get("bbo_mid", b_entry.get("mid"))
        epx = b_entry.get("bbo_ask") if side == "BUY" else b_entry.get("bbo_bid")
        
        trajectory = []
        for i in range(ebar, len(bars)):
            b = bars[i]
            mid = b.get("bbo_mid", b.get("mid"))
            bid = b.get("bbo_bid", mid - 0.05)
            ask = b.get("bbo_ask", mid + 0.05)
            gross = (bid - epx)/epx*10000.0 if side == "BUY" else (epx - ask)/epx*10000.0
            trajectory.append(gross)
            
        g_1m = trajectory[6] if len(trajectory) > 6 else trajectory[-1]
        g_2m = trajectory[12] if len(trajectory) > 12 else trajectory[-1]
        g_3m = trajectory[18] if len(trajectory) > 18 else trajectory[-1]
        g_5m = trajectory[30] if len(trajectory) > 30 else trajectory[-1]
        g_10m = trajectory[60] if len(trajectory) > 60 else trajectory[-1]
        max_mfe = max(trajectory)
        final_g = trajectory[-1]
        
        print(f"{cyc:<16} | {side:<4} | {g_1m:>+6.2f}b  | {g_2m:>+6.2f}b   | {g_3m:>+6.2f}b   | {g_5m:>+6.2f}b   | {g_10m:>+7.2f}b   | {max_mfe:>+6.2f}b  | {final_g:>+8.2f}b")
        
if __name__ == "__main__":
    audit_loss_progression()

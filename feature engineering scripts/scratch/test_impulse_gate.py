import json
import numpy as np

LOG_PATH = "feature engineering scripts/scratch/monitor_30m_log.jsonl"

def load_bars():
    bars = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))
    return bars

bars = load_bars()

def test_impulse_gate(bars, min_chop_bars_before_entry=10, max_hold=25, tp_bps=12.0, sl_bps=12.0):
    trades = []
    active = None
    chop_counter = 0
    last_exit_idx = -999
    
    maker_fee = 1.8 # bps with BNB discount
    taker_fee = 4.5 # bps with BNB discount
    open_fee = maker_fee
    
    for i, b in enumerate(bars):
        idx = b["bar_idx"]
        p = b["binance_price"]
        d = b["direction"]
        auth = b["authorized"]
        
        if not auth:
            chop_counter += 1
            
        # Manage active trade
        if active is not None:
            hold = idx - active["entry_bar"]
            d_mult = 1.0 if active["direction"] == 2 else -1.0
            un_bps = d_mult * ((p - active["entry_price"]) / active["entry_price"]) * 10000.0
            
            # SL
            if un_bps <= -sl_bps:
                close_fee = taker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "reason": "STOP_LOSS", "entry_price": active["entry_price"], "exit_price": p,
                    "gross": un_bps, "fee": open_fee + close_fee, "net": net, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                chop_counter = 0
                continue
                
            # TP
            if un_bps >= tp_bps:
                close_fee = maker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "reason": "TAKE_PROFIT", "entry_price": active["entry_price"], "exit_price": p,
                    "gross": un_bps, "fee": open_fee + close_fee, "net": net, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                chop_counter = 0
                continue
                
            # Max hold
            if hold >= max_hold:
                close_fee = maker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "reason": "MAX_HOLD_EXPIRED", "entry_price": active["entry_price"], "exit_price": p,
                    "gross": un_bps, "fee": open_fee + close_fee, "net": net, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                chop_counter = 0
                continue
                
        # Entry: ONLY enter if flat AND authorized AND we had at least `min_chop_bars_before_entry` bars of consolidation/patience!
        if active is None and auth:
            if chop_counter >= min_chop_bars_before_entry:
                active = {
                    "entry_bar": idx,
                    "direction": d,
                    "entry_price": p,
                    "chop_before": chop_counter
                }
            # Reset chop counter once authorized
            chop_counter = 0

    return trades

print("=" * 85)
print("TESTING FIRST-IMPULSE BREAKOUT GATE (Avoiding Late-Swing Chasing)")
print("=" * 85)
for chop_req in [5, 8, 10, 12, 15, 20]:
    t = test_impulse_gate(bars, min_chop_bars_before_entry=chop_req, max_hold=25, tp_bps=12.0, sl_bps=12.0)
    n = len(t)
    n_win = sum(1 for x in t if x["is_win"])
    win_r = (n_win / n * 100) if n > 0 else 0
    gross = sum(x["gross"] for x in t)
    net = sum(x["net"] for x in t)
    fee = sum(x["fee"] for x in t)
    pf = sum(x["net"] for x in t if x["net"] > 0) / abs(min(-0.001, sum(x["net"] for x in t if x["net"] < 0)))
    print(f"Chop Required: {chop_req:2d} bars ({chop_req*10:3d}s) | Trades: {n:2d} | WinRate: {win_r:5.1f}% | Gross: {gross:+6.2f} bps | Fee: {fee:4.1f} bps | Net: {net:+6.2f} bps | PF: {pf:5.2f}")
    for tr in t:
        print(f"   Trade #{tr['id']} Entry: Bar {tr['entry']} (${tr['entry_price']:.1f}) -> Exit: Bar {tr['exit']} (${tr['exit_price']:.1f}) | [{tr['reason']:<16s}] Gross: {tr['gross']:+6.2f} bps | Net: {tr['net']:+6.2f} bps {'[WIN]' if tr['is_win'] else '[LOSS]'}")

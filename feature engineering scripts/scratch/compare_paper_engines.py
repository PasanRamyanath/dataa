import json

LOG_PATH = "feature engineering scripts/scratch/monitor_30m_log.jsonl"

def load_bars():
    bars = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))
    return bars

def run_simulation(bars, name, tp_pct, sl_pct, max_hold, trail_pct, cooldown, veto_min_roe, execution_style):
    trades = []
    active = None
    last_exit_idx = -999
    
    tp_bps = tp_pct * 100.0   # e.g. 0.08% = 8 bps, 0.14% = 14 bps
    sl_bps = sl_pct * 100.0   # e.g. 0.05% = 5 bps, 0.12% = 12 bps
    trail_bps = trail_pct * 100.0 if trail_pct else None
    
    # VIP 0 with BNB discount fees
    maker_fee = 1.8 # bps
    taker_fee = 4.5 # bps
    
    open_fee = maker_fee if execution_style in ("LIMIT_MAKER", "SMART_HYBRID") else taker_fee
    
    for i, b in enumerate(bars):
        idx = b["bar_idx"]
        p = b["binance_price"]
        d = b["direction"]
        act = b["act"]
        auth = b["authorized"]
        
        if active is not None:
            hold = idx - active["entry_bar"]
            d_mult = 1.0 if active["direction"] == 2 else -1.0
            un_bps = d_mult * ((p - active["entry_price"]) / active["entry_price"]) * 10000.0
            roe_pct = un_bps * 1.0  # at 100x leverage, 1 bps = 1% ROE
            
            if un_bps > active["peak_bps"]:
                active["peak_bps"] = un_bps
                
            # Trailing stop
            if trail_bps is not None and active["peak_bps"] >= 6.0:
                if un_bps <= (active["peak_bps"] - trail_bps):
                    close_fee = maker_fee if execution_style in ("LIMIT_MAKER", "SMART_HYBRID") else taker_fee
                    net = un_bps - (open_fee + close_fee)
                    trades.append({
                        "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                        "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "TRAILING_STOP",
                        "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                    })
                    active = None
                    last_exit_idx = idx
                    continue
                    
            # Stop loss
            if un_bps <= -sl_bps:
                close_fee = taker_fee # Stop loss executes as taker
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "STOP_LOSS",
                    "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                continue
                
            # Take profit
            if un_bps >= tp_bps:
                close_fee = maker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "TAKE_PROFIT",
                    "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                continue
                
            # Veto invalidation exit (if configured)
            if veto_min_roe is not None and act < 0.45 and roe_pct >= veto_min_roe:
                close_fee = maker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "VETO_PROFIT_EXIT",
                    "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                continue
                
            # Opposing signal reversal
            if auth:
                opposing = (active["direction"] == 2 and d == 0) or (active["direction"] == 0 and d == 2)
                if opposing:
                    close_fee = maker_fee
                    net = un_bps - (open_fee + close_fee)
                    trades.append({
                        "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                        "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "SIGNAL_FLIP",
                        "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                    })
                    active = None
                    last_exit_idx = idx
                    
            # Max hold time expiry
            if hold >= max_hold:
                close_fee = maker_fee if execution_style in ("LIMIT_MAKER", "SMART_HYBRID") else taker_fee
                net = un_bps - (open_fee + close_fee)
                trades.append({
                    "id": len(trades) + 1, "entry": active["entry_bar"], "exit": idx, "hold": hold,
                    "dir": "LONG" if active["direction"] == 2 else "SHORT", "reason": "MAX_HOLD_EXPIRED",
                    "gross": un_bps, "net": net, "fee": open_fee + close_fee, "is_win": net > 0
                })
                active = None
                last_exit_idx = idx
                continue
                
        # Entry logic
        if active is None and auth and (idx - last_exit_idx) >= cooldown:
            active = {
                "entry_bar": idx,
                "direction": d,
                "entry_price": p,
                "peak_bps": 0.0
            }
            
    n = len(trades)
    n_win = sum(1 for t in trades if t["is_win"])
    win_rate = (n_win / n * 100) if n > 0 else 0
    gross_tot = sum(t["gross"] for t in trades)
    net_tot = sum(t["net"] for t in trades)
    fee_tot = sum(t["fee"] for t in trades)
    
    print(f"\n{'='*80}")
    print(f"ENGINE CONFIGURATION: {name}")
    print(f"{'='*80}")
    print(f"  Trades: {n:2d} | Win Rate: {win_rate:5.1f}% ({n_win}/{n}) | Gross PnL: {gross_tot:+7.2f} bps | Fee Friction: {fee_tot:6.2f} bps | NET PnL: {net_tot:+7.2f} bps (100x Margin: {net_tot*1.0:+.1f}%)")
    print(f"  Detailed Trades:")
    for t in trades:
        print(f"    Trade #{t['id']:02d} ({t['dir']}) Hold: {t['hold']:2d} bars | [{t['reason']:<18s}] Gross: {t['gross']:+6.2f} bps | Fee: {t['fee']:4.1f} bps | Net: {t['net']:+6.2f} bps {'[PROFIT]' if t['is_win'] else '[LOSS]'}")

if __name__ == "__main__":
    bars = load_bars()
    
    # 1. OLD DEFAULT ENGINE
    run_simulation(bars, "OLD DEFAULT ENGINE (tp=0.08%, sl=0.05%, maxHold=45, veto=0.8% ROE, TAKER)",
                   tp_pct=0.08, sl_pct=0.05, max_hold=45, trail_pct=0.08, cooldown=0, veto_min_roe=0.8,
                   execution_style="MARKET_TAKER")
                   
    # 2. PROPOSED ENHANCED ENGINE
    run_simulation(bars, "PROPOSED ENHANCED ENGINE (tp=0.14%, sl=0.12%, maxHold=30, cooldown=2, SMART_HYBRID)",
                   tp_pct=0.14, sl_pct=0.12, max_hold=30, trail_pct=0.05, cooldown=2, veto_min_roe=6.0,
                   execution_style="SMART_HYBRID")

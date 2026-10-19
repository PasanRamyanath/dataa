import json

LOG_PATH = "feature engineering scripts/scratch/monitor_30m_log.jsonl"

def load_bars():
    bars = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line.strip()))
    return bars

def test_fine_grid(bars):
    configs = []
    for tp in [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0]:
        for sl in [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]:
            for max_hold in [15, 20, 25, 30, 35, 40, 45]:
                for trail_callback in [None, 3.0, 4.0, 5.0]: # Trailing stop from peak once peak >= 6.0 bps
                    for cooldown in [1, 2, 3]:
                        trades = []
                        active = None
                        last_exit = -999
                        
                        for i, b in enumerate(bars):
                            idx = b["bar_idx"]
                            p = b["binance_price"]
                            d = b["direction"]
                            auth = b["authorized"]
                            
                            if active is not None:
                                hold = idx - active["entry_bar"]
                                d_mult = 1.0 if active["direction"] == 2 else -1.0
                                un_bps = d_mult * ((p - active["entry_price"]) / active["entry_price"]) * 10000.0
                                
                                if un_bps > active["peak_bps"]:
                                    active["peak_bps"] = un_bps
                                    
                                # Trailing stop check
                                if trail_callback is not None and active["peak_bps"] >= 6.0:
                                    if un_bps <= (active["peak_bps"] - trail_callback):
                                        net = un_bps - 3.6 # Maker open + close
                                        trades.append({
                                            "entry_bar": active["entry_bar"], "exit_bar": idx, "hold": hold,
                                            "dir": active["direction"], "reason": "TRAILING_STOP",
                                            "gross": un_bps, "net": net, "is_win": net > 0
                                        })
                                        active = None
                                        last_exit = idx
                                        continue
                                        
                                # SL check
                                if un_bps <= -sl:
                                    net = un_bps - (1.8 + 4.5) # Maker open + Taker SL
                                    trades.append({
                                        "entry_bar": active["entry_bar"], "exit_bar": idx, "hold": hold,
                                        "dir": active["direction"], "reason": "STOP_LOSS",
                                        "gross": un_bps, "net": net, "is_win": net > 0
                                    })
                                    active = None
                                    last_exit = idx
                                    continue
                                    
                                # TP check
                                if un_bps >= tp:
                                    net = un_bps - 3.6 # Maker open + Maker TP
                                    trades.append({
                                        "entry_bar": active["entry_bar"], "exit_bar": idx, "hold": hold,
                                        "dir": active["direction"], "reason": "TAKE_PROFIT",
                                        "gross": un_bps, "net": net, "is_win": net > 0
                                    })
                                    active = None
                                    last_exit = idx
                                    continue
                                    
                                # Max hold expiry
                                if hold >= max_hold:
                                    net = un_bps - 3.6 # Maker open + close
                                    trades.append({
                                        "entry_bar": active["entry_bar"], "exit_bar": idx, "hold": hold,
                                        "dir": active["direction"], "reason": "MAX_HOLD_EXPIRED",
                                        "gross": un_bps, "net": net, "is_win": net > 0
                                    })
                                    active = None
                                    last_exit = idx
                                    continue
                                    
                            if active is None and auth and (idx - last_exit) >= cooldown:
                                active = {
                                    "entry_bar": idx,
                                    "direction": d,
                                    "entry_price": p,
                                    "peak_bps": 0.0
                                }
                                
                        if trades:
                            n = len(trades)
                            n_win = sum(1 for x in trades if x["is_win"])
                            net_tot = sum(x["net"] for x in trades)
                            gross_tot = sum(x["gross"] for x in trades)
                            wins_sum = sum(x["net"] for x in trades if x["net"] > 0)
                            loss_sum = abs(min(-0.001, sum(x["net"] for x in trades if x["net"] < 0)))
                            pf = wins_sum / loss_sum
                            configs.append({
                                "tp": tp, "sl": sl, "hold": max_hold, "trail": trail_callback, "cd": cooldown,
                                "n": n, "win_rate": (n_win / n) * 100, "gross": gross_tot, "net": net_tot, "pf": pf,
                                "trades": trades
                            })
                            
    configs.sort(key=lambda x: x["net"], reverse=True)
    return configs

if __name__ == "__main__":
    bars = load_bars()
    res = test_fine_grid(bars)
    print(f"Total simulated parameter sets: {len(res)}")
    print(f"{'Config':<42} | {'Trades':<6} | {'WinRate':<7} | {'Gross PnL':<10} | {'Net PnL':<10} | {'PF':<6}")
    print("-" * 90)
    for r in res[:20]:
        cfg_str = f"TP:{r['tp']} SL:{r['sl']} H:{r['hold']} Tr:{r['trail']} C:{r['cd']}"
        print(f"{cfg_str:<42} | {r['n']:<6} | {r['win_rate']:5.1f}%  | {r['gross']:+7.2f} bps | {r['net']:+7.2f} bps | {r['pf']:5.2f}")
        
    print("\n" + "=" * 90)
    best = res[0]
    print(f"BEST OVERALL: TP:{best['tp']} SL:{best['sl']} H:{best['hold']} Tr:{best['trail']} C:{best['cd']}")
    print(f"Net PnL: {best['net']:+.2f} bps | Win Rate: {best['win_rate']:.1f}% | Profit Factor: {best['pf']:.2f}")
    for i, t in enumerate(best["trades"]):
        print(f"  #{i+1:02d} ({'LONG' if t['dir']==2 else 'SHORT'}) Hold: {t['hold']} bars | "
              f"[{t['reason']:<18s}] Gross: {t['gross']:+6.2f} bps | Net: {t['net']:+6.2f} bps")

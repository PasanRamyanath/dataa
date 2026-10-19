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

def test_parameters(bars, 
                    tp_bps=8.0, 
                    sl_bps=15.0, 
                    trail_activate_bps=4.0, 
                    trail_lock_bps=2.0,
                    max_hold_bars=18, 
                    cooldown_bars=3,
                    exit_on_veto=False,
                    fee_open_bps=1.8,
                    fee_close_maker=1.8,
                    fee_close_taker=4.5):
    
    trades = []
    active_pos = None
    last_exit_bar = -999
    
    for i, bar in enumerate(bars):
        bar_idx = bar["bar_idx"]
        price = bar["binance_price"]
        direction = bar["direction"] # 0 = Short, 2 = Long
        act = bar["act"]
        authorized = bar["authorized"]
        
        # 1. Manage active position if any
        if active_pos is not None:
            hold_bars = bar_idx - active_pos["entry_bar"]
            dir_mult = 1.0 if active_pos["direction"] == 2 else -1.0
            
            # Unrealized return in basis points
            unrealized_bps = dir_mult * ((price - active_pos["entry_price"]) / active_pos["entry_price"]) * 10000.0
            
            if unrealized_bps > active_pos["peak_bps"]:
                active_pos["peak_bps"] = unrealized_bps
            
            # Dynamic Trailing Profit Lock:
            if trail_activate_bps is not None and active_pos["peak_bps"] >= trail_activate_bps:
                if unrealized_bps <= trail_lock_bps:
                    net_bps = unrealized_bps - (fee_open_bps + fee_close_maker)
                    trades.append({
                        "trade_id": len(trades) + 1,
                        "entry_bar": active_pos["entry_bar"],
                        "exit_bar": bar_idx,
                        "hold_bars": hold_bars,
                        "direction": active_pos["direction"],
                        "dir_name": "LONG" if active_pos["direction"] == 2 else "SHORT",
                        "entry_price": active_pos["entry_price"],
                        "exit_price": price,
                        "reason": "TRAILING_PROFIT_LOCK",
                        "gross_bps": unrealized_bps,
                        "net_bps": net_bps,
                        "is_win": net_bps > 0
                    })
                    active_pos = None
                    last_exit_bar = bar_idx
                    continue
            
            # Stop Loss
            if unrealized_bps <= -sl_bps:
                net_bps = unrealized_bps - (fee_open_bps + fee_close_taker)
                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_bar": active_pos["entry_bar"],
                    "exit_bar": bar_idx,
                    "hold_bars": hold_bars,
                    "direction": active_pos["direction"],
                    "dir_name": "LONG" if active_pos["direction"] == 2 else "SHORT",
                    "entry_price": active_pos["entry_price"],
                    "exit_price": price,
                    "reason": "STOP_LOSS",
                    "gross_bps": unrealized_bps,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Take Profit
            if unrealized_bps >= tp_bps:
                net_bps = unrealized_bps - (fee_open_bps + fee_close_maker)
                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_bar": active_pos["entry_bar"],
                    "exit_bar": bar_idx,
                    "hold_bars": hold_bars,
                    "direction": active_pos["direction"],
                    "dir_name": "LONG" if active_pos["direction"] == 2 else "SHORT",
                    "entry_price": active_pos["entry_price"],
                    "exit_price": price,
                    "reason": "TAKE_PROFIT",
                    "gross_bps": unrealized_bps,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Optional Veto Exit
            if exit_on_veto and act < 0.45 and unrealized_bps >= 2.0:
                net_bps = unrealized_bps - (fee_open_bps + fee_close_maker)
                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_bar": active_pos["entry_bar"],
                    "exit_bar": bar_idx,
                    "hold_bars": hold_bars,
                    "direction": active_pos["direction"],
                    "dir_name": "LONG" if active_pos["direction"] == 2 else "SHORT",
                    "entry_price": active_pos["entry_price"],
                    "exit_price": price,
                    "reason": "VETO_EXIT",
                    "gross_bps": unrealized_bps,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Time Barrier Expiry
            if hold_bars >= max_hold_bars:
                net_bps = unrealized_bps - (fee_open_bps + fee_close_maker)
                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_bar": active_pos["entry_bar"],
                    "exit_bar": bar_idx,
                    "hold_bars": hold_bars,
                    "direction": active_pos["direction"],
                    "dir_name": "LONG" if active_pos["direction"] == 2 else "SHORT",
                    "entry_price": active_pos["entry_price"],
                    "exit_price": price,
                    "reason": "MAX_HOLD_EXPIRED",
                    "gross_bps": unrealized_bps,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
        # 2. Entry Logic
        if active_pos is None and authorized:
            if (bar_idx - last_exit_bar) >= cooldown_bars:
                active_pos = {
                    "entry_bar": bar_idx,
                    "direction": direction,
                    "entry_price": price,
                    "act": act,
                    "peak_bps": 0.0
                }
                
    return trades

if __name__ == "__main__":
    bars = load_bars()
    
    print("Testing Parameter Grid across 181 real live bars:")
    print(f"{'Config':<42} | {'Trades':<6} | {'WinRate':<7} | {'Gross PnL':<10} | {'Net PnL':<10} | {'PF':<6}")
    print("-" * 90)
    
    results = []
    
    for tp in [5.0, 7.0, 8.0, 10.0, 12.0, 15.0]:
        for sl in [8.0, 10.0, 12.0, 15.0, 20.0]:
            for trail_act, trail_lock in [(None, None), (4.0, 2.0), (5.0, 2.5), (6.0, 3.0)]:
                for max_hold in [12, 18, 24, 30]:
                    for cooldown in [1, 2, 3]:
                        t = test_parameters(bars, tp_bps=tp, sl_bps=sl, 
                                            trail_activate_bps=trail_act, trail_lock_bps=trail_lock,
                                            max_hold_bars=max_hold, cooldown_bars=cooldown, exit_on_veto=False)
                        if t:
                            n = len(t)
                            win_r = (sum(1 for x in t if x["is_win"]) / n) * 100
                            gross = sum(x["gross_bps"] for x in t)
                            net = sum(x["net_bps"] for x in t)
                            wins_sum = sum(x["net_bps"] for x in t if x["net_bps"] > 0)
                            loss_sum = abs(min(-0.001, sum(x["net_bps"] for x in t if x["net_bps"] < 0)))
                            pf = wins_sum / loss_sum
                            results.append({
                                "config": f"TP:{tp} SL:{sl} Tr:{trail_act}/{trail_lock} H:{max_hold} C:{cooldown}",
                                "n": n, "win_r": win_r, "gross": gross, "net": net, "pf": pf, "trades": t
                            })
                            
    # Sort by Net PnL descending
    results.sort(key=lambda x: x["net"], reverse=True)
    
    for r in results[:15]:
        print(f"{r['config']:<42} | {r['n']:<6} | {r['win_r']:5.1f}%  | {r['gross']:+7.2f} bps | {r['net']:+7.2f} bps | {r['pf']:5.2f}")
        
    print("\n" + "=" * 90)
    best = results[0]
    print(f"BEST CONFIGURATION: {best['config']}")
    print(f"Net PnL: {best['net']:+.2f} bps | Win Rate: {best['win_r']:.1f}% | Profit Factor: {best['pf']:.2f}")
    print("Trade details of best configuration:")
    for t in best["trades"]:
        print(f"  Trade #{t['trade_id']:02d} ({t['dir_name']}) Bars: {t['entry_bar']}->{t['exit_bar']} ({t['hold_bars']} bars) | "
              f"[{t['reason']:<20s}] Entry: ${t['entry_price']:.1f} Exit: ${t['exit_price']:.1f} | "
              f"Gross: {t['gross_bps']:+6.2f} bps | Net: {t['net_bps']:+6.2f} bps {'[WIN]' if t['is_win'] else '[LOSS]'}")

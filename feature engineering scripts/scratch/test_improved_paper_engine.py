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

def simulate_engine(bars, 
                    tp_bps=8.0, 
                    sl_bps=15.0, 
                    trail_activate_bps=4.0, 
                    trail_lock_bps=2.0,
                    max_hold_bars=18, 
                    cooldown_bars=3,
                    exit_on_flip=True,
                    exit_on_veto_profit=True,
                    fee_open_bps=1.8,   # Maker limit with BNB discount
                    fee_close_maker=1.8, # Maker limit with BNB discount
                    fee_close_taker=4.5  # Taker with BNB discount
                   ):
    """
    Simulates paper trading over the 181 live bars with configurable risk and execution rules.
    """
    trades = []
    active_pos = None
    last_exit_bar = -999
    
    for i, bar in enumerate(bars):
        bar_idx = bar["bar_idx"]
        price = bar["binance_price"]
        direction = bar["direction"] # 0 = Short, 2 = Long
        act = bar["act"]
        authorized = bar["authorized"] # act >= 0.70 and direction in (0, 2)
        
        # 1. Manage active position if any
        if active_pos is not None:
            hold_bars = bar_idx - active_pos["entry_bar"]
            dir_mult = 1.0 if active_pos["direction"] == 2 else -1.0
            
            # Unrealized return in basis points
            unrealized_bps = dir_mult * ((price - active_pos["entry_price"]) / active_pos["entry_price"]) * 10000.0
            
            # Update peak profit reached during trade
            if unrealized_bps > active_pos["peak_bps"]:
                active_pos["peak_bps"] = unrealized_bps
            
            # Dynamic Trailing Profit Lock:
            # If peak exceeded trail_activate_bps, raise dynamic floor to trail_lock_bps
            dynamic_sl_bps = sl_bps
            if trail_activate_bps is not None and active_pos["peak_bps"] >= trail_activate_bps:
                # Dynamic floor locked
                dynamic_floor = trail_lock_bps
                if unrealized_bps <= dynamic_floor:
                    # Profit lock triggered!
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
                        "fee_bps": fee_open_bps + fee_close_maker,
                        "net_bps": net_bps,
                        "is_win": net_bps > 0
                    })
                    active_pos = None
                    last_exit_bar = bar_idx
                    continue
            
            # Check Stop Loss
            if unrealized_bps <= -dynamic_sl_bps:
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
                    "fee_bps": fee_open_bps + fee_close_taker,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Check Take Profit
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
                    "fee_bps": fee_open_bps + fee_close_maker,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Check Invalidation / Veto with profit lock
            if exit_on_veto_profit and act < 0.45 and unrealized_bps >= 2.0:
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
                    "reason": "VETO_PROFIT_EXIT",
                    "gross_bps": unrealized_bps,
                    "fee_bps": fee_open_bps + fee_close_maker,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
            # Check Opposing Signal Flip
            if exit_on_flip and authorized:
                opposing = (active_pos["direction"] == 2 and direction == 0) or (active_pos["direction"] == 0 and direction == 2)
                if opposing:
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
                        "reason": "SIGNAL_FLIP",
                        "gross_bps": unrealized_bps,
                        "fee_bps": fee_open_bps + fee_close_maker,
                        "net_bps": net_bps,
                        "is_win": net_bps > 0
                    })
                    active_pos = None
                    last_exit_bar = bar_idx
                    # Can immediately re-enter in new direction
                    
            # Check Time Barrier Expiry
            if hold_bars >= max_hold_bars:
                # Time barrier exit
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
                    "fee_bps": fee_open_bps + fee_close_maker,
                    "net_bps": net_bps,
                    "is_win": net_bps > 0
                })
                active_pos = None
                last_exit_bar = bar_idx
                continue
                
        # 2. Entry Logic (when Flat)
        if active_pos is None and authorized:
            # Enforce Cooldown period (in bars) since last exit to prevent stacking on exhausted swings
            if (bar_idx - last_exit_bar) >= cooldown_bars:
                active_pos = {
                    "entry_bar": bar_idx,
                    "direction": direction,
                    "entry_price": price,
                    "act": act,
                    "peak_bps": 0.0
                }
                
    return trades

def print_summary(name, trades):
    if not trades:
        print(f"[{name}] No trades executed.")
        return
    
    n = len(trades)
    n_win = sum(1 for t in trades if t["is_win"])
    win_rate = (n_win / n) * 100
    gross_pnl = sum(t["gross_bps"] for t in trades)
    net_pnl = sum(t["net_bps"] for t in trades)
    avg_gross = gross_pnl / n
    avg_net = net_pnl / n
    profit_factor = sum(t["net_bps"] for t in trades if t["net_bps"] > 0) / abs(min(-0.001, sum(t["net_bps"] for t in trades if t["net_bps"] < 0)))
    
    print(f"\n{'='*75}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*75}")
    print(f"  Total Trades:     {n}")
    print(f"  Win Rate:         {win_rate:.1f}% ({n_win}/{n})")
    print(f"  Total Gross PnL:  {gross_pnl:+.2f} bps")
    print(f"  Total Net PnL:    {net_pnl:+.2f} bps  (100x Margin Return: {net_pnl*1.0:+.1f}%)")
    print(f"  Avg Net per Trade:{avg_net:+.2f} bps")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Trade Breakdown:")
    for t in trades:
        print(f"    Trade #{t['trade_id']:02d} ({t['dir_name']}) Bars: {t['entry_bar']}->{t['exit_bar']} ({t['hold_bars']} bars) | "
              f"[{t['reason']:<20s}] Entry: ${t['entry_price']:.1f} Exit: ${t['exit_price']:.1f} | "
              f"Gross: {t['gross_bps']:+6.2f} bps | Net: {t['net_bps']:+6.2f} bps {'[WIN]' if t['is_win'] else '[LOSS]'}")

if __name__ == "__main__":
    bars = load_bars()
    print(f"Loaded {len(bars)} recorded live bars from Binance.")
    
    # Baseline 1: The Old Naive Setup (5 bps SL, 8 bps TP, Taker fee, no trailing stop)
    t_base = simulate_engine(bars, tp_bps=8.0, sl_bps=5.0, trail_activate_bps=None, max_hold_bars=18, cooldown_bars=0,
                             fee_open_bps=5.0, fee_close_maker=5.0, fee_close_taker=5.0)
    print_summary("BASELINE (Original: 5 bps SL, 8 bps TP, 10 bps Taker Fee, No Cooldown)", t_base)
    
    # Baseline 2: Single Position + Maker Fee (5 bps SL, 8 bps TP)
    t_maker = simulate_engine(bars, tp_bps=8.0, sl_bps=5.0, trail_activate_bps=None, max_hold_bars=18, cooldown_bars=2,
                              fee_open_bps=1.8, fee_close_maker=1.8, fee_close_taker=4.5)
    print_summary("IMPROVEMENT 1: Maker Fee (1.8 bps) + Single Position Cooldown", t_maker)
    
    # Improvement 2: Volatility-Adjusted Stop Loss (15 bps SL) to avoid noise sweeps
    t_sl15 = simulate_engine(bars, tp_bps=8.0, sl_bps=15.0, trail_activate_bps=None, max_hold_bars=18, cooldown_bars=2,
                             fee_open_bps=1.8, fee_close_maker=1.8, fee_close_taker=4.5)
    print_summary("IMPROVEMENT 2: Volatility-Adjusted SL (15 bps SL, 8 bps TP, Maker Fee)", t_sl15)
    
    # Improvement 3: Dynamic Trailing Profit Lock (+4.0 bps peak locks in +2.0 bps profit)
    t_trail = simulate_engine(bars, tp_bps=8.0, sl_bps=15.0, trail_activate_bps=4.0, trail_lock_bps=2.0, max_hold_bars=18, cooldown_bars=2,
                              fee_open_bps=1.8, fee_close_maker=1.8, fee_close_taker=4.5)
    print_summary("IMPROVEMENT 3: Trailing Profit Lock (Peak +4.0 bps locks +2.0 bps, 15 bps SL, 8 bps TP)", t_trail)
    
    # Improvement 4: High-Frequency Scalper (TP: 6.0 bps, Peak +3.5 bps locks +2.0 bps, 15 bps SL, 3-bar Cooldown)
    t_hft = simulate_engine(bars, tp_bps=6.0, sl_bps=15.0, trail_activate_bps=3.5, trail_lock_bps=2.0, max_hold_bars=12, cooldown_bars=3,
                            fee_open_bps=1.8, fee_close_maker=1.8, fee_close_taker=4.5)
    print_summary("IMPROVEMENT 4: Precision Scalper (TP: 6.0 bps, Trail +3.5/+2.0 bps, SL: 15 bps, MaxHold: 12 bars)", t_hft)

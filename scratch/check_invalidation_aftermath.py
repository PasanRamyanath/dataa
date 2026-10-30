import json
from pathlib import Path

for c in [61, 62, 63]:
    ledger_p = Path(f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl")
    results_p = Path(f"scratch/archive_cycle{c}/champion_challenger_final_results.json")
    if not ledger_p.exists() or not results_p.exists():
        continue
    
    bars = [json.loads(line) for line in ledger_p.read_text(encoding="utf-8").strip().split("\n")]
    res_data = json.loads(results_p.read_text(encoding="utf-8"))
    
    # get champion trades
    trades = res_data.get("competitor_trade_ledgers", {}).get("Champion", [])
    
    print("="*80)
    print(f"CYCLE {c} LOSS ANALYSIS: Did price reverse after DUAL_HORIZON_INVALIDATION_TAKER?")
    print("="*80)
    
    bar_map = {b["ts_ms"]: b for b in bars}
    ts_list = [b["ts_ms"] for b in bars]
    prices = [b["bbo_mid"] for b in bars]
    
    for i, t in enumerate(trades):
        if "INVALIDATION" in t.get("exit_reason", "") or t.get("net_pnl_bps", 0) < 0:
            entry_p = t["entry_price"]
            exit_p = t["exit_price"]
            exit_ts = t["exit_ts_ms"]
            side = t["side"]
            net_bps = t.get("net_pnl_bps", t.get("net_return_bps", 0))
            
            # find index of exit_ts in bars
            try:
                exit_idx = min(range(len(ts_list)), key=lambda idx: abs(ts_list[idx] - exit_ts))
            except:
                continue
            
            # check price 30s (3 bars), 60s (6 bars), 120s (12 bars), 300s (30 bars) later
            p_later = {}
            for future_bars, label in [(3, "+30s"), (6, "+60s"), (12, "+120s"), (30, "+300s")]:
                f_idx = min(len(prices) - 1, exit_idx + future_bars)
                p_f = prices[f_idx]
                # PnL if we held to that point
                if side == "SELL":
                    gain_from_entry = ((entry_p - p_f) / entry_p) * 1e4 - 4.0 # maker exit
                else:
                    gain_from_entry = ((p_f - entry_p) / entry_p) * 1e4 - 4.0
                p_later[label] = (p_f, gain_from_entry)
            
            print(f"Trade {i+1} ({side} @ {entry_p:.1f} -> exit @ {exit_p:.1f}, Cut PnL: {net_bps:+.2f} bps):")
            for label, (pf, gain) in p_later.items():
                print(f"    {label}: BTC={pf:.1f} -> PnL would be {gain:+.2f} bps {'[REVERSED TO PROFIT!]' if gain > 0 else '[stayed loss]'}")

import json

def test_rule(cycle_name):
    path = f"scratch/archive_{cycle_name}/live_hierarchical_session_1h_ledger.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
    
    # Check Trade 2 entry bar in cycle
    if cycle_name == "cycle11":
        exit_p = 79990.65
        exit_ts = 1793080170000 # bar 185
        t2_bar = 221
    else: # cycle10
        exit_p = 79676.00
        exit_ts = 1793076180000 # bar 130
        t2_bar = 148
        
    b = bars[t2_bar-1]
    now_ts = b["ts_ms"]
    mid = b["bbo_mid"]
    elapsed_sec = (now_ts - exit_ts) / 1000.0
    
    # For BUY: retracement is (exit_p - mid) / exit_p * 10000
    # For SELL: retracement is (mid - exit_p) / exit_p * 10000
    side = "BUY" if cycle_name == "cycle11" else "SELL"
    if side == "BUY":
        retrace_bps = (exit_p - mid) / exit_p * 10000.0
    else:
        retrace_bps = (mid - exit_p) / exit_p * 10000.0
        
    print(f"=== {cycle_name.upper()} Trade 2 Evaluation ===")
    print(f"  Exit price: {exit_p:.2f}")
    print(f"  T2 price:   {mid:.2f}")
    print(f"  Elapsed:    {elapsed_sec:.1f}s (Threshold: 480s)")
    print(f"  Retrace:    {retrace_bps:.2f} bps (Threshold: >= 15.0 bps)")
    allowed = (elapsed_sec >= 480.0) or (retrace_bps >= 15.0)
    print(f"  --> Trade 2 Allowed? {allowed} (Filtered = {not allowed})")

test_rule("cycle11")
test_rule("cycle10")

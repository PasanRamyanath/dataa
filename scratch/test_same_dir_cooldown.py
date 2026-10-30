import json

def replay_session(cycle_name, cooldown_sec=480):
    path = f"scratch/archive_{cycle_name}/live_hierarchical_session_1h_ledger.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
    
    # Check what trades would be taken
    trades = []
    active = None
    last_exit_ts = 0
    last_exit_dir = 1
    
    # We can inspect original closed trades from metrics
    metrics_p = f"scratch/archive_{cycle_name}/live_session_final_metrics.json"
    try:
        with open(metrics_p, "r", encoding="utf-8") as f:
            m = json.load(f)
            orig_trades = m["closed_trades"]
    except Exception:
        orig_trades = []
        
    print(f"\n=== REPLAYING {cycle_name.upper()} WITH {cooldown_sec}s SAME-DIRECTION COOLDOWN ===")
    print(f"Original trades count: {len(orig_trades)}")
    for i, t in enumerate(orig_trades):
        print(f"  Orig Trade {i+1}: Side={t['side']} Entry={t['entry_price']} Exit={t['exit_price']} Net={t['net_return_bps']:.2f} bps ({t['exit_reason']})")

replay_session("cycle11", 480)
replay_session("cycle10", 480)

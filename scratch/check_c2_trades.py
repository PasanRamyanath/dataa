import json

fpath = "scratch/live_hierarchical_session_1h_ledger.jsonl"
with open(fpath, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d.get("closed_trades_count", 0) > 0 or d.get("active_position") is not None:
            print("Ledger trade bar:", d['bar_idx'], d['bbo_mid'], d['closed_trades_count'])

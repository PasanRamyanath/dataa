import json
from pathlib import Path

for c in ["cycle5", "cycle7", "cycle10", "cycle11"]:
    p = Path(f"scratch/archive_{c}/live_hierarchical_session_1h_ledger.jsonl")
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        bars = [json.loads(line) for line in f]
    
    # Extract trade closed events and entries
    # Check gap from exit to next entry
    closed_events = []
    for b in bars:
        pos = b.get("active_position")
        cnt = b.get("closed_trades_count", 0)
        closed_events.append((b["bar_idx"], b["ts_ms"], pos, cnt))
    
    print(f"=== {c.upper()} ===")
    prev_cnt = 0
    exit_ts = None
    exit_bar = None
    for bar_idx, ts_ms, pos, cnt in closed_events:
        if cnt > prev_cnt:
            exit_ts = ts_ms
            exit_bar = bar_idx
            prev_cnt = cnt
            print(f"  Trade {cnt} EXITED at bar {bar_idx}")
        elif pos is not None and exit_ts is not None:
            # We entered a new trade after an exit
            gap_sec = (ts_ms - exit_ts) / 1000.0
            print(f"  Next trade ENTERED at bar {bar_idx} (gap: {gap_sec:.0f}s / {gap_sec/60:.1f}min from bar {exit_bar})")
            exit_ts = None # reset until next exit

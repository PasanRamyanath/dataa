import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get('trade_id') is not None or d.get('trade') is not None or d.get('action') is not None:
                print("ENTRY:", d)
        except Exception:
            pass

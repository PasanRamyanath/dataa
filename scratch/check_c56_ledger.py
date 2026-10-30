import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in ledger: {len(lines)}")
for line in lines:
    try:
        t = json.loads(line)
        print("Trade:", t.get("trade_id"), t.get("side"), "entry:", t.get("entry_price"), "exit:", t.get("exit_price"), "net_pnl_bps:", t.get("net_pnl_bps"), "reason:", t.get("exit_reason"))
    except Exception as e:
        print(line[:100])

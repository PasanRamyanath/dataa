import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    bars = [json.loads(line) for line in f]

# Find Trade 2 entry and exit
t2_entry = None
t2_exit = None

for b in bars:
    idx = b["bar_idx"]
    pos = b.get("active_position")
    cnt = b.get("closed_trades_count", 0)
    if cnt == 1 and pos and not t2_entry:
        t2_entry = b
    if cnt == 2 and not t2_exit:
        t2_exit = b

print("=== TRADE 2 ENTRY ===")
if t2_entry:
    print(f"Bar {t2_entry['bar_idx']}: Mid={t2_entry['bbo_mid']}, Dir={t2_entry['primary_direction']}, Edge={t2_entry['expected_net_edge_bps']}")
    print("Pos:", t2_entry.get("active_position"))

print("\n=== TRADE 2 EXIT ===")
if t2_exit:
    print(f"Bar {t2_exit['bar_idx']}: Mid={t2_exit['bbo_mid']}, Dir={t2_exit['primary_direction']}")

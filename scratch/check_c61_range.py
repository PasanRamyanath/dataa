import json

with open("scratch/archive_cycle61/live_hierarchical_session_1h_ledger.jsonl") as f:
    mids = [json.loads(line)["bbo_mid"] for line in f]

print(f"Cycle 61 Mid range: min={min(mids):.1f}, max={max(mids):.1f}, span=${max(mids)-min(mids):.1f}")
print(f"Start price: {mids[0]:.1f}, End price: {mids[-1]:.1f}")

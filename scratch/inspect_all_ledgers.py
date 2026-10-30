import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

valid_cycles = []
for d in sorted(os.listdir('scratch')):
    if d.startswith('archive_cycle') and os.path.isdir(os.path.join('scratch', d)):
        ledger = os.path.join('scratch', d, 'live_hierarchical_session_1h_ledger.jsonl')
        if os.path.exists(ledger) and os.path.getsize(ledger) > 0:
            valid_cycles.append(d)

valid_cycles = sorted(valid_cycles, key=lambda x: int(x.replace('archive_cycle', '')))
print(f"Total valid archives: {len(valid_cycles)}")

schemas = {}
for c in valid_cycles:
    p = os.path.join('scratch', c, 'live_hierarchical_session_1h_ledger.jsonl')
    with open(p, 'r', encoding='utf-8') as f:
        first = json.loads(f.readline())
        schema_key = tuple(sorted(first.keys()))
        if schema_key not in schemas:
            schemas[schema_key] = []
        schemas[schema_key].append(c)

print(f"Unique schema counts: {len(schemas)}")
for k, v in schemas.items():
    print(f"Cycles: {len(v)} (e.g. {v[:3]}) -> Keys: {k[:8]}")

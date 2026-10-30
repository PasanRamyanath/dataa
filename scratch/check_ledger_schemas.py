import glob, os, json, re

def get_cycle_num(path):
    m = re.search(r'archive_cycle(\d+)', path)
    return int(m.group(1)) if m else 999

ledgers = sorted(glob.glob('scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl'), key=get_cycle_num)
print(f"Checking {len(ledgers)} ledgers for field consistency...")

field_sets = {}
for l in ledgers:
    num = get_cycle_num(l)
    with open(l, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        if line:
            d = json.loads(line)
            keys = tuple(sorted(d.keys()))
            if keys not in field_sets:
                field_sets[keys] = []
            field_sets[keys].append(num)

print(f"Found {len(field_sets)} unique schemas across 56 ledgers:")
for i, (k, cycles) in enumerate(field_sets.items()):
    print(f"Schema {i+1} (Cycles {min(cycles)}-{max(cycles)}, count={len(cycles)}): {k[:10]}...")

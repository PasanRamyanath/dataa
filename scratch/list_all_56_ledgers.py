import glob, os, json

ledgers = sorted(glob.glob('scratch/archive_cycle*/live_hierarchical_session_1h_ledger.jsonl'))
print(f"Total available 1h ledgers: {len(ledgers)}")

# Sort by cycle number
def get_cycle_num(path):
    import re
    m = re.search(r'archive_cycle(\d+)', path)
    return int(m.group(1)) if m else 999

ledgers = sorted(ledgers, key=get_cycle_num)
for l in ledgers:
    num = get_cycle_num(l)
    bars = sum(1 for _ in open(l, 'r', encoding='utf-8'))
    print(f"Cycle {num:2d}: {bars:3d} bars ({bars*10/60:.1f} min)")

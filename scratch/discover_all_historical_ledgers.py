import os, glob

print("--- ARCHIVED CYCLES FOUND ---")
archives = sorted(glob.glob('scratch/archive_cycle*'))
for a in archives:
    files = os.listdir(a)
    json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
    print(f"{a}: {len(files)} files -> {json_files}")

print("\n--- LEDGERS IN SCRATCH ---")
ledgers = glob.glob('scratch/*ledger*.jsonl')
for l in ledgers:
    print(l, os.path.getsize(l), "bytes")

print("\n--- OTHER RECORDED CYCLE FILES ---")
cycle_files = glob.glob('scratch/*cycle*.json') + glob.glob('scratch/*metrics*.json')
for c in sorted(cycle_files):
    print(c, os.path.getsize(c), "bytes")

import glob, json, os

archives = sorted(glob.glob('scratch/archive_cycle*'))
print(f"Total cycle archives found: {len(archives)}")

sample_archives = [archives[0], archives[len(archives)//2], archives[-1]]
for a in sample_archives:
    ledger = os.path.join(a, 'live_hierarchical_session_1h_ledger.jsonl')
    if os.path.exists(ledger):
        with open(ledger, 'r') as f:
            first_line = json.loads(f.readline())
        print(f"Archive: {a} | Bars: {sum(1 for _ in open(ledger))} | Keys: {list(first_line.keys())[:8]}")
    else:
        print(f"Archive: {a} | Ledger not found")

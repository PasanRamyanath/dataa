import os, shutil

archive_dir = 'scratch/archive_cycle58'
os.makedirs(archive_dir, exist_ok=True)

files_to_copy = [
    'scratch/champion_challenger_leaderboard.json',
    'scratch/live_session_1h_heartbeat.json',
    'scratch/live_hierarchical_session_1h_ledger.jsonl',
    'scratch/live_champion_challenger.log',
    'scratch/champion_challenger_final_results.json',
    'scratch/live_session_final_metrics.json'
]

for src in files_to_copy:
    if os.path.exists(src):
        dst = os.path.join(archive_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"File not found: {src}")

print("Cycle 58 archive complete!")

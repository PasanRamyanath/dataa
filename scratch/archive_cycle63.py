import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

archive_dir = 'scratch/archive_cycle63'
os.makedirs(archive_dir, exist_ok=True)

files_to_copy = [
    'champion_challenger_final_results.json',
    'champion_challenger_leaderboard.json',
    'live_champion_challenger.log',
    'live_hierarchical_session_1h_ledger.jsonl',
    'live_session_1h_heartbeat.json',
    'live_session_final_metrics.json',
]

copied = 0
for f in files_to_copy:
    src = os.path.join('scratch', f)
    dst = os.path.join(archive_dir, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024.0
        print(f"Archived: {f} ({size_kb:.1f} KB)")
        copied += 1
    else:
        print(f"Warning: {src} does not exist!")

print(f"\nSuccessfully archived {copied}/{len(files_to_copy)} artifacts into {archive_dir}/")

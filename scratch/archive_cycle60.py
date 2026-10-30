import os, shutil

src_dir = 'scratch'
dst_dir = os.path.join(src_dir, 'archive_cycle60')
os.makedirs(dst_dir, exist_ok=True)

files_to_copy = [
    'champion_challenger_final_results.json',
    'champion_challenger_leaderboard.json',
    'live_hierarchical_session_1h_ledger.jsonl',
    'live_session_1h_heartbeat.json',
    'live_session_final_metrics.json',
    'live_champion_challenger.log',
]

for fname in files_to_copy:
    src_path = os.path.join(src_dir, fname)
    if os.path.exists(src_path):
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {fname} -> {dst_path} ({os.path.getsize(dst_path)} bytes)")
    else:
        print(f"Warning: {src_path} not found!")

print("Cycle 60 archiving complete!")

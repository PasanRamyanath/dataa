import os, shutil

archive_dir = "scratch/archive_cycle61"
os.makedirs(archive_dir, exist_ok=True)

files_to_archive = [
    "live_hierarchical_session_1h_ledger.jsonl",
    "champion_challenger_leaderboard.json",
    "champion_challenger_final_results.json",
    "live_champion_challenger.log",
    "live_session_1h_heartbeat.json",
    "live_session_final_metrics.json",
]

for f in files_to_archive:
    src = os.path.join("scratch", f)
    dst = os.path.join(archive_dir, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Archived: {src} -> {dst} ({os.path.getsize(dst)} bytes)")
    else:
        print(f"Warning: {src} does not exist!")

print(f"\nCycle 61 archived successfully into {archive_dir}/")

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/archive_cycle57/live_champion_challenger.log", "r", encoding="utf-8") as f:
    for line in f:
        if "TOURNAMENT LEADERBOARD AUDIT" in line or "# 1   |" in line:
            print(line.strip())

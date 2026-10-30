import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/archive_cycle35/live_champion_challenger.log", "r", encoding="utf-8") as f:
    for line in f:
        if "Challenger_5" in line or "Challenger 5" in line:
            if "08:0" in line or "08:1" in line:
                print(line.strip())

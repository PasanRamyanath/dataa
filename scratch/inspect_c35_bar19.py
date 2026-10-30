import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/archive_cycle35/live_champion_challenger.log", "r", encoding="utf-8") as f:
    for line in f:
        if "#0018]" in line or "#0019]" in line or "#0020]" in line or "#0021]" in line or "#0022]" in line or "#0023]" in line:
            print(line.strip())

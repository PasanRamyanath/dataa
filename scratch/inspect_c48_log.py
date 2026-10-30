import re

fpath = "scratch/archive_cycle48/live_champion_challenger.log"
with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if "#0049" in line or "#0050" in line or "#0051" in line:
            print(line.strip())

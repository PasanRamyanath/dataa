import re

with open("scratch/live_champion_challenger.log", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "Challenger_2" in line or "Challenger 2" in line:
            print(line.strip())

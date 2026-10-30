import json

arch = "scratch/archive_cycle35"
with open(f"{arch}/live_champion_challenger.log", 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if "Challenger_5" in line and ("ENTRY" in line or "Trade" in line or "FILL" in line or "ORDER" in line):
            print(line.strip())

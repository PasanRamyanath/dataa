with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for l in lines[-25:]:
    print(l.strip())

with open('scratch/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'Champ Pos:' in line and 'Champ Pos: NONE' not in line:
            print(line.strip())

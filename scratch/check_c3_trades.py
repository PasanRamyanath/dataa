with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'Challenger_3' in line:
            print(line.strip())

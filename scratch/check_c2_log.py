with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'Challenger_2' in line or 'TRADE' in line or 'Fill' in line or 'BBO' in line:
            if 'Challenger_2' in line:
                print(line.strip())

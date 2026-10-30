with open('scratch/archive_cycle60/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '02:08:' in line:
            print(line.strip())

with open('scratch/archive_cycle60/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if any(f'#{i:04d}' in line for i in range(105, 115)):
            print(line.strip())

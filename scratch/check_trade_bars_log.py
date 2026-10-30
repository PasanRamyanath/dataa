with open('scratch/live_champion_challenger.log', 'r', encoding='utf-8') as f:
    for line in f:
        if any(f'#{k:04d}' in line for k in range(105, 125)) or any(f'#{k:04d}' in line for k in range(155, 170)):
            print(line.strip())

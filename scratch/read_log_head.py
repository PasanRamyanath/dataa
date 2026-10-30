with open('scratch/archive_cycle60/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [f.readline() for _ in range(30)]
    for l in lines:
        print(l.strip())

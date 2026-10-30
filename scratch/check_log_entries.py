with open('scratch/archive_cycle60/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if any(term in line for term in ['PLACING_MAKER_ENTRY', 'RESTING_ENTRY_QUEUE', 'FILLED_SWING_HOLDING', 'ENTRY FILLED', 'ENTRY PLACED']):
            print(line.strip())

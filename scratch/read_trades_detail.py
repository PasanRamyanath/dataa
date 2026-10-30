with open('scratch/live_champion_challenger.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if any(term in line for term in ['ENTRY PLACED', 'ENTRY FILLED', 'EXIT', 'closed trade', 'Trade #', 'TRADE CLOSED', 'MODEL_HARVEST', 'INVALIDATION']):
            print(line.strip())

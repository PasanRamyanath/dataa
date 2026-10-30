with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'Challenger_2' in line and ('ENTRY' in line or 'EXIT' in line or 'Filled' in line or 'SL' in line or 'TP' in line or 'Closed' in line or 'P&L' in line or 'pnl' in line):
            print(line.strip())

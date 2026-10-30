import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/archive_cycle35/live_champion_challenger.log', encoding='utf-8') as f:
    for line in f:
        if ('BUY' in line or 'SELL' in line or 'Trade' in line) and ('Challenger_5' in line or 'Challenger 5' in line):
            print(line.strip())

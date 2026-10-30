import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_champion_challenger.log', 'r', encoding='utf-8') as f:
    for line in f:
        if 'Closed' in line or 'FILLED' in line or 'Entry' in line or 'trade' in line.lower() or 'pnl' in line.lower():
            if any(k in line for k in ['Trade', 'Closed', 'Exit', 'FILLED_SWING_HOLDING', 'exit_reason', 'gross_return']):
                print(line.strip())

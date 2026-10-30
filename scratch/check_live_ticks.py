import sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

# Let's inspect the live session log or see if ticks_seen is logged
with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'ticks_seen' in line or 'STALKING' in line or 'sniper_maker' in line:
            print(line.strip())

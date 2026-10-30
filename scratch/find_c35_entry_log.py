import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/archive_cycle35/live_champion_challenger.log', encoding='utf-8') as f:
    for line in f:
        if '13:50:' in line or '13:51:' in line or '13:52:' in line or '13:53:' in line or '13:54:' in line or '13:55:' in line:
            if 'INFO' in line and ('#' in line or 'ENTRY' in line or 'FILLED' in line or 'ORDER' in line or 'Trade' in line):
                print(line.strip())

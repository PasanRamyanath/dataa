with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    matches = [l.strip() for l in f if 'RESTING' in l or 'entry' in l.lower() or 'placed' in l.lower() or 'stalking' in l.lower()]
print(f"Total matching lines in log: {len(matches)}")
for m in matches[:20]:
    print(m)

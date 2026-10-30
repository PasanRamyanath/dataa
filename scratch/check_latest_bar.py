with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    lines = [l.strip() for l in f if 'BTC: $' in l]

print(f"Total bars logged: {len(lines)}")
for l in lines[-5:]:
    print(l)

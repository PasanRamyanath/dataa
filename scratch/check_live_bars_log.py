# Let's inspect the live session task log to see what the live bar log shows
with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"Total lines in live_champion_challenger.log: {len(lines)}")
# Print lines with [INFO] and BTC:
btc_lines = [l.strip() for l in lines if 'BTC: $' in l]
print(f"Total bar lines: {len(btc_lines)}")
for l in btc_lines[-10:]:
    print(l)

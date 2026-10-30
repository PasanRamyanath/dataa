with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    lines = [line.strip() for line in f if 'ENTRY' in line or 'EXIT' in line or 'CLOSE' in line or 'TRADE' in line or 'ORDER' in line or 'BUY' in line or 'SELL' in line]
    print(f"Total matching lines: {len(lines)}")
    for l in lines[:30]:
        print(l)

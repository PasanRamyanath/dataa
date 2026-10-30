import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/live_champion_challenger.log', encoding='utf-8', errors='ignore') as f:
    all_lines = f.readlines()
print(f"Total lines: {len(all_lines)}")
print("--- Last 40 lines ---")
for l in all_lines[-40:]:
    print(l.strip())

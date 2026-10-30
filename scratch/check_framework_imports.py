import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    for i in range(70):
        print(f.readline().strip())

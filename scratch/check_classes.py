with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    for line in f:
        if line.startswith('class '):
            print(line.strip())

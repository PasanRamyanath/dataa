with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(15, 45):
    print(f"L{i+1}: {lines[i]}", end="")

import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class Cycle35_ReversionPureScalper')
print(text[idx:idx+1500])

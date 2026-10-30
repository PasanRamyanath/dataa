import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/hierarchical_sniper_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = 25412
print(text[idx:idx+2500])

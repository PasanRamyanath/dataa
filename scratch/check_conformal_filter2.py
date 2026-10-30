import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('cand_dir = 2')
if idx != -1:
    print(text[idx-50:idx+2500])

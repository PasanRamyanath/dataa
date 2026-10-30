with open('feature engineering scripts/execution/hierarchical_sniper_engine.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('RESTING_ENTRY_QUEUE')
while idx != -1:
    print(text[idx-50:idx+300])
    print("="*40)
    idx = text.find('RESTING_ENTRY_QUEUE', idx+1)

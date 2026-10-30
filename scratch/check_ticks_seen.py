with open('feature engineering scripts/execution/hierarchical_sniper_engine.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('self.ticks_seen += 1')
if idx != -1:
    print(text[idx-200:idx+400])

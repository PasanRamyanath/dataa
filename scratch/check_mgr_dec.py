with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('def on_hierarchical_decision(self, decision')
if idx != -1:
    print(text[idx-50:idx+600])

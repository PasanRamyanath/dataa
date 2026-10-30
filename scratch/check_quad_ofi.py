with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class QuadHorizonOFIScalper')
if idx != -1:
    print(text[idx:idx+800])

with open('feature engineering scripts/scratch/run_1h_champion_challenger_session.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('LEDGER_PATH')
while idx != -1:
    print(text[idx-50:idx+250])
    print("="*40)
    idx = text.find('LEDGER_PATH', idx+1)

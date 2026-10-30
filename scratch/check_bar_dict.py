with open('feature engineering scripts/scratch/run_1h_champion_challenger_session.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('def process_10s_bar')
if idx != -1:
    print(text[idx:idx+800])

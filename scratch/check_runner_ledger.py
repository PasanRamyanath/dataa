with open('feature engineering scripts/scratch/run_1h_champion_challenger_session.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('live_hierarchical_session')
if idx != -1:
    print(text[idx-100:idx+300])

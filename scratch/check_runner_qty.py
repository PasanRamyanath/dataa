with open('feature engineering scripts/scratch/run_1h_champion_challenger_session.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('bbo_bid_qty')
while idx != -1:
    print(text[idx-100:idx+250])
    print("="*40)
    idx = text.find('bbo_bid_qty', idx+1)

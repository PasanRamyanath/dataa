with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class ChampionChallengerTournamentManager:')
idx2 = text.find('"Challenger_4"', idx)
print(text[idx2:idx2+600])

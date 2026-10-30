with open('feature engineering scripts/execution/champion_challenger_framework.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class ChampionChallengerTournamentManager:')
print(text[idx:idx+2000])

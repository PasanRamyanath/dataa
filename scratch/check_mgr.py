with open('feature engineering scripts/execution/champion_challenger_framework.py', encoding='utf-8') as f:
    text = f.read()

idx = text.find('class ChampionChallengerTournamentManager')
if idx != -1:
    print(text[idx:idx+2500])

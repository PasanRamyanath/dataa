import subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')
out = subprocess.check_output(['git', 'show', '37b3b7d4:feature engineering scripts/execution/champion_challenger_framework.py'], encoding='utf-8')
idx = out.find('"Challenger_5":')
print(out[max(0, idx-100):idx+500])

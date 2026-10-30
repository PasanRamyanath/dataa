import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
arch = 'archive_cycle52'
ledgers = glob.glob(os.path.join('scratch', arch, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

rolling = []
for b in range(63):
    mid = bars[b].get('bbo_mid', bars[b].get('mid', 0.0))
    rolling.append(mid)

sess_high = max(rolling)
sess_low = min(rolling)
rng = sess_high - sess_low
pct = (rolling[-1] - sess_low) / rng
print(f"At Bar 62: low={sess_low:.2f}, high={sess_high:.2f}, rng={rng:.2f}, mid={rolling[-1]:.2f}, pct={pct:.4f}")

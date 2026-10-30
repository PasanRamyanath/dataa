import sys, os, glob, json
sys.stdout.reconfigure(encoding='utf-8')
arch = 'archive_cycle52'
ledgers = glob.glob(os.path.join('scratch', arch, '*ledger*.jsonl'))
with open(ledgers[0], 'r', encoding='utf-8') as f:
    bars = [json.loads(l) for l in f if l.strip().startswith('{')]

entry = 79871.9
for b in range(62, 110):
    mid = bars[b].get('bbo_mid', bars[b].get('mid', 0.0))
    tact = bars[b].get('tactical_probs', [0.33, 0.33, 0.33])
    meso = bars[b].get('meso_probs', [0.33, 0.33, 0.33])
    macro = bars[b].get('macro_probs', [0.33, 0.33, 0.33])
    p_dir = bars[b].get('primary_direction', 1)
    unreal = (entry - mid) / entry * 1e4
    print(f"Bar {b:3d}: mid={mid:.2f}, unreal={unreal:+.2f}b, p_dir={p_dir}, tact_d={tact[2]-tact[0]:+.3f}, meso_d={meso[2]-meso[0]:+.3f}, macro_d={macro[2]-macro[0]:+.3f}")

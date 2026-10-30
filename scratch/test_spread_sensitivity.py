import json
import os

for c in [35, 43, 48, 54]:
    fpath = f"scratch/archive_cycle{c}/live_hierarchical_session_1h_ledger.jsonl"
    if not os.path.exists(fpath):
        continue
    
    cnt_14 = 0
    cnt_08 = 0
    total = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            total += 1
            t_probs = d.get('tactical_probs', [0.33, 0.34, 0.33])
            m_probs = d.get('meso_probs', [0.33, 0.34, 0.33])
            mic_probs = d.get('micro_probs', [0.33, 0.34, 0.33])
            
            d_tact = t_probs[2] - t_probs[0]
            d_meso = m_probs[2] - m_probs[0]
            d_mic  = mic_probs[2] - mic_probs[0]
            
            # Classic 0.14
            if d_tact >= 0.14 and d_meso >= 0.03 and d_mic > 0.015:
                cnt_14 += 1
            # Adaptive 0.08
            if d_tact >= 0.08 and d_meso >= 0.0 and d_mic > 0.010:
                cnt_08 += 1
                
    print(f"Cycle {c:02d} ({total} bars) | Classic (0.14 spread): {cnt_14} bars | Adaptive (0.08 spread): {cnt_08} bars")

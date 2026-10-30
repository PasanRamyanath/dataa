import json

with open("scratch/archive_cycle36/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

for i, b in enumerate(bars):
    conf_tact = max(b["tactical_probs"])
    conf_meso = max(b["meso_probs"])
    if conf_tact < 0.45 or conf_meso < 0.45:
        continue
    d_tact = b["tactical_probs"][2] - b["tactical_probs"][0]
    d_micro = b["micro_probs"][2] - b["micro_probs"][0]
    d_meso = b["meso_probs"][2] - b["meso_probs"][0]
    
    tact_dir = 2 if d_tact >= 0.12 else (0 if d_tact <= -0.12 else 1)
    tact_confirmed = (tact_dir in (0, 2)) and (
        (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.03) or
        (tact_dir == 0 and d_micro < -0.015 and d_meso <= -0.03)
    )
    if tact_confirmed:
        bid = b["bbo_bid"]
        print(f"Bar {i:3d} (t={i*10:4d}s): Dir={tact_dir}, d_tact={d_tact:.3f}, d_micro={d_micro:.3f}, d_meso={d_meso:.3f}, bid={bid:.1f}")

import json

with open("scratch/archive_cycle36/live_hierarchical_session_1h_ledger.jsonl") as f:
    bars = [json.loads(line) for line in f if line.strip()]

rolling_15m = []
for i, b in enumerate(bars):
    mid = b["bbo_mid"]
    rolling_15m.append(mid)
    if len(rolling_15m) > 90:
        rolling_15m.pop(0)

    conf_tact = max(b["tactical_probs"])
    conf_meso = max(b["meso_probs"])
    if conf_tact < 0.45 or conf_meso < 0.45:
        continue
    d_tact = b["tactical_probs"][2] - b["tactical_probs"][0]
    d_micro = b["micro_probs"][2] - b["micro_probs"][0]
    d_meso = b["meso_probs"][2] - b["meso_probs"][0]
    
    tact_dir = 2 if d_tact >= 0.11 else (0 if d_tact <= -0.11 else 1)
    tact_confirmed = (tact_dir in (0, 2)) and (
        (tact_dir == 2 and d_micro > 0.015 and d_meso >= 0.02) or
        (tact_dir == 0 and d_micro < -0.015 and d_meso <= 0.01)
    )
    if not tact_confirmed:
        continue

    sess_high = max(rolling_15m)
    sess_low = min(rolling_15m)
    high_viol = tact_dir == 2 and (mid >= sess_high - 10.0)
    low_viol = tact_dir == 0 and (mid <= sess_low + 10.0)

    print(f"Bar {i:3d} (t={i*10:4d}s) | Dir={tact_dir} | mid={mid:.2f} | 15m High={sess_high:.2f} Low={sess_low:.2f} | HighViol={high_viol} LowViol={low_viol}")

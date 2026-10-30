import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
champ = mgr.competitors["Champion"].engine
champ.ticks_seen = 50

print(f"Total bars to evaluate: {len(bars)}")

breakdown = {}

for b_idx, bar in enumerate(bars):
    mid = bar.get('bbo_mid')
    macro = tuple(bar.get('macro_probs'))
    meso = tuple(bar.get('meso_probs'))
    tact = tuple(bar.get('tactical_probs'))
    micro = tuple(bar.get('micro_probs'))
    prim_dir = bar.get('primary_direction')
    edge = bar.get('expected_net_edge_bps')
    
    d_macro = macro[2] - macro[0]
    d_meso = meso[2] - meso[0]
    d_tact = tact[2] - tact[0]
    d_micro = micro[2] - micro[0]
    
    # 1. Quad-horizon check in Challenger1_QuadHorizonConformalResonanceScalper
    # quad_long = (d_micro > 0) and (d_tact > 0) and (d_meso >= 0) and (d_macro > -0.05)
    if d_tact <= 0:
        breakdown["1. d_tact <= 0"] = breakdown.get("1. d_tact <= 0", 0) + 1
        continue
    if d_micro <= 0:
        breakdown["2. d_micro <= 0"] = breakdown.get("2. d_micro <= 0", 0) + 1
        continue
    if d_meso < 0:
        breakdown["3. d_meso < 0"] = breakdown.get("3. d_meso < 0", 0) + 1
        continue
    if d_macro <= -0.05:
        breakdown["4. d_macro <= -0.05"] = breakdown.get("4. d_macro <= -0.05", 0) + 1
        continue
        
    # 2. Conformal check in ConformalConfidenceFilterScalper
    conf_tact = max(tact)
    conf_meso = max(meso)
    has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.40) or (
        d_tact >= 0.12 and d_meso >= 0.03 and edge >= 10.0
    )
    if not has_conviction:
        breakdown["5. No Conviction (tact/meso conf & spread)"] = breakdown.get("5. No Conviction (tact/meso conf & spread)", 0) + 1
        continue
        
    # 3. tact_confirmed check
    tact_confirmed = (d_tact >= 0.10 and d_micro > 0.010 and d_meso >= 0.02)
    cand_dir = 2 if tact_confirmed else None
    if cand_dir is None and prim_dir == 2 and edge >= champ.min_net_edge_bps and d_meso >= 0.02:
        cand_dir = 2
        
    if cand_dir != 2:
        breakdown["6. cand_dir != 2 (tact_confirmed & prim_dir)"] = breakdown.get("6. cand_dir != 2 (tact_confirmed & prim_dir)", 0) + 1
        continue
        
    # 4. Range boundary
    # Let's seed champ with recent prices so range boundary can be evaluated
    champ.rolling_15m_prices.append(mid)
    if champ._is_range_boundary_violation(cand_dir, mid):
        breakdown["7. Range Boundary Violation"] = breakdown.get("7. Range Boundary Violation", 0) + 1
        continue
        
    # 5. Parabolic surge
    if champ._is_parabolic_surge(cand_dir, bar.get('ts_ms'), mid):
        breakdown["8. Parabolic Surge Violation"] = breakdown.get("8. Parabolic Surge Violation", 0) + 1
        continue
        
    breakdown["9. ALL CHECKS PASSED -> ENTRY TRIGGER!"] = breakdown.get("9. ALL CHECKS PASSED -> ENTRY TRIGGER!", 0) + 1
    print(f"Bar {b_idx} PASSED: Price={mid} Edge={edge} d_tact={d_tact:.3f} d_meso={d_meso:.3f} d_micro={d_micro:.3f} d_macro={d_macro:.3f}")

print("\n=== BREAKDOWN OF FILTER GATING ===")
for k, v in sorted(breakdown.items()):
    print(f"{k}: {v} bars")

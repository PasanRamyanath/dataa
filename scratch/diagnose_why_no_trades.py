import json, sys
sys.path.insert(0, 'feature engineering scripts')
sys.stdout.reconfigure(encoding='utf-8')

from fusion.hierarchical_fusion_v3 import HierarchicalDecisionOutput
from execution.champion_challenger_framework import ChampionChallengerTournamentManager

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', encoding='utf-8') as f:
    bars = [json.loads(line) for line in f if line.strip()]

mgr = ChampionChallengerTournamentManager(symbol="BTCUSDT")
champ = mgr.competitors["Champion"].engine

reasons = {}

for b_idx, bar in enumerate(bars):
    ts_ms = bar.get('ts_ms')
    mid = bar.get('bbo_mid')
    bid = bar.get('bbo_bid', mid - 0.05)
    ask = bar.get('bbo_ask', mid + 0.05)
    bid_qty = 1.0
    ask_qty = 1.0
    
    macro = tuple(bar.get('macro_probs'))
    meso = tuple(bar.get('meso_probs'))
    tact = tuple(bar.get('tactical_probs'))
    micro = tuple(bar.get('micro_probs'))
    prim_dir = bar.get('primary_direction')
    edge = bar.get('expected_net_edge_bps')
    
    dec = HierarchicalDecisionOutput(
        ts_ms=ts_ms,
        macro_probs=macro,
        meso_probs=meso,
        tactical_probs=tact,
        micro_probs=micro,
        fused_swing_probs=(0.33, 0.34, 0.33),
        primary_direction=prim_dir,
        cross_horizon_alignment=0.8,
        expected_gross_move_bps=20.0,
        expected_net_edge_bps=edge,
        meta_act_probability=0.8,
        micro_sniper_trigger_prob=0.8,
        target_holding_seconds=120,
        take_profit_bps=15.0,
        stop_loss_bps=12.0,
        kelly_position_fraction=1.0,
        execution_mode="MAKER_STALKING"
    )
    
    # Check what happens on champ
    # First tick
    champ.on_market_micro_tick(ts_ms=ts_ms, bbo_bid=bid, bbo_ask=ask, bbo_bid_qty=bid_qty, bbo_ask_qty=ask_qty)
    
    # Let's inspect each gating condition in Champion manually
    # 1. State
    if champ.state != "IDLE":
        reasons["NOT_IDLE"] = reasons.get("NOT_IDLE", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
    
    if champ.ticks_seen < 10:
        reasons["TICKS_SEEN_LESS_10"] = reasons.get("TICKS_SEEN_LESS_10", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    d_macro = macro[2] - macro[0]
    d_meso = meso[2] - meso[0]
    d_tact = tact[2] - tact[0]
    d_micro = micro[2] - micro[0]
    
    # Quad-horizon agreement check: Strictly Long-Only
    quad_long = (d_micro > 0) and (d_tact > 0) and (d_meso >= 0) and (d_macro > -0.05)
    if not quad_long:
        sub_fail = []
        if d_micro <= 0: sub_fail.append("d_micro<=0")
        if d_tact <= 0: sub_fail.append("d_tact<=0")
        if d_meso < 0: sub_fail.append("d_meso<0")
        if d_macro <= -0.05: sub_fail.append("d_macro<=-0.05")
        reason_str = f"QUAD_HORIZON_FAIL: {', '.join(sub_fail)}"
        reasons[reason_str] = reasons.get(reason_str, 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
    
    # OFI check
    if d_tact > 0 and (bid_qty / ask_qty) < 1.02:
        reasons["OFI_IMBALANCE_FAIL"] = reasons.get("OFI_IMBALANCE_FAIL", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    # Conformal check in ConformalConfidenceFilterScalper:
    conf_tact = max(tact)
    conf_meso = max(meso)
    has_conviction = (conf_tact >= 0.42 and conf_meso >= 0.40) or (
        d_tact >= 0.12 and d_meso >= 0.03 and edge >= 10.0
    )
    if not has_conviction:
        reasons[f"NO_CONVICTION: conf_tact={conf_tact:.3f}, conf_meso={conf_meso:.3f}, d_tact={d_tact:.3f}, d_meso={d_meso:.3f}, edge={edge:.1f}"] = reasons.get("NO_CONVICTION", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    # Tact confirmed check
    tact_confirmed = (d_tact >= 0.10 and d_micro > 0.010 and d_meso >= 0.02)
    cand_dir = 2 if tact_confirmed else None
    if cand_dir is None and prim_dir == 2 and edge >= champ.min_net_edge_bps and d_meso >= 0.02:
        cand_dir = 2
        
    if cand_dir != 2:
        reasons["CAND_DIR_NOT_LONG"] = reasons.get("CAND_DIR_NOT_LONG", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    # Range boundary
    if champ._is_range_boundary_violation(cand_dir, mid):
        reasons["RANGE_BOUNDARY_VIOLATION"] = reasons.get("RANGE_BOUNDARY_VIOLATION", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    # Parabolic surge
    if champ._is_parabolic_surge(cand_dir, ts_ms, mid):
        reasons["PARABOLIC_SURGE_VIOLATION"] = reasons.get("PARABOLIC_SURGE_VIOLATION", 0) + 1
        champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)
        continue
        
    reasons["PASSED_ALL_ENTRY_CHECKS!"] = reasons.get("PASSED_ALL_ENTRY_CHECKS!", 0) + 1
    champ.on_hierarchical_decision(dec, bid, ask, bid_qty, ask_qty)

print("=== REASONS WHY CHAMPION DID NOT ENTER ON BARS ===")
for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"{count:3d} bars: {r}")

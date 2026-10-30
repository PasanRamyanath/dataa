import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        idx = data['bar_idx']
        if (93 <= idx <= 125) or (150 <= idx <= 165):
            print(f"Bar {idx:03d} | Mid: {data['bbo_mid']} | State: {data.get('fsm_state')} | Pos: {'YES' if data.get('active_position') else 'NO'} | Macro: {data.get('macro_probs')} | Meso: {data.get('meso_probs')} | Tact: {data.get('tactical_probs')} | Micro: {data.get('micro_probs')} | Edge: {data.get('expected_net_edge_bps')} | Mode: {data.get('execution_mode')}")

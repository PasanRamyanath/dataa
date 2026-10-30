import json

with open('scratch/live_hierarchical_session_1h_ledger.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        b = d['bar_idx']
        if (35 <= b <= 42) or (115 <= b <= 130):
            macro = [round(x, 4) for x in d['macro_probs']]
            meso = [round(x, 4) for x in d['meso_probs']]
            tact = [round(x, 4) for x in d['tactical_probs']]
            micro = [round(x, 4) for x in d['micro_probs']]
            edge = d['expected_net_edge_bps']
            p_dir = d['primary_direction']
            mode = d['execution_mode']
            fsm = d['fsm_state']
            pos = d['active_position']
            print(f"Bar {b:3d} | mid={d['bbo_mid']:.2f} | edge={edge:+6.2f} | dir={p_dir:+2d} | mode={mode:12s} | fsm={fsm:10s} | macro={macro} | meso={meso} | tact={tact} | micro={micro}")
            if pos:
                print(f"   ACTIVE POS: side={pos.get('side')} entry={pos.get('entry_price')} sl={pos.get('sl_price')} tp={pos.get('tp_price')}")

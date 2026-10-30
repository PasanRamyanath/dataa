import json
from pathlib import Path

# Let's inspect Cycle 11 Trade 1, Cycle 15 Trade 2, Cycle 16 Trade 1, Cycle 19 Trade 1
trades_to_inspect = [
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"), "Cycle 11 Trade 1", 173, 203),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"), "Cycle 15 Trade 2", 183, 194),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle16"), "Cycle 16 Trade 1", 81, 102),
    (Path(r"D:\PrivateProjects\BTC Futures\scratch"), "Cycle 19 Trade 1", 49, 64),
]

for arch, name, start_b, end_b in trades_to_inspect:
    ledger_files = list(arch.glob("*hierarchical_session_1h_ledger.jsonl"))
    if not ledger_files:
        continue
    with open(ledger_files[0], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    
    print(f"\n==================== {name} ====================")
    entry_p = None
    side = None
    for r in rows:
        if r['bar_idx'] == start_b:
            ap = r.get('active_position')
            if ap:
                entry_p = ap['entry_price']
                side = ap['side']
            break
            
    print(f"Trade Side: {side} | Entry Price: {entry_p}")
    print(f"{'Bar':4s} | {'Time':8s} | {'BBO Mid':8s} | {'Unrealized':10s} | {'Peak MFE':9s} | {'Dir':5s} | {'Edge':6s} | {'Tac Probs [Dn, Fl, Up]':24s} | {'Micro Probs [Dn, Fl, Up]':24s}")
    print("-" * 110)
    
    for r in rows:
        b = r['bar_idx']
        if start_b <= b <= end_b:
            mid = r['bbo_mid']
            unreal = 0.0
            if entry_p:
                unreal = ((mid - entry_p) / entry_p * 1e4) if side == 'BUY' else ((entry_p - mid) / entry_p * 1e4)
            ap = r.get('active_position')
            pfe = ap.get('peak_favorable_excursion_bps', 0.0) if ap else 0.0
            pdir = "SHORT" if r['primary_direction'] == 0 else ("LONG" if r['primary_direction'] == 2 else "FLAT")
            edge = r['expected_net_edge_bps']
            tac = [round(x, 2) for x in r['tactical_probs']]
            micro = [round(x, 2) for x in r['micro_probs']]
            print(f"{b:4d} | {r['datetime'][11:19]} | {mid:8.2f} | {unreal:+9.2f}b | {pfe:8.2f}b | {pdir:5s} | {edge:5.2f}b | {str(tac):24s} | {str(micro):24s}")

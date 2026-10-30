import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== TRADE 2 (CYCLE 19): WHAT WERE MODELS DOING DURING ADVERSE MOVE? ===")
print(f"{'Bar':4s} | {'Time':8s} | {'Mid':8s} | {'Unrealized':10s} | {'Dir':5s} | {'Edge':6s} | {'Meso [Dn, Fl, Up]':24s} | {'Tac [Dn, Fl, Up]':24s} | {'Micro [Dn, Fl, Up]':24s}")
print("-" * 115)

entry_p = 80000.4
for b in range(158, 181):
    r = rows[b]
    mid = r['bbo_mid']
    # Short trade: unrealized is (entry - mid)
    unreal = ((entry_p - mid) / entry_p) * 1e4
    pdir = "SHORT" if r['primary_direction'] == 0 else ("LONG" if r['primary_direction'] == 2 else "FLAT")
    edge = r['expected_net_edge_bps']
    meso = [round(x, 2) for x in r['meso_probs']]
    tac = [round(x, 2) for x in r['tactical_probs']]
    micro = [round(x, 2) for x in r['micro_probs']]
    print(f"{r['bar_idx']:4d} | {r['datetime'][11:19]} | {mid:8.2f} | {unreal:+9.2f}b | {pdir:5s} | {edge:5.2f}b | {str(meso):24s} | {str(tac):24s} | {str(micro):24s}")

import json
from pathlib import Path

LEDGER_PATH = Path(r"D:\PrivateProjects\BTC Futures\scratch\live_hierarchical_session_1h_ledger.jsonl")

with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("=== INSPECTING PASSIVE_MAKER_SNIPER BARS ===")
for r in rows:
    if r['execution_mode'] == 'PASSIVE_MAKER_SNIPER':
        b_idx = r['bar_idx']
        dt = r['datetime'][11:19]
        mid = r['bbo_mid']
        pdir = r['primary_direction']
        edge = r['expected_net_edge_bps']
        fsm = r['fsm_state']
        print(f"Bar {b_idx} | {dt} | Mid: {mid} | Dir: {pdir} | Edge: {edge} | FSM: {fsm}")

print("\n=== TOP 10 MICRO PROBABILITIES DURING STALKING ===")
# Let's inspect micro_probs during WAIT_FOR_MICRO_SNIPER
micro_triggers = []
for r in rows:
    if r['execution_mode'] == 'WAIT_FOR_MICRO_SNIPER':
        # In ledger: micro_probs is [p_down, p_flat, p_up]
        m_probs = r['micro_probs']
        pdir = r['primary_direction']
        trig_prob = m_probs[2] if pdir == 2 else (m_probs[0] if pdir == 0 else 0.0)
        micro_triggers.append((r['bar_idx'], r['datetime'][11:19], r['bbo_mid'], pdir, trig_prob, r['expected_net_edge_bps']))

micro_triggers.sort(key=lambda x: x[4], reverse=True)
for mt in micro_triggers[:15]:
    print(f"Bar {mt[0]} | {mt[1]} | Mid: {mt[2]} | Dir: {mt[3]} | MicroTrigProb: {mt[4]:.4f} | Edge: {mt[5]:.2f}")

import json

with open("scratch/live_hierarchical_session_1h_ledger.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

for i in range(1, len(rows)):
    if rows[i].get("closed_trades_count", 0) > rows[i-1].get("closed_trades_count", 0):
        pos = rows[i-1].get("active_position")
        side = pos.get("side") if pos else "UNKNOWN"
        entry_p = pos.get("entry_price") if pos else 0.0
        mfe = pos.get("peak_favorable_excursion_bps") if pos else 0.0
        exit_p = rows[i]["bbo_mid"]
        sign = 1.0 if side == "BUY" else -1.0
        gross = sign * ((exit_p - entry_p) / entry_p) * 1e4 if entry_p > 0 else 0.0
        print(f"Trade #{rows[i]['closed_trades_count']}: Bar {rows[i]['bar_idx']} | {side} @ {entry_p:.2f} -> Exit @ {exit_p:.2f} | MFE: {mfe:+.2f} bps | Gross: {gross:+.2f} bps")

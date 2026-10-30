import json
from pathlib import Path

for c in [35, 60, 61, 62, 63]:
    d = Path(f"scratch/archive_cycle{c}")
    res_f = d / "champion_challenger_final_results.json"
    if not res_f.exists():
        continue
    data = json.loads(res_f.read_text(encoding="utf-8"))
    print("="*80)
    print(f"CYCLE {c} DETAILS:")
    print(f"Winner: {data.get('tournament_winner')}")
    print(f"Winner Profile: {data.get('winner_profile')}")
    print(f"Winner Net PnL: {data.get('winner_net_pnl_bps')} bps | WR: {data.get('winner_win_rate_pct')}% | Trades: {data.get('winner_trades')}")
    
    # Check trades of the winner
    ledgers = data.get("competitor_trade_ledgers", {})
    winner_name = data.get("tournament_winner")
    # find key matching winner_name
    w_trades = []
    for k, v in ledgers.items():
        if winner_name.lower() in k.lower() or ("champion" in k.lower() and "champion" in winner_name.lower()):
            w_trades = v
            break
    if not w_trades and "Champion" in ledgers:
        w_trades = ledgers["Champion"]
    
    print(f"Trades count: {len(w_trades)}")
    for i, t in enumerate(w_trades):
        side = t.get("side", "")
        entry = t.get("entry_price", 0.0)
        exit_p = t.get("exit_price", 0.0)
        net = t.get("net_pnl_bps", t.get("net_return_bps", 0.0))
        gross = t.get("gross_pnl_bps", t.get("gross_return_bps", 0.0))
        mfe = t.get("peak_mfe_bps", 0.0)
        mae = t.get("max_mae_bps", 0.0)
        dur = t.get("duration_sec", t.get("holding_duration_sec", 0.0))
        reason = t.get("exit_reason", "")
        print(f"  Trade {i+1}: {side:<4} @ {entry:.1f} -> {exit_p:.1f} | Net: {net:+6.2f} bps | Gross: {gross:+6.2f} bps | MFE: {mfe:+5.2f} | MAE: {mae:+5.2f} | Dur: {dur:4.0f}s | {reason}")

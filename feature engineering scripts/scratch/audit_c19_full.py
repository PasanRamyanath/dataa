"""
scratch/audit_c19_full.py
Audits Cycle 19 results from ledger and metrics files.
"""
import json
from pathlib import Path

ledger_path = Path("scratch/live_hierarchical_session_1h_ledger.jsonl")
metrics_path = Path("scratch/live_session_1h_final_metrics.json")
heartbeat_path = Path("scratch/live_session_1h_heartbeat.json")

print("=== CYCLE 19 AUDIT ===")
if heartbeat_path.exists():
    with open(heartbeat_path, "r", encoding="utf-8") as f:
        hb = json.load(f)
    print(f"Heartbeat Bars: {hb.get('bars_evaluated')}, Elapsed: {hb.get('elapsed_minutes')} min")
    print(f"Trades Count: {hb.get('trades_count')}, Wins: {hb.get('wins_count')}, WR: {hb.get('win_rate_pct')}%, Net PnL: {hb.get('net_pnl_bps')} bps")
    print("\nAccuracy Summary:")
    for tf, data in hb.get('accuracy_summary', {}).items():
        if tf == 'tiers':
            continue
        print(f"  {tf:4s}: Overall={data.get('overall_accuracy_pct', 0.0):5.1f}% ({data.get('evaluated_samples', 0)} smp), Active Precision={data.get('active_precision_pct', 0.0):5.1f}% ({data.get('active_signals_count', 0)} sig)")
    tiers = hb.get('accuracy_summary', {}).get('tiers', {})
    print(f"  Tiers: Micro={tiers.get('micro', {}).get('accuracy_pct', 0.0):5.1f}%, Tactical={tiers.get('tactical', {}).get('accuracy_pct', 0.0):5.1f}%, Meso={tiers.get('meso', {}).get('accuracy_pct', 0.0):5.1f}%, Macro={tiers.get('macro', {}).get('accuracy_pct', 0.0):5.1f}%")

if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        met = json.load(f)
    print("\nFinal Metrics Trades:")
    for t in met.get('closed_trades', []):
        print(f"  {t.get('side')} | Entry: {t.get('entry_price')} | Exit: {t.get('exit_price')} | Gross: {t.get('gross_return_bps'):+6.2f} bps | Net: {t.get('net_return_bps'):+6.2f} bps | Reason: {t.get('exit_reason')}")

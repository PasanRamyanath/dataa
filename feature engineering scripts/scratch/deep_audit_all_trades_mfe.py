import json
from pathlib import Path

archives = [
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle5"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle6"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle7"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle8"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle9"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle10"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle11"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle12"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle13"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle14"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle15"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle16"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle17"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch\archive_cycle18"),
    Path(r"D:\PrivateProjects\BTC Futures\scratch"), # cycle 19
]

print("=== DEEP AUDIT OF ALL HISTORICAL TRADES: MFE vs REALIZED PROFIT ===")
print(f"{'Cycle':12s} | {'Bar':4s} | {'Side':4s} | {'Entry':8s} | {'Peak MFE':9s} | {'Exit P':8s} | {'Gross PnL':9s} | {'Net PnL':9s} | {'Giveback':9s} | {'Exit Reason':25s}")
print("-" * 115)

for arch in archives:
    c_name = arch.name if "archive" in arch.name else "cycle19_active"
    metrics_files = list(arch.glob("*final_metrics.json"))
    if not metrics_files:
        continue
    with open(metrics_files[0], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    closed = metrics.get('closed_trades', [])
    for t in closed:
        side = t['side']
        entry_p = t['entry_price']
        exit_p = t['exit_price']
        peak_mfe = t['peak_mfe_bps']
        gross = t['gross_return_bps']
        net = t['net_return_bps']
        reason = t['exit_reason']
        giveback = peak_mfe - gross if peak_mfe > 0 else 0.0
        print(f"{c_name:12s} | {t.get('entry_ts_ms', 0)} | {side:4s} | {entry_p:8.2f} | {peak_mfe:+8.2f}b | {exit_p:8.2f} | {gross:+8.2f}b | {net:+8.2f}b | {giveback:8.2f}b | {reason:25s}")

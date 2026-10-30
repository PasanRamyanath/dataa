from benchmark_framework import UnifiedPeakBenchmarkEngine

durations = [180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 600.0, 900.0]
hurdles = [3.0, 3.5, 3.8]

print("="*105)
print("SWEEP OF FIX 2: ASYMMETRIC MFE-GATED DEAD-TRADE INVALIDATION (WITH FIX 1 & FIX 3 ACTIVE)")
print("="*105)
print(f"{'Inval Window (Dur, Hurdle)':<30} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
print("-" * 105)

for h in hurdles:
    for d in durations:
        eng = UnifiedPeakBenchmarkEngine(
            fix1_taker_stop=True,
            fix3_max_entry_bar=270,
            fix2_dead_trade_inval_sec=d,
            fix2_mfe_hurdle=h
        )
        res = eng.evaluate_all_cycles()
        label = f"Dur {int(d)}s ({d/60:.1f}m), Hurdle {h:.1f}b"
        print(f"{label:<30} | {res['total_trades']:<6} | {res['win_rate']:>6.2f}% | {res['total_net_pnl']:>+12.2f} bps | {res['avg_net_trade']:>+6.2f}b | {res['wr100_sessions']}/{res['active_sessions']} ({res['wr100_sessions']/res['active_sessions']*100:.1f}%)")
print("="*105)

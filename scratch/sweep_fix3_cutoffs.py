from benchmark_framework import UnifiedPeakBenchmarkEngine

cutoffs = [240, 260, 270, 280, 300, 360]

print("="*95)
print("SWEEP OF FIX 3: PRE-SESSION-END RUNWAY CUTOFF (MAX ENTRY BAR)")
print("="*95)
print(f"{'Max Entry Bar (Cutoff)':<25} | {'Trades':<6} | {'Win Rate':<8} | {'Net Realized PnL':<16} | {'Avg/Tr':<8} | {'100% WR Cycles':<14}")
print("-" * 95)

for bar_lim in cutoffs:
    eng = UnifiedPeakBenchmarkEngine(fix1_taker_stop=True, fix3_max_entry_bar=bar_lim)
    res = eng.evaluate_all_cycles()
    label = f"Bar {bar_lim} ({bar_lim*10//60} mins)" if bar_lim < 360 else "Bar 360 (No Cutoff)"
    print(f"{label:<25} | {res['total_trades']:<6} | {res['win_rate']:>6.2f}% | {res['total_net_pnl']:>+12.2f} bps | {res['avg_net_trade']:>+6.2f}b | {res['wr100_sessions']}/{res['active_sessions']} ({res['wr100_sessions']/res['active_sessions']*100:.1f}%)")
    print(f"  Exit Reasons: {res['reasons']}")
print("="*95)

from benchmark_framework import UnifiedPeakBenchmarkEngine

eng_baseline = UnifiedPeakBenchmarkEngine(fix1_taker_stop=False)
res_baseline = eng_baseline.evaluate_all_cycles()

eng_fix1 = UnifiedPeakBenchmarkEngine(fix1_taker_stop=True)
res_fix1 = eng_fix1.evaluate_all_cycles()

print("="*85)
print("STEP 1 AUDIT: IMPACT OF FIX 1 (TAKER EMERGENCY STOP EXECUTION)")
print("="*85)
print(f"Baseline (Passive Maker SL): Net PnL = {res_baseline['total_net_pnl']:+.2f} bps | Win Rate = {res_baseline['win_rate']:.2f}% | Wins/Losses = {res_baseline['wins']}/{res_baseline['losses']}")
print(f"Fix 1 (Guaranteed Taker SL): Net PnL = {res_fix1['total_net_pnl']:+.2f} bps | Win Rate = {res_fix1['win_rate']:.2f}% | Wins/Losses = {res_fix1['wins']}/{res_fix1['losses']}")
diff_pnl = res_fix1['total_net_pnl'] - res_baseline['total_net_pnl']
print(f"Net PnL Difference: {diff_pnl:+.2f} bps (Cost of guaranteed emergency taker execution across 7 stops)")
print(f"Trades hitting stop loss: {res_fix1['reasons']['STOP_LOSS']}")
for t in res_fix1['trades']:
    if t['reason'] == 'STOP_LOSS':
        print(f"  Cycle: {t['cycle']:<18} Side: {t['side']} Net: {t['net']:.2f}b Dur: {t['dur']:.1f}s MAE: {t['mae']:.2f}b")
print("="*85)

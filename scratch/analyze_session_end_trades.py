from benchmark_framework import UnifiedPeakBenchmarkEngine

eng = UnifiedPeakBenchmarkEngine(fix1_taker_stop=True)
res = eng.evaluate_all_cycles()

session_end_trades = [t for t in res['trades'] if t['reason'] == 'SESSION_END_MARK']
print("="*85)
print(f"ANATOMY OF THE {len(session_end_trades)} TRADES TRAPPED AT SESSION END")
print("="*85)
print(f"{'Cycle':<18} | {'Side':<4} | {'Entry Bar':<9} | {'Duration (s)':<12} | {'Net PnL':<10} | {'MAE':<8} | {'MFE':<8}")
print("-" * 85)
for t in session_end_trades:
    print(f"{t['cycle']:<18} | {t['side']:<4} | Bar {t['entry_bar']:<5} | {t['dur']:>10.1f}s | {t['net']:>+8.2f}b | {t['mae']:>+6.2f}b | {t['mfe']:>+6.2f}b")

entry_bars = [t['entry_bar'] for t in session_end_trades]
print("="*85)
print(f"Total Net PnL of Session-End Trades: {sum(t['net'] for t in session_end_trades):+.2f} bps")
print(f"Entry Bar Distribution:")
print(f"  Entered after Bar 270 (last 15m): {sum(1 for b in entry_bars if b >= 270)} of {len(entry_bars)}")
print(f"  Entered after Bar 240 (last 20m): {sum(1 for b in entry_bars if b >= 240)} of {len(entry_bars)}")
print(f"  Entered after Bar 180 (second half of session): {sum(1 for b in entry_bars if b >= 180)} of {len(entry_bars)}")
print(f"  Entered in first half (Bar < 180, i.e. held >30 min without resolving): {sum(1 for b in entry_bars if b < 180)} of {len(entry_bars)}")

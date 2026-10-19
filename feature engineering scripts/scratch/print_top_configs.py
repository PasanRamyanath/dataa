import fine_grid_paper_engine as fg

bars = fg.load_bars()
res = fg.test_fine_grid(bars)

# Find configurations with positive Net PnL and various trade counts
print("=" * 80)
print("ALL CONFIGURATIONS WITH POSITIVE NET PNL (> +2.0 bps):")
print("=" * 80)
seen = set()
for r in res:
    if r["net"] > 2.0:
        key = (r["tp"], r["sl"], r["hold"], r["trail"])
        if key not in seen:
            seen.add(key)
            print(f"TP: {r['tp']:4.1f} | SL: {r['sl']:4.1f} | MaxHold: {r['hold']:2d} bars ({r['hold']*10}s) | "
                  f"Trail: {str(r['trail']):4s} | Trades: {r['n']:2d} | WinRate: {r['win_rate']:5.1f}% | "
                  f"Gross: {r['gross']:+6.2f} bps | Net: {r['net']:+6.2f} bps | PF: {r['pf']:4.2f}")

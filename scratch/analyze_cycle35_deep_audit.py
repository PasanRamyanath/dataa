"""
scratch/analyze_cycle35_deep_audit.py
Exhaustive re-analysis of Cycle 35:
- Regime phases & price trajectory
- Bar-by-bar model behavior during Phase 1 (Bar 11 loss), Phase 3 (Rally wins), and Phase 4 (Selloff)
- Exact entry/exit comparison across all 6 competitors
- Why no SHORT trades were taken during the 320-dollar selloff from 81,776 to 81,457
"""
import json
from pathlib import Path

def main():
    ledger_path = Path("scratch/archive_cycle35/live_hierarchical_session_1h_ledger.jsonl")
    results_path = Path("scratch/archive_cycle35/champion_challenger_final_results.json")

    bars = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))

    results = json.load(open(results_path, "r", encoding="utf-8"))

    print("=" * 100)
    print("CYCLE 35 DEEP FORENSIC AUDIT (360 BARS / 60.04 MINUTES)")
    print("=" * 100)

    # 1. Market Trajectory Overview
    prices = [b["bbo_mid"] for b in bars]
    start_p = prices[0]
    min_p = min(prices)
    max_p = max(prices)
    end_p = prices[-1]
    min_bar = prices.index(min_p) + 1
    max_bar = prices.index(max_p) + 1

    print(f"Start Price: ${start_p:.2f} (Bar 1)")
    print(f"Max Price:   ${max_p:.2f} (Bar {max_bar}) -> +{((max_p - start_p)/start_p)*1e4:.2f} bps from start")
    print(f"Min Price:   ${min_p:.2f} (Bar {min_bar}) -> {((min_p - start_p)/start_p)*1e4:.2f} bps from start")
    print(f"End Price:   ${end_p:.2f} (Bar {len(prices)}) -> Net session move: {((end_p - start_p)/start_p)*1e4:+.2f} bps")
    print(f"Total Range: ${max_p - min_p:.2f} ({((max_p - min_p)/min_p)*1e4:.2f} bps)")

    # 2. Competitor Leaderboard
    print("\n" + "=" * 100)
    print("FINAL COMPETITOR STANDINGS")
    print("=" * 100)
    for c in results["final_leaderboard"]:
        print(f"Rank {c['rank']}: {c['name']:14s} | Trades: {c['trades']} | Wins: {c['wins']} | Losses: {c['losses']} | WR: {c['win_rate_pct']:5.1f}% | Net PnL: {c['net_pnl_bps']:+6.2f} bps | Avg/Tr: {c['avg_trade_bps']:+5.2f} bps | PF: {c['profit_factor']:6.2f}")

    # 3. Detailed Trade Audit per Competitor
    print("\n" + "=" * 100)
    print("TRADE-BY-TRADE LEDGER AUDIT")
    print("=" * 100)
    for name, trades in results["competitor_trade_ledgers"].items():
        print(f"\n--- {name} ({len(trades)} Trades) ---")
        for i, t in enumerate(trades, 1):
            print(f"  T{i}: {t['side']} @ {t['entry_price']:.2f} -> {t['exit_price']:.2f} | Net: {t['net_return_bps']:+6.2f} bps (Gross: {t['gross_return_bps']:+6.2f}b, Fees: {t['total_fee_bps']:.1f}b) | MFE: {t['peak_mfe_bps']:+5.2f}b | MAE: {t['max_mae_bps']:+5.2f}b | Hold: {t['holding_duration_sec']:.0f}s | Reason: {t['exit_reason']}")

    # 4. Phase-by-Phase Model Diagnostics
    print("\n" + "=" * 100)
    print("PHASE-BY-PHASE MODEL AUDIT")
    print("=" * 100)

    # Phase 1: Bar 1 to 30
    p1_bars = bars[0:30]
    p1_dirs = [b["primary_direction"] for b in p1_bars]
    print(f"\nPhase 1 (Bars 1-30 | Initial Chop & Flash Dump):")
    print(f"  Price range: ${min(b['bbo_mid'] for b in p1_bars):.2f} to ${max(b['bbo_mid'] for b in p1_bars):.2f}")
    print(f"  Primary Directions: LONG={p1_dirs.count(2)}, FLAT={p1_dirs.count(1)}, SHORT={p1_dirs.count(0)}")
    b11 = bars[10]
    print(f"  Bar 11 (Trade 1 Trigger): Mid=${b11['bbo_mid']:.2f} | Dir={b11['primary_direction']} | Edge={b11['expected_net_edge_bps']:.2f} bps")
    print(f"    Macro probs: {b11['macro_probs']} | Meso probs: {b11['meso_probs']} | Tact probs: {b11['tactical_probs']} | Micro probs: {b11['micro_probs']}")
    print(f"    Max Tact Conf: {max(b11['tactical_probs']):.4f} | Max Meso Conf: {max(b11['meso_probs']):.4f}")

    # Phase 3: Bars 150 to 230 (The Rally)
    p3_bars = bars[150:230]
    p3_dirs = [b["primary_direction"] for b in p3_bars]
    print(f"\nPhase 3 (Bars 151-230 | Parabolic Surge $81,180 -> $81,776):")
    print(f"  Price range: ${min(b['bbo_mid'] for b in p3_bars):.2f} to ${max(b['bbo_mid'] for b in p3_bars):.2f}")
    print(f"  Primary Directions: LONG={p3_dirs.count(2)}, FLAT={p3_dirs.count(1)}, SHORT={p3_dirs.count(0)}")

    # Phase 4: Bars 231 to 360 (The Selloff from $81,776 to $81,457)
    p4_bars = bars[230:360]
    p4_dirs = [b["primary_direction"] for b in p4_bars]
    print(f"\nPhase 4 (Bars 231-360 | Top-out & Selloff $81,776 -> $81,457):")
    print(f"  Price range: ${min(b['bbo_mid'] for b in p4_bars):.2f} to ${max(b['bbo_mid'] for b in p4_bars):.2f}")
    print(f"  Primary Directions: LONG={p4_dirs.count(2)}, FLAT={p4_dirs.count(1)}, SHORT={p4_dirs.count(0)}")
    
    # Check why no SHORT was taken in Phase 4
    short_cand_bars = []
    for b in p4_bars:
        d_tact = b["tactical_probs"][2] - b["tactical_probs"][0]
        d_meso = b["meso_probs"][2] - b["meso_probs"][0]
        d_macro = b["macro_probs"][2] - b["macro_probs"][0]
        if d_tact <= -0.10 or b["primary_direction"] == 0:
            short_cand_bars.append((b["bar_idx"], b["bbo_mid"], b["primary_direction"], d_macro, d_meso, d_tact))

    print(f"\n  Bars with negative Tactical/Primary direction in Phase 4: {len(short_cand_bars)} bars")
    for b_idx, mid, p_dir, d_mac, d_mes, d_tac in short_cand_bars[:10]:
        print(f"    Bar {b_idx:3d} | Mid: ${mid:.2f} | Dir: {p_dir} | d_macro: {d_mac:+.4f} | d_meso: {d_mes:+.4f} | d_tact: {d_tac:+.4f}")

if __name__ == "__main__":
    main()

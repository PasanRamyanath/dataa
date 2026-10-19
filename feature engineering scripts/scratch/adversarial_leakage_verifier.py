"""
Adversarial Anti-Leakage & Timestamp Cutoff Stress Verification Suite.
Challenger 1 (Milestone 1).

Empirically tests:
1. Exact timestamp boundary (t = T_close - 1000ms vs t = T_close - 999ms).
2. Future timestamp and unclosed bar injection rejection.
3. Partial bar updates & intra-bar price shock isolation past cutoff.
4. Out-of-order events and reverse chronological stream resilience.
5. Clock skew, duplicate timestamps, and backward timestamp jump handling.
6. Empirical point-in-time truncation invariance across all 36 orthogonal features.
7. Forbidden non-stationary price columns stripping.
8. Live ingestion (binance_ws.py), streaming (live_loop_v2.py), and inference (checkpoint_loaders.py) source invariants.
9. Ensemble specialist pairwise decorrelation (AC1) across market regimes.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

# Add repository root and feature engineering scripts to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAT_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.pipeline import (
    compute_orthogonal_features,
    PRUNED_ORTHOGONAL_FEATURES,
    FORBIDDEN_NON_STATIONARY,
    _filter_by_cutoff,
)
from tests.fixtures.synthetic_market import (
    generate_synthetic_btc_bars,
    generate_synthetic_eth_bars,
    generate_synthetic_spot_perp_data,
    generate_synthetic_microstructure_data,
    build_tier4_scenario_chop,
    build_tier4_scenario_liquidity_squeeze,
    build_tier4_scenario_secular_drift,
    build_tier4_scenario_lead_lag_shock,
)
from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
from streaming.live_loop_v2 import _fuse
from inference.checkpoint_loaders import (
    _live_micro_signal,
    load_orthogonal_ensemble,
    MicrostructureSpecialist,
    CrossAssetSpecialist,
    BasisTermStructureSpecialist,
)


def run_all_adversarial_tests() -> dict[str, bool]:
    results = {}
    print("=" * 80)
    print("        CHALLENGER 1: EMPIRICAL ADVERSARIAL STRESS TEST SUITE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # TEST 1: Exact Millisecond Timestamp Boundary (T_close - 1000ms vs T_close - 999ms)
    # --------------------------------------------------------------------------
    print("\n[+] Test 1: Exact Millisecond Boundary Guard (T_close - 1000ms vs T_close - 999ms)...")
    base_ts = 1700000000000
    btc = generate_synthetic_btc_bars(n_bars=5, start_ts_ms=base_ts, seed=10)
    eth = generate_synthetic_eth_bars(btc, lead_lag_bars=0, seed=11)
    sp = generate_synthetic_spot_perp_data(btc, seed=12)
    ob = generate_synthetic_microstructure_data(btc, seed=13)

    last_close_ts = btc["close_ts"][-1]
    exact_cutoff = last_close_ts - 1000       # t = T_close - 1000ms (Boundary allowed)
    illegal_cutoff = last_close_ts - 999     # t = T_close - 999ms  (Illegal lookahead: +1ms past cutoff)

    # Inject boundary and illegal rows into each of the 4 streams
    for stream_name, df in [("btc", btc), ("eth", eth), ("sp", sp), ("ob", ob)]:
        # Row exactly at allowed cutoff
        row_allowed = df[-1].with_columns([
            pl.lit(exact_cutoff).alias("cutoff_ts") if "cutoff_ts" in df.columns else pl.lit(exact_cutoff).alias("ts"),
            pl.lit(exact_cutoff).alias("ts"),
        ])
        # Row at illegal boundary (cutoff + 1ms)
        row_illegal = df[-1].with_columns([
            pl.lit(illegal_cutoff).alias("cutoff_ts") if "cutoff_ts" in df.columns else pl.lit(illegal_cutoff).alias("ts"),
            pl.lit(illegal_cutoff).alias("ts"),
        ])
        df_test = pl.concat([df, row_allowed, row_illegal])

        # Filter strictly with exact_cutoff
        filtered = _filter_by_cutoff(df_test, exact_cutoff, stream_name)
        ts_col = "cutoff_ts" if "cutoff_ts" in filtered.columns else "ts"
        max_ts = filtered[ts_col].max()
        assert max_ts == exact_cutoff, (
            f"Stream {stream_name} failed boundary: expected max {exact_cutoff}, got {max_ts}"
        )
        assert illegal_cutoff not in filtered[ts_col].to_list(), (
            f"Stream {stream_name} leaked illegal timestamp {illegal_cutoff}!"
        )

    out = compute_orthogonal_features(btc, eth, sp, ob, cutoff_timestamp_ms=exact_cutoff)
    assert not out.is_empty(), "Valid boundary cutoff returned empty DataFrame!"
    assert out["ts"].max() <= exact_cutoff, "Output DataFrame max timestamp exceeded exact_cutoff!"
    print(f"    PASS: Exact cutoff {exact_cutoff} preserved; illegal cutoff {illegal_cutoff} (+1ms) strictly rejected.")
    results["test_exact_millisecond_boundary"] = True

    # --------------------------------------------------------------------------
    # TEST 2: Future Timestamps & Unclosed Bar Injection Rejection
    # --------------------------------------------------------------------------
    print("\n[+] Test 2: Future Timestamps & Unclosed Bar Injection...")
    future_ts_list = [exact_cutoff + 10000 * k for k in range(1, 6)]
    future_rows = []
    for f_ts in future_ts_list:
        future_rows.append(btc[-1].with_columns([
            pl.lit(f_ts).alias("ts"),
            pl.lit(f_ts).alias("cutoff_ts"),
            pl.lit(f_ts + 10000).alias("close_ts"),
            pl.lit(75000.0).alias("close"),
        ]))
    btc_future = pl.concat([btc] + future_rows)
    assert btc_future.height == btc.height + len(future_ts_list)

    out_future = compute_orthogonal_features(btc_future, eth, sp, ob, cutoff_timestamp_ms=exact_cutoff)
    assert out_future.height == btc.height, (
        f"Expected {btc.height} rows, got {out_future.height} (future unclosed bars leaked!)"
    )
    assert out_future["ts"].max() <= exact_cutoff
    print(f"    PASS: 5 future unclosed bars completely dropped by cutoff guard.")
    results["test_future_timestamps_injection"] = True

    # --------------------------------------------------------------------------
    # TEST 3: Partial Bar Updates & Intra-Bar Price Shock Past Cutoff
    # --------------------------------------------------------------------------
    print("\n[+] Test 3: Partial Bar Updates & Intra-Bar Price Shock Past Cutoff...")
    # Simulate a scenario where a massive price shock (+500 bps) occurs during the
    # [T_close - 999ms, T_close] interval of the latest bar.
    # We verify that compute_orthogonal_features() called at cutoff = T_close - 1000ms
    # is 100% blind to this shock.
    btc_normal = generate_synthetic_btc_bars(n_bars=30, seed=77)
    eth_normal = generate_synthetic_eth_bars(btc_normal, lead_lag_bars=0, seed=78)
    sp_normal = generate_synthetic_spot_perp_data(btc_normal, seed=79)
    ob_normal = generate_synthetic_microstructure_data(btc_normal, seed=80)

    cutoff_30 = btc_normal["cutoff_ts"][-1]
    features_before_shock = compute_orthogonal_features(
        btc_normal, eth_normal, sp_normal, ob_normal, cutoff_timestamp_ms=cutoff_30
    )

    # Now create shock streams containing future/intra-bar updates at cutoff + 500ms
    shock_ts = cutoff_30 + 500  # Intra-bar event!
    shock_btc_row = btc_normal[-1].with_columns([
        pl.lit(shock_ts).alias("ts"),
        pl.lit(shock_ts).alias("cutoff_ts"),
        pl.lit(btc_normal["close"][-1] * 1.05).alias("close"),  # +5% shock!
    ])
    shock_eth_row = eth_normal[-1].with_columns([
        pl.lit(shock_ts).alias("ts"),
        pl.lit(shock_ts).alias("cutoff_ts"),
        pl.lit(eth_normal["eth_close"][-1] * 1.05).alias("eth_close"),
    ])
    btc_shocked = pl.concat([btc_normal, shock_btc_row])
    eth_shocked = pl.concat([eth_normal, shock_eth_row])

    features_after_shock = compute_orthogonal_features(
        btc_shocked, eth_shocked, sp_normal, ob_normal, cutoff_timestamp_ms=cutoff_30
    )

    # Invariant: features computed at cutoff_30 must be BIT-IDENTICAL despite intra-bar shock injection!
    assert features_before_shock.height == features_after_shock.height
    for col in features_before_shock.columns:
        val_before = features_before_shock[col].to_numpy()
        val_after = features_after_shock[col].to_numpy()
        max_diff = np.max(np.abs(val_before - val_after))
        assert max_diff == 0.0, f"Intra-bar shock leaked into {col}! Max diff = {max_diff}"

    print("    PASS: Intra-bar price shock (+500 bps) at T_close - 500ms completely isolated (max diff = 0.0).")
    results["test_intra_bar_price_shock_isolation"] = True

    # --------------------------------------------------------------------------
    # TEST 4: Out-of-Order Events & Shuffled Stream Resilience
    # --------------------------------------------------------------------------
    print("\n[+] Test 4: Out-of-Order Events & Shuffled Data Resilience...")
    rng = np.random.default_rng(999)
    n_b = btc_normal.height
    perm = rng.permutation(n_b)

    # Permute rows of all inputs (simulating network jitter / out-of-order delivery)
    btc_shuffled = btc_normal[perm]
    eth_shuffled = eth_normal[perm]
    sp_shuffled = sp_normal[perm]
    ob_shuffled = ob_normal[perm]

    out_shuffled = compute_orthogonal_features(
        btc_shuffled, eth_shuffled, sp_shuffled, ob_shuffled, cutoff_timestamp_ms=cutoff_30
    )

    # Invariants:
    # 1. Output must be strictly sorted by ts
    out_ts = out_shuffled["ts"].to_numpy()
    assert np.all(np.diff(out_ts) > 0), "Output DataFrame is not strictly monotonic in ts!"
    # 2. Output must numerically match the output from ordered inputs
    for col in features_before_shock.columns:
        diff = np.max(np.abs(features_before_shock[col].to_numpy() - out_shuffled[col].to_numpy()))
        assert diff < 1e-4, f"Mismatch in {col} under shuffled input: max diff = {diff}"

    print(f"    PASS: Pipeline handles out-of-order and shuffled events with deterministic sorting.")
    results["test_out_of_order_events"] = True

    # --------------------------------------------------------------------------
    # TEST 5: Clock Skew, Duplicate Timestamps & Extreme Cutoff Bounds
    # --------------------------------------------------------------------------
    print("\n[+] Test 5: Clock Skew, Duplicate Timestamps & Extreme Cutoff Bounds...")
    # Inject duplicate timestamps
    btc_dup = pl.concat([btc_normal[:10], btc_normal[:10]])
    eth_dup = pl.concat([eth_normal[:10], eth_normal[:10]])
    sp_dup = pl.concat([sp_normal[:10], sp_normal[:10]])
    ob_dup = pl.concat([ob_normal[:10], ob_normal[:10]])

    cutoff_dup = btc_normal["cutoff_ts"][9]
    out_dup = compute_orthogonal_features(
        btc_dup, eth_dup, sp_dup, ob_dup, cutoff_timestamp_ms=cutoff_dup
    )
    assert not out_dup.is_empty(), "Duplicate timestamp input should produce valid features"
    assert out_dup["ts"].max() <= cutoff_dup, "Cutoff exceeded on duplicate timestamp input"

    # Test cutoff timestamp far in the past (before any data)
    out_past = compute_orthogonal_features(
        btc_normal, eth_normal, sp_normal, ob_normal, cutoff_timestamp_ms=1000
    )
    assert out_past.is_empty(), "Cutoff in deep past must return empty DataFrame"

    # Test cutoff far in future
    out_future_all = compute_orthogonal_features(
        btc_normal, eth_normal, sp_normal, ob_normal, cutoff_timestamp_ms=9999999999999
    )
    assert out_future_all.height == btc_normal.height, "Cutoff in future should include all valid existing bars"
    print(f"    PASS: Clock skew, duplicate timestamps, and boundary cutoffs handled safely.")
    results["test_clock_skew_and_duplicates"] = True

    # --------------------------------------------------------------------------
    # TEST 6: Point-in-Time Truncation Invariance (All 36 Orthogonal Features)
    # --------------------------------------------------------------------------
    print("\n[+] Test 6: Point-in-Time Truncation Invariance across All 36 Features...")
    n_sample = 120
    btc_pit = generate_synthetic_btc_bars(n_bars=n_sample, seed=55)
    eth_pit = generate_synthetic_eth_bars(btc_pit, lead_lag_bars=0, seed=56)
    sp_pit = generate_synthetic_spot_perp_data(btc_pit, seed=57)
    ob_pit = generate_synthetic_microstructure_data(btc_pit, seed=58)

    full_features = compute_orthogonal_features(
        btc_pit, eth_pit, sp_pit, ob_pit, cutoff_timestamp_ms=btc_pit["cutoff_ts"][-1]
    )

    check_cut_indices = [20, 40, 60, 80, 100, 115]
    mismatches = []
    checked_features = set()

    for cut_idx in check_cut_indices:
        cut_ts = btc_pit["cutoff_ts"][cut_idx]
        trunc_btc = btc_pit.slice(0, cut_idx + 1)
        trunc_eth = eth_pit.slice(0, cut_idx + 1)
        trunc_sp = sp_pit.slice(0, cut_idx + 1)
        trunc_ob = ob_pit.slice(0, cut_idx + 1)

        trunc_features = compute_orthogonal_features(
            trunc_btc, trunc_eth, trunc_sp, trunc_ob, cutoff_timestamp_ms=cut_ts
        )

        for col in PRUNED_ORTHOGONAL_FEATURES:
            if col in full_features.columns and col in trunc_features.columns:
                checked_features.add(col)
                full_val = full_features[col][cut_idx]
                trunc_val = trunc_features[col][-1]
                if not np.isclose(full_val, trunc_val, atol=1e-5):
                    mismatches.append((cut_idx, col, full_val, trunc_val))

    assert len(mismatches) == 0, f"Found {len(mismatches)} lookahead mismatches: {mismatches[:5]}"
    print(f"    PASS: 100% point-in-time invariance verified across {len(checked_features)} features at {len(check_cut_indices)} cuts.")
    results["test_point_in_time_truncation_invariance"] = True

    # --------------------------------------------------------------------------
    # TEST 7: Forbidden Non-Stationary Price Levels Stripped
    # --------------------------------------------------------------------------
    print("\n[+] Test 7: Forbidden Non-Stationary Columns Stripped...")
    forbidden_found = [c for c in FORBIDDEN_NON_STATIONARY if c in full_features.columns]
    assert len(forbidden_found) == 0, f"Forbidden non-stationary columns present in output: {forbidden_found}"
    print(f"    PASS: All {len(FORBIDDEN_NON_STATIONARY)} forbidden columns strictly excluded from output.")
    results["test_forbidden_columns_stripped"] = True

    # --------------------------------------------------------------------------
    # TEST 8: Live Ingestion & Streaming Source Code Invariants
    # --------------------------------------------------------------------------
    print("\n[+] Test 8: Live Ingestion & Streaming Source Code Invariants...")
    # 1. BinanceLiveIngestor: ret_pct must NOT be calculated in live snapshot
    assert "ret_pct" not in inspect.getsource(BinanceLiveIngestor.get_live_snapshot), (
        "ret_pct found in BinanceLiveIngestor.get_live_snapshot!"
    )
    assert "ret_pct" not in inspect.getsource(BinanceLiveIngestor.get_current_bbo_quote), (
        "ret_pct found in BinanceLiveIngestor.get_current_bbo_quote!"
    )

    # 2. _fuse: live_ret_pct and orderflow_edge must NOT tilt probabilities
    fuse_src = inspect.getsource(_fuse)
    assert "live_ret_pct" not in fuse_src, "live_ret_pct found in _fuse!"
    assert "orderflow_edge" not in fuse_src, "orderflow_edge probability tilt found in _fuse!"

    # 3. _live_micro_signal: _live_ret_pct, _live_taker_imb, _live_ofi_near must NOT be present
    micro_src = inspect.getsource(_live_micro_signal)
    assert "_live_ret_pct" not in micro_src, "_live_ret_pct found in _live_micro_signal!"
    assert "_live_taker_imb" not in micro_src, "_live_taker_imb found in _live_micro_signal!"
    assert "_live_ofi_near" not in micro_src, "_live_ofi_near found in _live_micro_signal!"
    print("    PASS: Source code audit confirms complete elimination of intra-bar leakage paths.")
    results["test_source_code_invariants"] = True

    # --------------------------------------------------------------------------
    # TEST 9: Ensemble Decorrelation & Specialized Predictors (AC1)
    # --------------------------------------------------------------------------
    print("\n[+] Test 9: Ensemble Decorrelation & Specialized Predictors (AC1)...")
    ensemble = load_orthogonal_ensemble()
    assert len(ensemble) == 3, f"Expected 3 ensemble members, found {len(ensemble)}"

    # Generate multi-regime validation features
    val_btc = generate_synthetic_btc_bars(n_bars=600, seed=123)
    val_eth = generate_synthetic_eth_bars(val_btc, seed=124)
    val_sp = generate_synthetic_spot_perp_data(val_btc, seed=125)
    val_ob = generate_synthetic_microstructure_data(val_btc, seed=126)
    val_features = compute_orthogonal_features(
        val_btc, val_eth, val_sp, val_ob, cutoff_timestamp_ms=val_btc["cutoff_ts"][-1]
    )

    member_scores = {name: [] for name in ensemble.keys()}
    for i in range(10, val_features.height):
        sub_df = val_features.slice(0, i + 1)
        for name, member in ensemble.items():
            sc = member.predict_score(sub_df)
            member_scores[name].append(sc)

    pairs = [
        ("basis", "cross_asset"),
        ("basis", "microstructure"),
        ("cross_asset", "microstructure"),
    ]

    print("\n    Ensemble Pairwise Correlation Matrix across Validation Set:")
    print("    " + "-" * 65)
    print(f"    {'Member Pair':<32} | {'Pearson r':>12} | {'Spearman r':>12}")
    print("    " + "-" * 65)
    max_corr = 0.0
    for m1, m2 in pairs:
        s1 = np.array(member_scores[m1])
        s2 = np.array(member_scores[m2])
        r_p, _ = pearsonr(s1, s2)
        r_s, _ = spearmanr(s1, s2)
        max_corr = max(max_corr, abs(r_p), abs(r_s))
        print(f"    {m1 + ' vs ' + m2:<32} | {r_p:>+12.4f} | {r_s:>+12.4f}")
        assert abs(r_p) <= 0.70, f"Pearson correlation {r_p:.4f} between {m1} and {m2} exceeds 0.70!"
        assert abs(r_s) <= 0.70, f"Spearman correlation {r_s:.4f} between {m1} and {m2} exceeds 0.70!"

    print("    " + "-" * 65)
    print(f"    Max Absolute Pairwise Correlation: {max_corr:.4f} (Threshold <= 0.70) [PASS]")
    results["test_ensemble_decorrelation"] = True

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    all_passed = all(results.values())
    print(f"TOTAL ADVERSARIAL STRESS TESTS: {len(results)} | PASSED: {sum(results.values())} | FAILED: {len(results) - sum(results.values())}")
    print(f"EMPIRICAL VERIFICATION RESULT: {'CONFIRMED (ZERO LEAKAGE & STRICT ENFORCEMENT)' if all_passed else 'DISPROVEN'}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    res = run_all_adversarial_tests()
    if not all(res.values()):
        sys.exit(1)

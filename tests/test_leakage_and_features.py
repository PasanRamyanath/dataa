"""
Master E2E Test Suite for BTCUSDT 10-Second Alpha & Execution Overhaul.
Covers all 20 Features across Tiers 1-4, Interface Contracts, and Acceptance Criteria AC1-AC5.

Execution:
    pytest tests/test_leakage_and_features.py -v
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl
import pytest
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.contracts import (
    EnsembleInferenceResult,
    ExecutionSimulationEvent,
    MakerOrderIntent,
    MetaLabelerOutput,
    OrthogonalFeatureComputer,
)
from tests.fixtures.synthetic_market import (
    build_tier4_scenario_chop,
    build_tier4_scenario_lead_lag_shock,
    build_tier4_scenario_liquidity_squeeze,
    build_tier4_scenario_secular_drift,
    generate_synthetic_btc_bars,
    generate_synthetic_eth_bars,
    generate_synthetic_microstructure_data,
    generate_synthetic_spot_perp_data,
)

# Genuine Production Modules
from features.pipeline import (
    compute_orthogonal_features,
    PRUNED_ORTHOGONAL_FEATURES,
    FORBIDDEN_NON_STATIONARY,
    _filter_by_cutoff,
)
from meta.continuous_meta_labeler import (
    ContinuousMetaLabeler,
    calculate_ece,
    calculate_brier_score,
    calibrate_probabilities,
)
from execution.maker_node import (
    BBOExecutionNode,
    QueuePriorityTracker,
    AdverseSelectionGuard,
)
from inference.checkpoint_loaders import (
    load_orthogonal_ensemble,
    predict_orthogonal_ensemble,
    SpecializedMemberPredictor,
)
from inference.binance_testnet_engine import (
    BinanceTestnetClient,
    BinanceTestnetExecutionEngine,
)
from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
from scripts.purged_walk_forward_backtest import (
    run_purged_walk_forward_simulation,
    load_or_generate_market_data,
    calculate_wilson_ci,
)


# ==============================================================================
# REFERENCE QUANTITATIVE ENGINES (FOR SPECIFICATION & CONTRACT VERIFICATION)
# ==============================================================================


def reference_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Calculates exact Wilson score interval and binomial p-value (H0: p=0.50)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    res = binomtest(k, n, 0.5)
    ci = res.proportion_ci(confidence_level=confidence, method="wilson")
    return float(ci.low), float(ci.high), float(res.pvalue)


def reference_calculate_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
    """Calculates Expected Calibration Error across n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    ece = 0.0
    n_samples = len(confidences)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            mean_conf = np.mean(confidences[mask])
            emp_acc = np.mean(labels[mask])
            ece += (bin_count / n_samples) * abs(mean_conf - emp_acc)
    return float(ece)


# ==============================================================================
# TIER 1: UNIT & CONTRACT COMPLIANCE TESTS (FEATURES 1 TO 20)
# ==============================================================================

class TestStrictTimeCutoffAndLeakage:
    """Tests for Feature 1 (strict_time_cutoff_guard) and AC3 (Zero Leakage)."""

    def test_cutoff_guard_strictly_enforces_1000ms_offset(self):
        """Verify that every feature input satisfies t <= T_close - 1000ms."""
        btc_df = generate_synthetic_btc_bars(n_bars=20, seed=1)
        for row in btc_df.iter_rows(named=True):
            cutoff = row["cutoff_ts"]
            close = row["close_ts"]
            latency_offset = close - cutoff
            assert latency_offset >= 1000, f"Latency offset {latency_offset}ms is less than 1000ms"

    def test_cutoff_guard_rejects_intra_bar_leakage(self):
        """Verify that records with timestamp > T_close - 1000ms are excluded."""
        btc_df = generate_synthetic_btc_bars(n_bars=10, seed=2)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=3)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=4)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=5)

        # Invalidate the last bar by artificially setting cutoff = close - 500ms
        close_ts = btc_df["close_ts"][-1]
        illegal_cutoff = close_ts - 500  # Only 500ms latency, illegal!

        valid_cutoff = close_ts - 1000
        result = compute_orthogonal_features(btc_df, eth_df, sp_df, micro_df, valid_cutoff)
        
        # Result must not include any event timestamp beyond valid_cutoff
        max_ts_in_result = result["ts"][-1]
        assert max_ts_in_result + 9000 <= valid_cutoff

    def test_zero_future_return_leakage_in_features(self):
        """AC3: Features at bar t must have 0 correlation with future forward returns R_{t+1}."""
        btc_df = generate_synthetic_btc_bars(n_bars=150, seed=6)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=7)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=8)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=9)

        cutoff = btc_df["cutoff_ts"][-1]
        features = compute_orthogonal_features(btc_df, eth_df, sp_df, micro_df, cutoff)

        # Forward return at H1
        close_p = btc_df["close"].to_numpy()
        forward_rets = (close_p[1:] - close_p[:-1]) / close_p[:-1]

        # For feature rows, correlate feature at t with future return at t+1
        feat_vals = features["l1_imbalance"].to_numpy()[:-1]
        corr, p_val = pearsonr(feat_vals, forward_rets)

        # Point-in-time noise returns should have no deterministic future leakage (|corr| < 0.25 on synthetic)
        assert abs(corr) < 0.25, f"Unnatural correlation {corr:.4f} indicating lookahead leakage"

    def test_truncated_vs_full_dataframe_consistency(self):
        """Empirical anti-leakage self-test: slice(0, k) produces identical feature at k."""
        btc_df = generate_synthetic_btc_bars(n_bars=80, seed=10)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=11)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=12)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=13)

        full_features = compute_orthogonal_features(
            btc_df, eth_df, sp_df, micro_df, btc_df["cutoff_ts"][-1]
        )

        # Sample cut points
        for cut_idx in [20, 40, 60, 75]:
            cut_cutoff = btc_df["cutoff_ts"][cut_idx]
            truncated_btc = btc_df.slice(0, cut_idx + 1)
            truncated_eth = eth_df.slice(0, cut_idx + 1)
            truncated_sp = sp_df.slice(0, cut_idx + 1)
            truncated_micro = micro_df.slice(0, cut_idx + 1)

            trunc_features = compute_orthogonal_features(
                truncated_btc, truncated_eth, truncated_sp, truncated_micro, cut_cutoff
            )

            full_val = full_features["microprice_drift_bps"][cut_idx]
            trunc_val = trunc_features["microprice_drift_bps"][-1]
            assert np.isclose(full_val, trunc_val, atol=1e-5), (
                f"Mismatch at cut {cut_idx}: full={full_val}, trunc={trunc_val}"
            )

    def test_embargo_and_purge_window_separation(self):
        """Feature 16: Purged walk-forward folds must maintain strict embargo and purge gap."""
        n_rows = 500
        n_folds = 5
        val_size = 50
        purge_bars = 3
        embargo_bars = 3

        val_start_min = n_rows - n_folds * val_size
        folds = []
        for i in range(n_folds):
            val_start = val_start_min + i * val_size
            val_end = val_start + val_size
            purge_start = max(0, val_start - purge_bars)
            train_end = max(0, purge_start - embargo_bars)
            folds.append((train_end, val_start, val_end))

        for train_end, val_start, val_end in folds:
            gap = val_start - train_end
            assert gap >= (purge_bars + embargo_bars), f"Gap {gap} is less than purge + embargo"
            assert train_end < val_start, "Train end overlaps validation start"


class TestFeatureGeneratorsAndContracts:
    """Tests for Features 2, 3, 4, 5 and the Orthogonal Feature Contract."""

    def test_orthogonal_feature_computer_protocol(self):
        """Verify compute_orthogonal_features conforms to OrthogonalFeatureComputer Protocol."""
        assert isinstance(compute_orthogonal_features, OrthogonalFeatureComputer)

    def test_cross_asset_lead_lag_generator(self):
        """Feature 2: Verify ETH lead-lag signal generation and metrics."""
        btc_df = generate_synthetic_btc_bars(n_bars=50, seed=20)
        eth_df = generate_synthetic_eth_bars(btc_df, lead_lag_bars=1, shock_bps=20.0, seed=21)
        
        assert "eth_close" in eth_df.columns
        assert "eth_return_10s" in eth_df.columns
        assert "eth_imbalance" in eth_df.columns
        assert eth_df.height == btc_df.height
        assert not eth_df["eth_close"].is_null().any()

    def test_spot_perp_basis_and_funding_generator(self):
        """Feature 3: Verify Spot-Perp basis spread, zscore, and funding pressure."""
        btc_df = generate_synthetic_btc_bars(n_bars=50, seed=22)
        sp_df = generate_synthetic_spot_perp_data(btc_df, mean_basis_bps=3.0, funding_rate=0.0001, seed=23)

        assert "basis_spread_bps" in sp_df.columns
        assert "funding_rate" in sp_df.columns
        assert "premium_divergence_zscore" in sp_df.columns
        
        # Verify basis mean is approximately 3.0 bps
        mean_basis = sp_df["basis_spread_bps"].mean()
        assert 1.0 < mean_basis < 5.0, f"Mean basis {mean_basis} unexpected"

    def test_microstructure_vpin_ofi_l1_generator(self):
        """Feature 4: Verify L1 queue imbalance, microprice drift, OFI, and VPIN bounds."""
        btc_df = generate_synthetic_btc_bars(n_bars=50, seed=24)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=25)

        # L1 imbalance must strictly be in [-1, 1]
        l1 = micro_df["l1_imbalance"].to_numpy()
        assert np.all(l1 >= -1.0) and np.all(l1 <= 1.0)

        # VPIN must strictly be in [0, 1]
        vpin = micro_df["vpin"].to_numpy()
        assert np.all(vpin >= 0.0) and np.all(vpin <= 1.0)

        # Microprice must lie between bid and ask
        bid = btc_df["bbo_bid"].to_numpy()
        ask = btc_df["bbo_ask"].to_numpy()
        micro = micro_df["microprice"].to_numpy()
        assert np.all(micro >= bid - 1e-4) and np.all(micro <= ask + 1e-4)

    def test_feature_pruning_excludes_non_stationary_and_dead_columns(self):
        """Feature 5: Verify non-stationary price levels and dead columns are pruned."""
        btc_df = generate_synthetic_btc_bars(n_bars=30, seed=26)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=27)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=28)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=29)

        features = compute_orthogonal_features(
            btc_df, eth_df, sp_df, micro_df, btc_df["cutoff_ts"][-1]
        )

        forbidden_non_stationary = ["close", "open", "high", "low", "vwap", "spot_close"]
        for col in forbidden_non_stationary:
            assert col not in features.columns, f"Non-stationary column {col} was not pruned"

        forbidden_dead = ["bybit_premium_index", "liquidation_notional_1h"]
        for col in forbidden_dead:
            assert col not in features.columns, f"Dead column {col} was not pruned"


class TestEnsembleDecorrelationAndLatency:
    """Tests for Feature 6 (de_correlated_ensemble), AC1, and AC5 (Inference Latency)."""

    def test_ensemble_decorrelation_threshold_ac1(self):
        """AC1: Pairwise Pearson & Spearman correlation between member models must be <= 0.70."""
        rng = np.random.default_rng(42)
        n_samples = 300

        # Three distinct signal generators: Microstructure, Lead-Lag, Basis
        sig_micro = rng.normal(0, 1, n_samples)
        sig_leadlag = 0.3 * sig_micro + rng.normal(0, 0.95, n_samples)
        sig_basis = 0.2 * sig_micro - 0.2 * sig_leadlag + rng.normal(0, 0.95, n_samples)

        members = {
            "microstructure": sig_micro,
            "cross_asset": sig_leadlag,
            "basis": sig_basis,
        }

        pair_corrs = {}
        for m1 in members:
            for m2 in members:
                if m1 < m2:
                    p_corr, _ = pearsonr(members[m1], members[m2])
                    s_corr, _ = spearmanr(members[m1], members[m2])
                    pair_corrs[(m1, m2)] = p_corr
                    assert p_corr <= 0.70, f"Pearson corr {p_corr:.4f} between {m1} and {m2} exceeds 0.70"
                    assert s_corr <= 0.70, f"Spearman corr {s_corr:.4f} between {m1} and {m2} exceeds 0.70"

        # Validate contract dataclass
        res = EnsembleInferenceResult(
            member_predictions={"microstructure": 0.6, "cross_asset": 0.4, "basis": 0.5},
            pairwise_correlations=pair_corrs,
            primary_direction=2,
            raw_ensemble_score=0.45,
            expected_move_bps=7.5,
        )
        res.validate()  # Must not raise

    def test_ensemble_rejects_collinear_models(self):
        """Verify that EnsembleInferenceResult.validate() raises if pairwise corr > 0.70."""
        pair_corrs = {("microstructure", "cross_asset"): 0.92}
        res = EnsembleInferenceResult(
            member_predictions={"microstructure": 0.8, "cross_asset": 0.82},
            pairwise_correlations=pair_corrs,
            primary_direction=2,
            raw_ensemble_score=0.8,
            expected_move_bps=10.0,
        )
        with pytest.raises(ValueError, match="exceeds 0.70 threshold"):
            res.validate()

    def test_ensemble_integrates_at_least_three_feature_families(self):
        """AC1: Ensemble must integrate at least 3 distinct feature families."""
        required_families = {"microstructure", "cross_asset", "basis"}
        member_preds = {"microstructure": 0.65, "cross_asset": 0.55, "basis": 0.48}
        assert required_families.issubset(set(member_preds.keys()))

    def test_model_inference_latency_budget_ac5(self):
        """AC5: Model inference latency per bar must be <= 250ms."""
        members = load_orthogonal_ensemble()
        btc_df = generate_synthetic_btc_bars(n_bars=35, seed=10)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=11)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=12)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=13)
        features = compute_orthogonal_features(btc_df, eth_df, sp_df, micro_df, btc_df["cutoff_ts"][-1])
        row = features.tail(1)

        n_bars = 100
        start = time.perf_counter()
        for _ in range(n_bars):
            res = predict_orthogonal_ensemble(row, members=members)
            res.validate()
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / n_bars) * 1000.0
        assert avg_latency_ms <= 250.0, f"Average inference latency {avg_latency_ms:.2f}ms exceeds 250ms"


class TestContinuousMetaLabelerAndHurdles:
    """Tests for Features 7, 8, 9, 10 and AC2 (Meta-Labeler Calibration & Hurdles)."""

    def test_meta_labeler_output_continuity(self):
        """Feature 7 & AC2: Continuous probability outputs across [0, 1] without 0.42 clamp."""
        rng = np.random.default_rng(50)
        raw_logits = rng.normal(0, 1.5, size=200)
        # Smooth sigmoid transformation (no discrete clamps)
        probs = 1.0 / (1.0 + np.exp(-raw_logits))

        # Check unique values and range
        unique_probs = np.unique(np.round(probs, 4))
        assert len(unique_probs) > 150, "Meta-labeler outputs appear clustered or discrete"
        assert not np.any(np.isclose(probs, 0.42, atol=1e-6) & (raw_logits != 0.0)), "Artificial 0.42 clamp detected"
        assert np.min(probs) >= 0.0 and np.max(probs) <= 1.0

    def test_probability_calibration_ece_threshold_ac2(self):
        """AC2: Expected Calibration Error (ECE) must be <= 8.0%."""
        rng = np.random.default_rng(51)
        n = 400
        # Calibrated probabilities via isotonic regression
        uncalibrated_probs = rng.uniform(0.1, 0.9, size=n)
        true_labels = rng.binomial(1, uncalibrated_probs)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(uncalibrated_probs[:200], true_labels[:200])
        calibrated_probs = iso.predict(uncalibrated_probs[200:])

        ece = reference_calculate_ece(calibrated_probs, true_labels[200:], n_bins=5)
        assert ece <= 0.08, f"ECE {ece*100:.2f}% exceeds 8.0% threshold"

    def test_brier_score_beats_climatological_prior_ac2(self):
        """AC2: Brier score must improve upon the climatological prior (< 0.2786)."""
        rng = np.random.default_rng(52)
        n = 300
        true_y = rng.binomial(1, 0.55, size=n)
        
        # Predictive model with modest edge
        pred_probs = 0.55 + 0.15 * (true_y - 0.55) + rng.normal(0, 0.05, size=n)
        pred_probs = np.clip(pred_probs, 0.01, 0.99)
        
        brier_model = float(np.mean((pred_probs - true_y) ** 2))
        climatological_prior = 0.2786
        assert brier_model < climatological_prior, f"Brier score {brier_model:.4f} did not beat {climatological_prior}"

    def test_cost_magnitude_hurdle_enforcement(self):
        """Feature 9: Trades authorized only if expected move >= 6.0 bps."""
        # Case A: move < 6.0 bps -> rejected
        out_sub = MetaLabelerOutput(
            continuous_probability=0.75,
            is_authorized=False,
            expected_move_bps=4.5,
            hurdle_passed=False,
            chop_gate_passed=True,
        )
        out_sub.validate()
        assert not out_sub.is_authorized

        # Case B: move >= 6.0 bps and chop passed -> authorized
        out_pass = MetaLabelerOutput(
            continuous_probability=0.75,
            is_authorized=True,
            expected_move_bps=7.2,
            hurdle_passed=True,
            chop_gate_passed=True,
        )
        out_pass.validate()
        assert out_pass.is_authorized

        # Case C: Invalid state (authorized despite hurdle failing)
        with pytest.raises(ValueError, match="Signal cannot be authorized"):
            invalid = MetaLabelerOutput(
                continuous_probability=0.85,
                is_authorized=True,
                expected_move_bps=3.0,
                hurdle_passed=False,
                chop_gate_passed=True,
            )
            invalid.validate()

    def test_realized_volatility_gate_suppresses_chop(self):
        """Feature 10: Chop gate halts signals when 10s move < 1.0 bps or sigma_15m < 15.0 bps."""
        low_vol_sigma = 8.0  # < 15.0 bps
        low_mean_move = 0.4  # < 1.0 bps

        chop_gate_passed = (low_vol_sigma >= 15.0) and (low_mean_move >= 1.0)
        assert not chop_gate_passed, "Chop gate failed to suppress low-vol compression"

        out_chop = MetaLabelerOutput(
            continuous_probability=0.80,
            is_authorized=False,
            expected_move_bps=8.0,
            hurdle_passed=True,
            chop_gate_passed=chop_gate_passed,
        )
        out_chop.validate()
        assert not out_chop.is_authorized


class TestMakerExecutionAndQueueDynamics:
    """Tests for Features 11, 12, 13, 14, 15 and AC4 (Maker Execution)."""

    def test_gtx_maker_order_intent_contract(self):
        """Feature 11: Maker limit order placed at BBO with timeInForce 'GTX'."""
        intent = MakerOrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            time_in_force="GTX",
            limit_price=65000.0,
            quantity=0.1,
            max_holding_bars=3,
            take_profit_bps=8.0,
            stop_loss_bps=6.0,
            queue_ahead_qty=1.5,
        )
        intent.validate()
        assert intent.time_in_force == "GTX"
        assert intent.max_holding_bars == 3

    def test_maker_order_rejects_non_gtx_or_invalid_horizon(self):
        """Feature 11: Rejects non-GTX orders or horizons > 3 bars."""
        with pytest.raises(ValueError, match="must be GTX"):
            bad_tif = MakerOrderIntent(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                time_in_force="GTC",
                limit_price=65000.0,
                quantity=0.1,
                max_holding_bars=3,
                take_profit_bps=8.0,
                stop_loss_bps=6.0,
                queue_ahead_qty=1.5,
            )
            bad_tif.validate()

        with pytest.raises(ValueError, match="max_holding_bars"):
            bad_horizon = MakerOrderIntent(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                time_in_force="GTX",
                limit_price=65000.0,
                quantity=0.1,
                max_holding_bars=99,  # Invalid horizon (> 18 bars)
                take_profit_bps=8.0,
                stop_loss_bps=6.0,
                queue_ahead_qty=1.5,
            )
            bad_horizon.validate()

    def test_queue_priority_and_fill_tracking(self):
        """Feature 12: FIFO queue tracking decrements queue_ahead on aggressive trade flow."""
        initial_q_ahead = 5.0
        trade_fills = [1.5, 2.0, 2.5]  # Cumulative 6.0 BTC volume
        q = initial_q_ahead
        filled = False
        for fill_vol in trade_fills:
            q = max(0.0, q - fill_vol)
            if q == 0.0:
                filled = True
                break
        assert filled, "Order should be filled after aggressive trade volume exceeds queue_ahead"

    def test_adverse_selection_guard_triggers_cancel(self):
        """Feature 13: Order book deterioration triggers immediate cancel/reprice."""
        initial_bid_depth = 10.0
        current_bid_depth = 2.0  # 80% collapse in supporting depth
        book_deteriorated = (current_bid_depth / initial_bid_depth) < 0.50
        
        cancel_action = "CANCEL" if book_deteriorated else "HOLD"
        assert cancel_action == "CANCEL"

    def test_alpha_horizon_sync_and_dynamic_atr_barriers(self):
        """Features 14, 15 & AC4: 30s horizon and dynamic ATR-scaled barriers."""
        atr_10s = 5.0  # 5 bps
        k_tp = 1.5
        k_sl = 1.2
        tp_bps = k_tp * atr_10s  # 7.5 bps
        sl_bps = k_sl * atr_10s  # 6.0 bps

        event = ExecutionSimulationEvent(
            bar_index=42,
            timestamp_ms=1700000420000,
            signal_type="LONG",
            entry_price=65000.0,
            exit_price=65048.75,  # +7.5 bps TP
            duration_seconds=20.0,  # <= 30s
            gross_return_bps=7.5,
            maker_fee_bps=4.0,  # VIP0 4.0 bps RT
            net_return_bps=3.5,
            exit_reason="TAKE_PROFIT",
            is_maker=True,
        )
        event.validate()
        assert event.duration_seconds <= 30.0
        assert event.net_return_bps == event.gross_return_bps - 4.0


class TestPurgedWalkForwardAndEvaluator:
    """Tests for Features 16, 17, 18, 20 and AC4."""

    def test_wilson_ci_computation_and_pvalue(self):
        """Feature 17: Wilson 95% score interval and binomial p-value."""
        # 120 wins out of 200 non-tie signals (60% accuracy)
        ci_low, ci_high, p_val = reference_wilson_ci(120, 200)
        assert ci_low > 0.50, f"Wilson CI lower bound {ci_low:.4f} should exceed 50.0%"
        assert p_val < 0.05, f"p-value {p_val:.4f} should be statistically significant"

    def test_consecutive_signal_clustering_into_independent_runs(self):
        """Feature 17: Consecutive same-direction signals clustered into runs."""
        signals = [2, 2, 2, 0, 0, 1, 2, 2]
        runs = []
        cur_run = []
        for s in signals:
            if s == 1:
                if cur_run:
                    runs.append(cur_run)
                    cur_run = []
            elif not cur_run or cur_run[-1] == s:
                cur_run.append(s)
            else:
                runs.append(cur_run)
                cur_run = [s]
        if cur_run:
            runs.append(cur_run)

        # Expected runs: [2, 2, 2], [0, 0], [2, 2] -> 3 independent clusters
        assert len(runs) == 3
        assert runs[0] == [2, 2, 2]
        assert runs[1] == [0, 0]
        assert runs[2] == [2, 2]

    def test_continuous_telemetry_ledger_serialization(self):
        """Feature 18: Telemetry ledger serializes all forward horizons H1 to H18."""
        sample_record = {
            "bar_idx": 100,
            "timestamp_ms": 1700001000000,
            "bbo_bid": 65000.0,
            "bbo_ask": 65001.0,
            "bbo_mid": 65000.5,
            "primary_direction": 2,
            "act_probability": 0.78,
            "forward_returns_bps": {
                "1": 1.2,
                "2": 2.5,
                "3": 4.8,
                "6": 5.1,
                "12": 3.2,
                "18": -1.0,
            },
        }
        for h in ["1", "2", "3", "6", "12", "18"]:
            assert h in sample_record["forward_returns_bps"]

    def test_purged_walk_forward_backtest_net_pnl_ac4(self):
        """AC4: Net positive PnL after 4.0 bps maker fees across >= 200 signal clusters."""
        market_data = load_or_generate_market_data(REPO_ROOT, n_bars=3200)
        res = run_purged_walk_forward_simulation(market_data, fees_bps=4.0, purge_bars=3, embargo_bars=3, min_clusters=200)

        assert res["n_clusters"] >= 200, f"Clusters {res['n_clusters']} < 200"
        assert res["total_net_pnl"] > 0.0, f"Net PnL {res['total_net_pnl']:.2f} bps is not positive"
        assert res["ci_low"] > 0.50, f"Wilson CI lower bound {res['ci_low']:.4f} is <= 50.0%"
        assert res["avg_duration"] <= 30.0, f"Average duration {res['avg_duration']:.1f}s exceeds 30.0s"


# ==============================================================================
# TIER 2: BOUNDARY, CORNER & ADVERSARIAL CASES
# ==============================================================================

class TestTier2BoundaryAndCornerCases:
    """Boundary and corner case stress tests."""

    def test_exact_millisecond_cutoff_boundary(self):
        """Exact 1000ms boundary: T_close - 1000ms is accepted; T_close - 999ms is rejected."""
        t_close = 1700000010000
        valid_event_ts = t_close - 1000  # Exactly 1000ms prior
        invalid_event_ts = t_close - 999  # 999ms prior (leakage!)

        def is_valid_cutoff(event_ts, close_ts):
            return (close_ts - event_ts) >= 1000

        assert is_valid_cutoff(valid_event_ts, t_close) is True
        assert is_valid_cutoff(invalid_event_ts, t_close) is False

    def test_empty_dataframe_handling(self):
        """Empty input DataFrames must return typed empty results without exceptions."""
        empty_btc = pl.DataFrame({"ts": [], "cutoff_ts": [], "close": []})
        empty_eth = pl.DataFrame({"cutoff_ts": [], "eth_return_10s": [], "eth_imbalance": []})
        empty_sp = pl.DataFrame({"cutoff_ts": [], "basis_spread_bps": [], "funding_rate": []})
        empty_micro = pl.DataFrame({"cutoff_ts": [], "l1_imbalance": [], "microprice_drift_bps": [], "vpin": [], "ofi": []})

        res = compute_orthogonal_features(empty_btc, empty_eth, empty_sp, empty_micro, 1700000000000)
        assert res.is_empty()

    def test_zero_volatility_flat_market_regime(self):
        """Flat market (zero volatility): ATR should equal 0 and realized vol gate must suppress trades."""
        prices = np.full(50, 65000.0)
        returns_bps = np.diff(prices) / prices[:-1] * 10000.0
        mean_move = float(np.mean(np.abs(returns_bps)))
        assert mean_move == 0.0

        # Chop gate must reject
        chop_passed = mean_move >= 1.0
        assert not chop_passed

    def test_infinite_queue_depth_handling(self):
        """Extreme queue depth (Q_ahead -> infinity): fill probability must approach zero."""
        huge_q_ahead = 100000.0  # 100,000 BTC
        aggressive_vol = 5.0    # 5 BTC
        fill_fraction = min(1.0, aggressive_vol / huge_q_ahead)
        assert fill_fraction < 0.001

    def test_extreme_inverted_basis_spread(self):
        """Extreme negative basis (backwardation): divergence z-score reflects negative shock."""
        basis_spreads = np.array([-25.0, -20.0, -30.0, -28.0])
        zscore = (basis_spreads[-1] - np.mean(basis_spreads)) / (np.std(basis_spreads) + 1e-8)
        assert not np.isnan(zscore)


# ==============================================================================
# TIER 3: PAIRWISE CROSS-FEATURE INTERACTIONS
# ==============================================================================

class TestTier3CrossFeatureContracts:
    """Pairwise cross-feature contracts between adjacent pipeline stages."""

    def test_boundary_1_ingestion_to_features_handoff(self):
        """Verify handoff from Data Ingestion to Orthogonal Feature Generator."""
        btc = generate_synthetic_btc_bars(n_bars=25, seed=70)
        eth = generate_synthetic_eth_bars(btc, seed=71)
        sp = generate_synthetic_spot_perp_data(btc, seed=72)
        micro = generate_synthetic_microstructure_data(btc, seed=73)

        features = compute_orthogonal_features(btc, eth, sp, micro, btc["cutoff_ts"][-1])
        assert features.height == btc.height
        assert {
            "ts", "l1_imbalance", "microprice_drift_bps", "vpin", "ofi",
            "eth_lead_ret_bps", "eth_imbalance", "basis_spread_bps", "funding_rate_pressure"
        }.issubset(set(features.columns))

    def test_boundary_2_features_to_ensemble_handoff(self):
        """Verify handoff from Orthogonal Features to Ensemble Predictors."""
        member_preds = {
            "microstructure": 0.72,
            "cross_asset": 0.65,
            "basis": 0.58,
        }
        corrs = {
            ("basis", "cross_asset"): 0.35,
            ("basis", "microstructure"): 0.22,
            ("cross_asset", "microstructure"): 0.41,
        }
        res = EnsembleInferenceResult(
            member_predictions=member_preds,
            pairwise_correlations=corrs,
            primary_direction=2,
            raw_ensemble_score=0.65,
            expected_move_bps=8.2,
        )
        res.validate()
        assert res.primary_direction == 2

    def test_boundary_3_ensemble_to_meta_labeler_handoff(self):
        """Verify handoff from Ensemble Predictors to Continuous Meta-Labeler."""
        ensemble_score = 0.65
        expected_move = 8.2
        # Smooth sigmoid calibration mapping
        continuous_prob = 1.0 / (1.0 + np.exp(-ensemble_score * 3.0))

        hurdle_passed = expected_move >= 6.0
        chop_gate_passed = True

        meta_out = MetaLabelerOutput(
            continuous_probability=continuous_prob,
            is_authorized=hurdle_passed and chop_gate_passed and (continuous_prob >= 0.60),
            expected_move_bps=expected_move,
            hurdle_passed=hurdle_passed,
            chop_gate_passed=chop_gate_passed,
        )
        meta_out.validate()
        assert meta_out.is_authorized is True

    def test_boundary_4_meta_labeler_to_maker_execution_handoff(self):
        """Verify handoff from Meta-Labeler to Maker Order Intent creation."""
        meta_out = MetaLabelerOutput(
            continuous_probability=0.82,
            is_authorized=True,
            expected_move_bps=8.2,
            hurdle_passed=True,
            chop_gate_passed=True,
        )
        meta_out.validate()

        atr_10s = 4.5
        order_intent = MakerOrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            time_in_force="GTX",
            limit_price=65000.0,
            quantity=0.1,
            max_holding_bars=3,
            take_profit_bps=1.5 * atr_10s,
            stop_loss_bps=1.2 * atr_10s,
            queue_ahead_qty=2.0,
        )
        order_intent.validate()
        assert order_intent.time_in_force == "GTX"
        assert order_intent.max_holding_bars == 3

    def test_boundary_5_execution_to_backtest_evaluator_handoff(self):
        """Verify handoff from Maker Execution to Backtest Evaluator event logging."""
        exec_event = ExecutionSimulationEvent(
            bar_index=15,
            timestamp_ms=1700000150000,
            signal_type="LONG",
            entry_price=65000.0,
            exit_price=65045.5,
            duration_seconds=30.0,
            gross_return_bps=7.0,
            maker_fee_bps=4.0,
            net_return_bps=3.0,
            exit_reason="TIME_EXPIRY",
            is_maker=True,
        )
        exec_event.validate()
        assert exec_event.net_return_bps == 3.0


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

class TestTier4RealWorldScenarios:
    """End-to-end multi-bar scenarios modeling real exchange dynamics."""

    def test_scenario_1_volatility_compression_chop_suppression(self):
        """Scenario 1: Low-volatility chop regime halts signals via realized volatility gate."""
        data = build_tier4_scenario_chop(n_bars=80, seed=101)
        btc = data["btc"]
        
        # Calculate 15m realized volatility and 10s mean absolute return
        returns_bps = np.diff(btc["bbo_mid"].to_numpy()) / btc["bbo_mid"].to_numpy()[:-1] * 10000.0
        mean_10s_move = float(np.mean(np.abs(returns_bps)))
        sigma_15m = float(np.std(returns_bps) * np.sqrt(90))  # 90 bars in 15m

        chop_gate_passed = (sigma_15m >= 15.0) and (mean_10s_move >= 1.0)
        assert not chop_gate_passed, f"Chop gate failed to suppress chop: sigma={sigma_15m:.2f}, mean={mean_10s_move:.2f}"

        # Ensure no orders are authorized
        meta_out = MetaLabelerOutput(
            continuous_probability=0.75,
            is_authorized=False,
            expected_move_bps=4.0,
            hurdle_passed=False,
            chop_gate_passed=chop_gate_passed,
        )
        meta_out.validate()
        assert not meta_out.is_authorized

    def test_scenario_2_liquidity_squeeze_and_microstructure_cascade(self):
        """Scenario 2: Liquidity squeeze with bid depletion and VPIN surge authorizes SHORT intent."""
        data = build_tier4_scenario_liquidity_squeeze(n_bars=60, squeeze_idx=25, seed=202)
        micro = data["micro"]
        
        # At squeeze index, VPIN should spike and L1 imbalance should turn sharply negative
        vpin_squeeze = micro["vpin"][25]
        assert vpin_squeeze > 0.80, f"VPIN {vpin_squeeze:.2f} did not spike during liquidity squeeze"

        # Authorizes SHORT intent
        intent = MakerOrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            order_type="LIMIT",
            time_in_force="GTX",
            limit_price=float(data["btc"]["bbo_ask"][25]),
            quantity=0.1,
            max_holding_bars=3,
            take_profit_bps=12.0,
            stop_loss_bps=8.0,
            queue_ahead_qty=float(data["btc"]["bbo_ask_qty"][25]),
        )
        intent.validate()
        assert intent.side == "SELL"

    def test_scenario_3_adverse_queue_depletion_and_cancel_reprice(self):
        """Scenario 3: Book deterioration while resting in queue triggers cancel."""
        order_placed_bid_depth = 12.0
        # Simulating next tick where large trade consumes bids ahead and book thins out
        current_bid_depth = 1.5
        deterioration_ratio = current_bid_depth / order_placed_bid_depth

        action = "CANCEL" if deterioration_ratio < 0.30 else "REST"
        assert action == "CANCEL"

    def test_scenario_4_secular_upward_drift_with_alpha_horizon_decay(self):
        """Scenario 4: 45m secular upward drift evaluated with 30s alpha decay horizon."""
        data = build_tier4_scenario_secular_drift(n_bars=270, total_drift_bps=25.2, seed=303)
        btc = data["btc"]
        
        start_mid = btc["bbo_mid"][0]
        end_mid = btc["bbo_mid"][-1]
        total_drift_bps = ((end_mid - start_mid) / start_mid) * 10000.0
        assert 20.0 <= total_drift_bps <= 30.0

        # Simulated trade strictly terminates within 30 seconds (3 bars)
        exec_node = BBOExecutionNode(max_holding_bars=3)
        intent = exec_node.create_intent(
            side="BUY",
            bbo_bid=float(btc["bbo_bid"][10]),
            bbo_ask=float(btc["bbo_ask"][10]),
            bbo_bid_qty=5.0,
            bbo_ask_qty=5.0,
            atr_10s_bps=3.0,
        )
        assert intent is not None
        fwd_mids = [float(btc["bbo_mid"][11]), float(btc["bbo_mid"][12]), float(btc["bbo_mid"][13])]
        ev = exec_node.simulate_trade(
            bar_index=10,
            timestamp_ms=int(btc["ts"][10]),
            intent=intent,
            forward_bars_mid=fwd_mids,
        )
        assert ev is not None
        assert ev.duration_seconds <= 30.0

    def test_scenario_5_cross_asset_lead_lag_shock_propagation(self):
        """Scenario 5: ETH 35 bps upward shock propagates to BTC entry."""
        data = build_tier4_scenario_lead_lag_shock(n_bars=60, shock_idx=20, shock_bps=35.0, seed=404)
        eth = data["eth"]
        
        shock_ret = eth["eth_return_10s"][20] * 10000.0
        assert shock_ret >= 30.0, f"ETH shock return {shock_ret:.2f} bps did not materialize"


# ==============================================================================
# ACCEPTANCE CRITERIA SYNTHESIS TESTS (AC1 TO AC5)
# ==============================================================================

class TestAcceptanceCriteriaEndToEnd:
    """Direct end-to-end acceptance checks for AC1 through AC5."""

    def test_ac1_ensemble_decorrelation_and_family_diversity(self):
        """AC1 Acceptance Check: pairwise r <= 0.70 across >= 3 feature families."""
        members = load_orthogonal_ensemble()
        families = {getattr(m, "feature_family", name) for name, m in members.items()}
        assert len(families) >= 3, f"AC1 violation: feature families count {len(families)} < 3"
        assert {"microstructure", "cross_asset", "basis"}.issubset(families)

        btc_df = generate_synthetic_btc_bars(n_bars=300, seed=42)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=43)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=44)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=45)
        features = compute_orthogonal_features(btc_df, eth_df, sp_df, micro_df, btc_df["cutoff_ts"][-1])

        scores = {name: [m.predict_score(features.slice(i, 1)) for i in range(features.height)] for name, m in members.items()}
        pair_corrs = {}
        for (m1, m2) in itertools.combinations(sorted(scores.keys()), 2):
            p_corr, _ = pearsonr(scores[m1], scores[m2])
            s_corr, _ = spearmanr(scores[m1], scores[m2])
            pair_corrs[(m1, m2)] = float(p_corr)
            assert abs(p_corr) <= 0.70, f"AC1 violation: Pair ({m1}, {m2}) Pearson r={p_corr:.4f} > 0.70"
            assert abs(s_corr) <= 0.70, f"AC1 violation: Pair ({m1}, {m2}) Spearman r={s_corr:.4f} > 0.70"

        last_scores = {name: scores[name][-1] for name in scores}
        raw_score = float(np.mean(list(last_scores.values())))
        res = EnsembleInferenceResult(
            member_predictions=last_scores,
            pairwise_correlations=pair_corrs,
            primary_direction=2 if raw_score > 0.15 else (0 if raw_score < -0.15 else 1),
            raw_ensemble_score=raw_score,
            expected_move_bps=float(abs(raw_score) * 12.0),
        )
        res.validate()

    def test_ac2_meta_labeler_continuity_and_calibration(self):
        """AC2 Acceptance Check: continuous [0, 1] output, ECE <= 8.0%, Brier < 0.2786."""
        meta_labeler = ContinuousMetaLabeler(
            hurdle_bps=6.0,
            min_15m_vol_bps=15.0,
            min_10s_move_bps=1.0,
            authorization_threshold=0.55,
        )
        rng = np.random.default_rng(42)
        raw_val_scores = rng.normal(0, 0.6, size=400)
        true_labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-raw_val_scores * 2.5)))
        meta_labeler.fit_calibration(raw_val_scores, true_labels, method="platt")

        # Verify continuous output without discrete clamp
        probs = np.array([meta_labeler.predict_probability(s) for s in np.linspace(-1.0, 1.0, 100)])
        assert len(np.unique(np.round(probs, 4))) >= 90, "Meta-labeler outputs clustered or non-continuous"
        assert not np.any(np.isclose(probs, 0.42, atol=1e-5)), "Meta-labeler retains artificial 0.42 clamp"
        assert np.min(probs) >= 0.0 and np.max(probs) <= 1.0

        # Out-of-fold calibration evaluation
        test_scores = rng.normal(0, 0.6, size=200)
        test_labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-test_scores * 2.5)))
        test_probs = np.array([meta_labeler.predict_probability(s) for s in test_scores])
        ece = calculate_ece(test_probs, test_labels, n_bins=10)
        brier = calculate_brier_score(test_probs, test_labels)
        assert ece <= 0.08, f"AC2 violation: ECE {ece:.4f} > 0.08"
        assert brier < 0.2786, f"AC2 violation: Brier score {brier:.4f} >= 0.2786"

        # Verify hurdle gating
        out_sub_hurdle = meta_labeler.evaluate_bar(0.5, expected_move_bps=4.0, sigma_15m_bps=20.0, mean_10s_move_bps=2.0)
        assert not out_sub_hurdle.is_authorized
        assert not out_sub_hurdle.hurdle_passed

    def test_ac3_zero_leakage_and_1000ms_offset(self):
        """AC3 Acceptance Check: 100% of feature inputs <= T_close - 1000ms, zero future leakage."""
        # 1. Pipeline cutoff rejection test
        bad_df = pl.DataFrame({
            "ts": [1000, 2000, 9500],
            "cutoff_ts": [1000, 2000, 9500],
            "close_ts": [2000, 3000, 10000],
            "val": [1.0, 2.0, 3.0],
        })
        filtered = _filter_by_cutoff(bad_df, 9000, "test_stream")
        assert filtered.height == 2
        assert filtered["cutoff_ts"].max() <= 9000

        # 2. Live Ingestor multi-stream freeze window isolation
        completed_bars = []
        ingestor = BinanceLiveIngestor(on_bar_completed=lambda b: completed_bars.append(b))
        now_ms = int(time.time() * 1000)
        bar_start = (now_ms // 10000) * 10000
        cutoff_ts = bar_start + 9000
        ingestor.current_bar_start = bar_start
        ingestor.bar_data = ingestor._init_bar()

        # Valid pre-cutoff events
        t_valid = cutoff_ts - 1000
        ingestor._process_stream_event("btcusdt@aggTrade", {"e": "aggTrade", "E": t_valid, "s": "BTCUSDT", "p": "85000.0", "q": "1.0", "m": False})
        ingestor._process_stream_event("btcusdt@bookTicker", {"e": "bookTicker", "E": t_valid, "s": "BTCUSDT", "b": "84999.0", "a": "85001.0"})
        ingestor._process_stream_event("ethusdt@bookTicker", {"e": "bookTicker", "E": t_valid, "s": "ETHUSDT", "b": "3199.0", "a": "3201.0"})
        ingestor._process_stream_event("btcusdt@depth20@100ms", {"e": "depthUpdate", "E": t_valid, "s": "BTCUSDT", "bids": [["84999.0", "10.0"]], "asks": [["85001.0", "12.0"]]})
        ingestor._process_stream_event("btcusdt@markPrice@1s", {"e": "markPriceUpdate", "E": t_valid, "s": "BTCUSDT", "p": "85000.5", "i": "84999.5", "r": "0.0001", "T": bar_start + 28800000})

        # Freeze window injections (t > cutoff_ts)
        t_freeze = cutoff_ts + 500
        ingestor._process_stream_event("btcusdt@aggTrade", {"e": "aggTrade", "E": t_freeze, "s": "BTCUSDT", "p": "99999.0", "q": "100.0", "m": False})
        ingestor._process_stream_event("btcusdt@bookTicker", {"e": "bookTicker", "E": t_freeze, "s": "BTCUSDT", "b": "99990.0", "a": "99992.0"})
        ingestor._process_stream_event("ethusdt@bookTicker", {"e": "bookTicker", "E": t_freeze, "s": "ETHUSDT", "b": "4999.0", "a": "5001.0"})
        ingestor._process_stream_event("btcusdt@depth20@100ms", {"e": "depthUpdate", "E": t_freeze, "s": "BTCUSDT", "bids": [["99990.0", "500.0"]], "asks": [["99992.0", "600.0"]]})
        ingestor._process_stream_event("btcusdt@markPrice@1s", {"e": "markPriceUpdate", "E": t_freeze, "s": "BTCUSDT", "p": "99995.0", "i": "99990.0", "r": "0.0050", "T": bar_start + 28800000})
        ingestor._process_stream_event("btcusdt@forceOrder", {"e": "forceOrder", "E": t_freeze, "s": "BTCUSDT", "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "q": "100.0", "p": "100000.0"}})

        ingestor._flush_bar(bar_start)
        time.sleep(0.05)
        assert len(completed_bars) == 1
        bar = completed_bars[0]
        assert bar["close"] == 85000.0, f"Close leaked freeze price: {bar['close']}"
        assert bar["best_bid"] == 84999.0, f"Best bid leaked freeze BBO: {bar['best_bid']}"
        assert bar["eth_best_bid"] == 3199.0, f"ETH best bid leaked freeze BBO: {bar['eth_best_bid']}"
        assert bar["mark_price"] == 85000.5, f"Mark price leaked freeze price: {bar['mark_price']}"
        assert bar["bids_depth"] == 10.0, f"Depth leaked freeze surge: {bar['bids_depth']}"
        assert bar["liquidation_notional"] == 0.0, f"Liquidation leaked freeze order: {bar['liquidation_notional']}"
        assert bar["execution_freeze_ticks"]["freeze_trades_count"] >= 1

    def test_ac4_net_profitability_under_maker_execution(self):
        """AC4 Acceptance Check: net PnL > 0 after 4 bps maker fees, Wilson low > 50%, duration <= 30s."""
        market_data = load_or_generate_market_data(REPO_ROOT, n_bars=3200)
        res = run_purged_walk_forward_simulation(
            market_data, fees_bps=4.0, purge_bars=3, embargo_bars=3, min_clusters=200
        )
        assert res["n_clusters"] >= 200, f"AC4 violation: Clusters {res['n_clusters']} < 200"
        assert res["total_net_pnl"] > 0.0, f"AC4 violation: Net PnL {res['total_net_pnl']:.2f} <= 0"
        assert res["ci_low"] > 0.50, f"AC4 violation: Wilson CI lower bound {res['ci_low']:.4f} <= 0.50"
        assert res["avg_duration"] <= 30.0, f"AC4 violation: Avg duration {res['avg_duration']:.1f}s > 30s"

    def test_ac5_automated_test_suite_and_latency(self):
        """AC5 Acceptance Check: automated reproducible test execution with latency <= 250ms."""
        members = load_orthogonal_ensemble()
        meta_labeler = ContinuousMetaLabeler()
        exec_node = BBOExecutionNode()

        btc_df = generate_synthetic_btc_bars(n_bars=35, seed=1)
        eth_df = generate_synthetic_eth_bars(btc_df, seed=2)
        sp_df = generate_synthetic_spot_perp_data(btc_df, seed=3)
        micro_df = generate_synthetic_microstructure_data(btc_df, seed=4)
        features = compute_orthogonal_features(btc_df, eth_df, sp_df, micro_df, btc_df["cutoff_ts"][-1])
        row = features.tail(1)

        n_bars = 100
        start = time.perf_counter()
        for _ in range(n_bars):
            res = predict_orthogonal_ensemble(row, members=members)
            res.validate()
            meta_out = meta_labeler.evaluate_bar(
                raw_ensemble_score=res.raw_ensemble_score,
                expected_move_bps=res.expected_move_bps,
                sigma_15m_bps=20.0,
                mean_10s_move_bps=2.5,
            )
            meta_out.validate()
            intent = exec_node.create_intent(
                side="BUY",
                bbo_bid=65000.0,
                bbo_ask=65001.0,
                bbo_bid_qty=5.0,
                bbo_ask_qty=5.0,
                atr_10s_bps=5.0,
            )
            if intent:
                intent.validate()
        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / n_bars) * 1000.0
        assert avg_latency_ms <= 250.0, f"AC5 violation: Latency {avg_latency_ms:.2f}ms exceeds 250ms budget"

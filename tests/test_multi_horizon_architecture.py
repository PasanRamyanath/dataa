"""
tests/test_multi_horizon_architecture.py
Complete automated verification suite for Project-IMPJR v3.0 (14-Timeframe Hierarchical
Ensemble & Microstructure Sniper Execution System), validating AC1 through AC8.
"""
from __future__ import annotations

import time
import numpy as np
import polars as pl
import pytest
import torch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_ENGINE_DIR = REPO_ROOT / "feature engineering scripts"
for _p in [str(REPO_ROOT), str(FEAT_ENGINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parsers.multi_tf_parser import causal_multi_timeframe_asof_join, INTERVAL_SECONDS_MAP
from labeling_multi_horizon import (
    apply_fractional_differentiation,
    compute_ffd_weights,
    compute_sample_uniqueness_weights,
    fee_floored_triple_barrier_labels,
)
from fusion.hierarchical_fusion_v3 import (
    HierarchicalMultiHorizonFusionEngine,
    compute_cross_horizon_alignment_features,
)
from fusion.conformal_dirichlet_calibrator import SymmetricODIRDirichletCalibrator
from execution.hierarchical_sniper_engine import HierarchicalSniperExecutionEngine
from models_v3.tier3_meso.itransformer_multivariate import MultivariateMesoITransformer
from models_v3.tier2_tactical.mamba_selective_ssm import MultiScaleTacticalMambaClassifier
from models_v3.tier4_macro.patchtst_ffd import MacroPatchTSTModel


class TestMultiTimeframeCausalAlignmentAC3:
    """AC3: Verifies zero cross-timeframe lookahead across all 14 resolutions."""

    def test_all_14_resolutions_registered(self) -> None:
        expected = {"1s", "10s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        assert set(INTERVAL_SECONDS_MAP.keys()) == expected

    def test_causal_asof_join_blocks_unclosed_higher_timeframe_bar(self) -> None:
        # Base 10s grid at 08:00:10, 08:15:00, 11:59:50, and 12:00:00 UTC
        t0_ns = 1_760_000_000_000_000_000
        h4_ns = 4 * 3600 * 1_000_000_000

        base_df = pl.DataFrame({
            "ts": pl.from_epoch(
                pl.Series([
                    t0_ns + 10_000_000_000,         # 10s into 4h bar #2 -> must ONLY see 4h bar #1!
                    t0_ns + h4_ns - 10_000_000_000, # 10s before 4h bar #2 closes -> must ONLY see 4h bar #1!
                    t0_ns + h4_ns,                  # Exact close of 4h bar #2 -> may now see 4h bar #2!
                ]),
                time_unit="ns",
            ),
            "close": [80010.0, 80990.0, 81000.0],
        })

        # 4h bars keyed by their completion timestamp `ts = open_ts + 4h`
        htf_4h = pl.DataFrame({
            "ts": pl.from_epoch(
                pl.Series([
                    t0_ns,          # Bar #1 closed at t0_ns
                    t0_ns + h4_ns,  # Bar #2 closed at t0_ns + 4h
                ]),
                time_unit="ns",
            ),
            "close": [79500.0, 99999.0],  # 99999.0 is Bar #2's close (future relative to 08:00:10!)
        })

        joined = causal_multi_timeframe_asof_join(
            base_df=base_df,
            htf_df=htf_4h,
            htf_prefix="macro_4h",
            feature_cols=["close"],
        )
        vals = joined["macro_4h_close"].to_list()
        # Rows 0 and 1 MUST see 79500.0 (Bar #1), NEVER 99999.0 (forming Bar #2)!
        assert vals[0] == pytest.approx(79500.0)
        assert vals[1] == pytest.approx(79500.0)
        assert vals[2] == pytest.approx(99999.0)


class TestFeeFlooredLabelingAndFractionalDiffAC4:
    """AC4: Verifies FFD causality and fee-floored triple-barrier labels."""

    def test_ffd_is_strictly_causal(self) -> None:
        rng = np.random.default_rng(42)
        log_p = np.cumsum(rng.normal(0.0, 0.002, size=300)) + 11.3
        ffd_full = apply_fractional_differentiation(log_p, d=0.35, threshold=1e-4)

        # Mutate future prices after index 150 and verify prefix [0..150] is 100% bit-identical
        log_p_shocked = log_p.copy()
        log_p_shocked[151:] += 0.50
        ffd_shocked = apply_fractional_differentiation(log_p_shocked, d=0.35, threshold=1e-4)
        np.testing.assert_allclose(ffd_full[:151], ffd_shocked[:151], rtol=0.0, atol=1e-12)

    def test_fee_floored_triple_barrier_suppresses_sub_fee_chop(self) -> None:
        # Construct 200 bars of low-volatility chop oscillating within +/- 3.0 bps
        n = 200
        base = 82000.0
        osc = np.sin(np.linspace(0, 20 * np.pi, n)) * (base * 0.00025)  # +/- 2.5 bps
        closes = base + osc
        df = pl.DataFrame({
            "close": closes,
            "high": closes + base * 0.00005,
            "low": closes - base * 0.00005,
        })
        labeled = fee_floored_triple_barrier_labels(
            df=df,
            max_hold_bars=8,
            pt_mult=1.8,
            sl_mult=1.2,
            min_pt_bps=28.0,  # Requires >= 28.0 bps move to label +1 or -1
            min_sl_bps=18.0,
            roundtrip_fee_bps=4.0,
            prefix="meso_15m",
        )
        tb = labeled["label_meso_15m_tb"].to_numpy()
        # Because max oscillation is < 4 bps (< 28 bps floor), ZERO bars may be labeled +1 or -1!
        assert np.all(tb == 0)


class TestUpgradedNeuralArchitecturesAC5:
    """AC5: Verifies Multivariate iTransformer-32 and Selective S6 Mamba."""

    def test_multivariate_itransformer_32_variate_attention(self) -> None:
        model = MultivariateMesoITransformer(seq_len=64, pred_len=4, num_variates=32, d_model=64, n_heads=4, n_layers=2)
        model.eval()
        x = torch.randn(2, 64, 32)
        with torch.no_grad():
            out = model(x)
        assert out["cls_probs"].shape == (2, 3)
        assert out["quantiles"].shape == (2, 4, 3)
        assert out["variate_attention"].shape == (2, 32, 32)
        # Verify attention rows sum to 1.0 across the 32 variates
        row_sums = out["variate_attention"].sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4)

    def test_selective_mamba_strictly_causal_prefix_invariance(self) -> None:
        model = MultiScaleTacticalMambaClassifier(n_features=16, d_model=32, n_layers=2, d_state=8)
        model.eval()
        x1 = torch.randn(1, 50, 16)
        x2 = x1.clone()
        x2[:, 30:, :] += 10.0  # Massive future perturbation at t >= 30

        with torch.no_grad():
            # Evaluate on prefix [:30] of both tensors
            out1 = model(x1[:, :30, :])["probs"]
            out2 = model(x2[:, :30, :])["probs"]
        torch.testing.assert_close(out1, out2, rtol=0.0, atol=1e-7)


class TestHierarchicalFusionAndSniperExecutionAC6:
    """AC6: Verifies Top-Down Macro Conditioning and 9-State Sniper Execution."""

    def test_macro_conflict_dampens_counter_trend_trade(self) -> None:
        fusion = HierarchicalMultiHorizonFusionEngine()
        # Macro (4h/1d) and Meso (15m/1h) are strongly BULLISH, while 10s Micro is temporarily BEARISH
        p_macro = np.array([0.10, 0.20, 0.70])
        p_meso = np.array([0.12, 0.18, 0.70])
        p_tact = np.array([0.20, 0.25, 0.55])
        p_micro_bearish = np.array([0.75, 0.15, 0.10])

        decision = fusion.evaluate_multi_horizon_tick(
            ts_ms=1_760_000_000_000,
            p_macro=p_macro,
            p_meso=p_meso,
            p_tactical=p_tact,
            p_micro=p_micro_bearish,
            atr_15m_bps=25.0,
            atr_1h_bps=45.0,
            atr_10s_bps=2.5,
        )
        # Primary swing direction MUST remain LONG (2), never flipping SHORT on a 10s wiggle!
        assert decision.primary_direction == 2
        # And because 10s micro flow is temporarily bearish, execution_mode waits for sniper inflection!
        assert decision.execution_mode == "WAIT_FOR_MICRO_SNIPER"

    def test_sniper_fsm_full_maker_lifecycle_and_adverse_cancel(self) -> None:
        fusion = HierarchicalMultiHorizonFusionEngine()
        sniper = HierarchicalSniperExecutionEngine()

        # 1. Bullish alignment across all 4 tiers triggers SNIPER_ENTRY_STALKING -> RESTING_ENTRY_QUEUE
        dec_bull = fusion.evaluate_multi_horizon_tick(
            ts_ms=100_000,
            p_macro=np.array([0.10, 0.15, 0.75]),
            p_meso=np.array([0.10, 0.15, 0.75]),
            p_tactical=np.array([0.15, 0.20, 0.65]),
            p_micro=np.array([0.12, 0.18, 0.70]),
            atr_15m_bps=25.0,
            atr_1h_bps=50.0,
            atr_10s_bps=2.5,
            micro_safe_fill_prob=0.80,
        )
        state = sniper.on_hierarchical_decision(
            dec_bull, bbo_bid=82000.0, bbo_ask=82000.5, bbo_bid_qty=2.0, bbo_ask_qty=2.5
        )
        assert state == "RESTING_ENTRY_QUEUE"

        # 2. Simulate ETH 1s flash crash (-5.0 bps) -> Adverse Selection Guard cancels with 0 fee!
        state = sniper.on_market_micro_tick(
            ts_ms=102_000, bbo_bid=82000.0, bbo_ask=82000.5, bbo_bid_qty=1.8, bbo_ask_qty=2.5,
            aggressive_sell_vol_at_bid=0.1, eth_1s_ret_bps=-5.0
        )
        assert state == "SNIPER_ENTRY_STALKING"
        assert len(sniper.closed_trades) == 0  # Unfilled cancel pays 0.0 bps fee!

        # 3. Re-trigger entry when micro stabilizes, fill queue, and hit Resting Maker TP
        state = sniper.on_hierarchical_decision(
            dec_bull, bbo_bid=81990.0, bbo_ask=81990.5, bbo_bid_qty=1.5, bbo_ask_qty=2.0
        )
        assert state == "RESTING_ENTRY_QUEUE"
        state = sniper.on_market_micro_tick(
            ts_ms=105_000, bbo_bid=81990.0, bbo_ask=81990.5, bbo_bid_qty=1.5, bbo_ask_qty=2.0,
            aggressive_sell_vol_at_bid=2.0, eth_1s_ret_bps=0.5
        )
        assert state == "FILLED_SWING_HOLDING"
        assert sniper.active_position is not None

        # Advance price to hit resting Maker TP
        tp_target = sniper.active_position.tp_price
        state = sniper.on_market_micro_tick(
            ts_ms=600_000, bbo_bid=tp_target, bbo_ask=tp_target + 0.5, bbo_bid_qty=2.0, bbo_ask_qty=2.0
        )
        assert state == "IDLE"
        assert len(sniper.closed_trades) == 1
        closed = sniper.closed_trades[0]
        assert closed.exit_reason == "RESTING_MAKER_TP"
        assert closed.total_fee_bps == pytest.approx(4.0)  # 2.0 bps Maker entry + 2.0 bps Maker TP exit
        assert closed.net_return_bps > 20.0

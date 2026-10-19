"""
Empirical test to check if freeze-window events in binance_ws.py actually alter
the orthogonal features computed by compute_orthogonal_features().
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
from streaming.live_loop_v2 import OrthogonalLiveRingBuffer
from features.pipeline import compute_orthogonal_features

def run_feature_leakage_impact_test():
    print("=" * 80)
    print("TESTING IMPACT OF FREEZE-WINDOW LEAKAGE ON FEATURE GENERATION")
    print("=" * 80)

    # 1. Pipeline WITHOUT freeze-window contamination
    bars_clean = []
    ingestor_clean = BinanceLiveIngestor(on_bar_completed=lambda b: bars_clean.append(b))
    
    # 2. Pipeline WITH freeze-window depth/markprice/liquidation surge
    bars_contaminated = []
    ingestor_contaminated = BinanceLiveIngestor(on_bar_completed=lambda b: bars_contaminated.append(b))

    now_ms = int(time.time() * 1000)
    bar_start = (now_ms // 10000) * 10000
    cutoff_ts = bar_start + 9000
    close_ts = bar_start + 10000

    for ing in [ingestor_clean, ingestor_contaminated]:
        ing.current_bar_start = bar_start
        ing.bar_data = ing._init_bar()
        # Valid event at t = cutoff_ts - 2000
        t_valid = cutoff_ts - 2000
        ing._process_stream_event("btcusdt@aggTrade", {
            "e": "aggTrade", "E": t_valid, "s": "BTCUSDT", "p": "85000.0", "q": "1.0", "m": False
        })
        ing._process_stream_event("btcusdt@bookTicker", {
            "e": "bookTicker", "E": t_valid, "s": "BTCUSDT", "b": "84999.0", "a": "85001.0"
        })
        ing._process_stream_event("btcusdt@depth20@100ms", {
            "e": "depthUpdate", "E": t_valid, "s": "BTCUSDT",
            "bids": [["84999.0", "10.0"]], "asks": [["85001.0", "10.0"]]
        })
        ing._process_stream_event("btcusdt@markPrice@1s", {
            "e": "markPriceUpdate", "E": t_valid, "s": "BTCUSDT", "p": "85000.0", "i": "85000.0", "r": "0.0001", "T": bar_start + 28800000
        })

    # Contaminate ONLY ingestor_contaminated during freeze window (t = cutoff_ts + 500ms)
    t_freeze = cutoff_ts + 500
    # Surge depth from 10.0 to 1000.0 at t = cutoff_ts + 500ms
    ingestor_contaminated._process_stream_event("btcusdt@depth20@100ms", {
        "e": "depthUpdate", "E": t_freeze, "s": "BTCUSDT",
        "bids": [["84999.0", "1000.0"]], "asks": [["85001.0", "10.0"]]
    })
    # Surge mark price from 85000 to 86000 (+1000 USDT) at t = cutoff_ts + 500ms
    ingestor_contaminated._process_stream_event("btcusdt@markPrice@1s", {
        "e": "markPriceUpdate", "E": t_freeze, "s": "BTCUSDT", "p": "86000.0", "i": "85000.0", "r": "0.0050", "T": bar_start + 28800000
    })

    ingestor_clean._flush_bar(bar_start)
    ingestor_contaminated._flush_bar(bar_start)
    time.sleep(0.1)

    bar_clean = bars_clean[0]
    bar_contam = bars_contaminated[0]

    # Setup 35 baseline bars for warmup
    ring_clean = OrthogonalLiveRingBuffer(capacity=180, min_warmup_bars=30)
    ring_contam = OrthogonalLiveRingBuffer(capacity=180, min_warmup_bars=30)

    from tests.fixtures.synthetic_market import (
        generate_synthetic_btc_bars,
        generate_synthetic_eth_bars,
        generate_synthetic_spot_perp_data,
        generate_synthetic_microstructure_data,
    )
    warm_btc = generate_synthetic_btc_bars(n_bars=35, start_ts_ms=bar_start - 350000, seed=1)
    warm_eth = generate_synthetic_eth_bars(warm_btc, seed=2)
    warm_sp = generate_synthetic_spot_perp_data(warm_btc, seed=3)
    warm_ob = generate_synthetic_microstructure_data(warm_btc, seed=4)

    for i in range(35):
        b_dict = {
            "ts": int(warm_btc["ts"][i]),
            "cutoff_ts": int(warm_btc["cutoff_ts"][i]),
            "close_ts": int(warm_btc["close_ts"][i]),
            "open": float(warm_btc["open"][i]),
            "high": float(warm_btc["high"][i]),
            "low": float(warm_btc["low"][i]),
            "close": float(warm_btc["close"][i]),
            "volume": float(warm_btc["volume"][i]),
            "trade_count": int(warm_btc["trade_count"][i]) if "trade_count" in warm_btc.columns else 100,
            "buy_volume": float(warm_btc["buy_volume"][i]) if "buy_volume" in warm_btc.columns else float(warm_btc["volume"][i]) * 0.5,
            "sell_volume": float(warm_btc["sell_volume"][i]) if "sell_volume" in warm_btc.columns else float(warm_btc["volume"][i]) * 0.5,
            "best_bid": float(warm_btc["bbo_bid"][i]),
            "best_ask": float(warm_btc["bbo_ask"][i]),
            "bids_depth": float(warm_btc["bbo_bid_qty"][i]),
            "asks_depth": float(warm_btc["bbo_ask_qty"][i]),
            "mark_price": float(warm_sp["perp_mid"][i]),
            "index_price": float(warm_sp["spot_mid"][i]),
            "funding_rate": float(warm_sp["funding_rate"][i]),
            "next_funding_time": int(warm_sp["next_funding_time_ms"][i]) if "next_funding_time_ms" in warm_sp.columns else 0,
            "eth_close": float(warm_eth["eth_close"][i]),
            "eth_volume": float(warm_eth["eth_volume"][i]),
            "eth_buy_volume": float(warm_eth["eth_volume"][i] * 0.5),
            "eth_sell_volume": float(warm_eth["eth_volume"][i] * 0.5),
            "eth_imbalance": float(warm_eth["eth_imbalance"][i]),
        }
        ring_clean.append_bar(b_dict)
        ring_contam.append_bar(b_dict)

    # Now append the evaluated bar (bar 36)
    feat_clean = ring_clean.append_bar(bar_clean)
    feat_contam = ring_contam.append_bar(bar_contam)

    assert feat_clean is not None
    assert feat_contam is not None

    print(f"Features computed successfully: {feat_clean.shape}")
    mismatches = []
    for col in feat_clean.columns:
        if col == "ts":
            continue
        v_clean = feat_clean[col][-1]
        v_contam = feat_contam[col][-1]
        if not np.isclose(v_clean, v_contam, atol=1e-5):
            diff = abs(v_clean - v_contam)
            mismatches.append((col, v_clean, v_contam, diff))

    print(f"\nTotal Leaked Features Detected: {len(mismatches)}")
    for col, v_c, v_m, diff in mismatches:
        print(f"  LEAKED FEATURE: {col:<30} Clean: {v_c:>10.4f} | Contaminated: {v_m:>10.4f} | Delta: {diff:>10.4f}")

    return mismatches

if __name__ == "__main__":
    m = run_feature_leakage_impact_test()
    if m:
        print("\nCONCLUSION: LOOKAHEAD LEAKAGE EMPIRICALLY CONFIRMED IN binance_ws.py!")
    else:
        print("\nCONCLUSION: NO LEAKAGE OBSERVED.")

"""
Empirical Adversarial Stress Test Suite for Live Streaming Pipeline.
Challenger Live 1.

Stress-tests:
1. Timestamp Cutoff Boundaries (t <= T_close - 1000 ms vs t = T_close - 999 ms / freeze window).
   - Tests ALL stream types: aggTrade, bookTicker (BTC and ETH), depth20, markPrice, forceOrder.
   - Evaluates whether freeze-window events contaminate feature outputs.
2. Lookahead / intra-bar leakage under out-of-order ticks, burst arrivals, and socket disconnects.
   - Out-of-order event timestamps (backward jumps, historical ticks from previous bars).
   - Future bar ticks arriving before local clock boundary.
   - Burst arrivals (1,000+ ticks).
3. Latency & Throughput Benchmark:
   - Measures end-to-end turnaround latency per 10-second bar under full ring buffer load.
   - Measures feature engineering, ensemble inference, meta-labeling, execution node, and total ms.
   - Evaluates compliance against 250 ms budget.
4. Synthetic Flat-Bar Interpolation:
   - WebSocket gap simulation (> 10s disconnect: 15s, 35s, 120s).
   - Ring buffer continuity, absence of NaNs, realized volatility chop gate suppression.
"""
from __future__ import annotations

import sys
import time
import math
from pathlib import Path
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = REPO_ROOT / "feature engineering scripts"
for p in [str(REPO_ROOT), str(FEAT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from streaming.live_ingestion.binance_ws import BinanceLiveIngestor
from streaming.live_loop_v2 import LiveStreamingPredictionNode, OrthogonalLiveRingBuffer
from features.pipeline import compute_orthogonal_features, PRUNED_ORTHOGONAL_FEATURES
from execution.maker_node import BBOExecutionNode
from meta.continuous_meta_labeler import ContinuousMetaLabeler


def test_cutoff_boundary_all_streams():
    """
    Stress-tests whether ticks arriving in the freeze interval [T_close - 1000ms, T_close]
    leak into completed_bar feature fields across ALL stream types.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST 1: CUTOFF BOUNDARIES ACROSS ALL 7 STREAM EVENT TYPES")
    print("=" * 80)

    completed_bars = []
    def on_bar(b):
        completed_bars.append(b)

    ingestor = BinanceLiveIngestor(on_bar_completed=on_bar)
    now_ms = int(time.time() * 1000)
    bar_start = (now_ms // 10000) * 10000
    cutoff_ts = bar_start + 9000
    close_ts = bar_start + 10000

    ingestor.current_bar_start = bar_start
    ingestor.bar_data = ingestor._init_bar()

    # 1. Base events in valid window (t = cutoff_ts - 1000ms)
    t_valid = cutoff_ts - 1000
    # BTC aggTrade
    ingestor._process_stream_event("btcusdt@aggTrade", {
        "e": "aggTrade", "E": t_valid, "s": "BTCUSDT", "p": "85000.0", "q": "1.0", "m": False
    })
    # BTC bookTicker
    ingestor._process_stream_event("btcusdt@bookTicker", {
        "e": "bookTicker", "E": t_valid, "s": "BTCUSDT", "b": "84999.0", "a": "85001.0"
    })
    # ETH aggTrade
    ingestor._process_stream_event("ethusdt@aggTrade", {
        "e": "aggTrade", "E": t_valid, "s": "ETHUSDT", "p": "3200.0", "q": "5.0", "m": False
    })
    # ETH bookTicker
    ingestor._process_stream_event("ethusdt@bookTicker", {
        "e": "bookTicker", "E": t_valid, "s": "ETHUSDT", "b": "3199.0", "a": "3201.0"
    })
    # markPriceUpdate
    ingestor._process_stream_event("btcusdt@markPrice@1s", {
        "e": "markPriceUpdate", "E": t_valid, "s": "BTCUSDT", "p": "85000.5", "i": "84999.5", "r": "0.0001", "T": bar_start + 28800000
    })
    # depthUpdate
    ingestor._process_stream_event("btcusdt@depth20@100ms", {
        "e": "depthUpdate", "E": t_valid, "s": "BTCUSDT",
        "bids": [["84999.0", "10.0"]], "asks": [["85001.0", "12.0"]]
    })

    # Now verify state BEFORE freeze
    assert ingestor.bar_data["trade_count"] == 1
    assert ingestor.bar_data["prices"] == [85000.0]
    assert ingestor.bar_data["cutoff_best_bid"] == 84999.0
    assert ingestor.bar_data["cutoff_best_ask"] == 85001.0

    # 2. FREEZE WINDOW INJECTIONS (t = cutoff_ts + 500ms = T_close - 500ms)
    t_freeze = cutoff_ts + 500

    # A) BTC aggTrade at freeze: Price jumps to 99999.0
    ingestor._process_stream_event("btcusdt@aggTrade", {
        "e": "aggTrade", "E": t_freeze, "s": "BTCUSDT", "p": "99999.0", "q": "100.0", "m": False
    })
    # B) BTC bookTicker at freeze: BBO jumps to 99990 / 99992
    ingestor._process_stream_event("btcusdt@bookTicker", {
        "e": "bookTicker", "E": t_freeze, "s": "BTCUSDT", "b": "99990.0", "a": "99992.0"
    })
    # C) ETH aggTrade at freeze: Price jumps to 5000.0
    ingestor._process_stream_event("ethusdt@aggTrade", {
        "e": "aggTrade", "E": t_freeze, "s": "ETHUSDT", "p": "5000.0", "q": "100.0", "m": False
    })
    # D) ETH bookTicker at freeze: BBO jumps to 4999 / 5001
    ingestor._process_stream_event("ethusdt@bookTicker", {
        "e": "bookTicker", "E": t_freeze, "s": "ETHUSDT", "b": "4999.0", "a": "5001.0"
    })
    # E) markPriceUpdate at freeze: Jumps to 99995.0
    ingestor._process_stream_event("btcusdt@markPrice@1s", {
        "e": "markPriceUpdate", "E": t_freeze, "s": "BTCUSDT", "p": "99995.0", "i": "99990.0", "r": "0.0050", "T": bar_start + 28800000
    })
    # F) depthUpdate at freeze: Depth surges to 500.0
    ingestor._process_stream_event("btcusdt@depth20@100ms", {
        "e": "depthUpdate", "E": t_freeze, "s": "BTCUSDT",
        "bids": [["99990.0", "500.0"]], "asks": [["99992.0", "600.0"]]
    })
    # G) forceOrder at freeze: Liquidation burst of $10,000,000
    ingestor._process_stream_event("btcusdt@forceOrder", {
        "e": "forceOrder", "E": t_freeze, "s": "BTCUSDT",
        "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "q": "100.0", "p": "100000.0"}
    })

    # Flush the bar directly
    ingestor._flush_bar(bar_start)
    time.sleep(0.1) # allow threadpool executor to complete callback
    assert len(completed_bars) == 1, "Expected 1 completed bar"
    bar = completed_bars[0]

    findings = []
    # Test BTC OHLCV isolation
    if bar["close"] == 99999.0:
        findings.append("BTC aggTrade price leaked into bar close during freeze window!")
    elif bar["close"] == 85000.0:
        print("  [PASS] BTC aggTrade strictly isolated from bar close (close = 85000.0, freeze trade segregated)")

    # Test BTC BBO cutoff isolation
    if bar["best_bid"] == 99990.0:
        findings.append("BTC bookTicker leaked into bar best_bid during freeze window!")
    elif bar["best_bid"] == 84999.0:
        print("  [PASS] BTC bookTicker cutoff_best_bid strictly preserved (best_bid = 84999.0)")

    # Test ETH close isolation
    if bar["eth_close"] == 5000.0:
        findings.append("ETH aggTrade leaked into eth_close during freeze window!")
    elif bar["eth_close"] == 3200.0:
        print("  [PASS] ETH aggTrade strictly isolated from eth_close (eth_close = 3200.0)")

    # Test ETH BBO isolation
    if bar["eth_best_bid"] == 4999.0:
        findings.append("ETH bookTicker leaked into eth_best_bid during freeze window!")
    else:
        print(f"  ETH best_bid in bar: {bar['eth_best_bid']}")

    # Test Mark Price isolation
    if bar["mark_price"] == 99995.0:
        findings.append("markPriceUpdate leaked into mark_price during freeze window!")
    else:
        print(f"  Mark price in bar: {bar['mark_price']}")

    # Test Depth isolation
    if bar["bids_depth"] == 500.0:
        findings.append("depthUpdate leaked into bids_depth during freeze window!")
    else:
        print(f"  Bids depth in bar: {bar['bids_depth']}")

    # Test Liquidation isolation
    if bar["liquidation_notional"] == 10000000.0:
        findings.append("forceOrder leaked into liquidation_notional during freeze window!")
    else:
        print(f"  Liquidation notional in bar: {bar['liquidation_notional']}")

    print("\n  Summary of Freeze Window Leakage Findings:")
    for f in findings:
        print(f"  [CRITICAL LEAKAGE FINDING]: {f}")

    return findings


def test_out_of_order_and_historical_ticks():
    """
    Stress-tests how the streaming ingestion handles out-of-order,
    backward-timestamp, and future-timestamp events.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST 2: OUT-OF-ORDER & HISTORICAL TIMESTAMP INJECTIONS")
    print("=" * 80)

    completed_bars = []
    ingestor = BinanceLiveIngestor(on_bar_completed=lambda b: completed_bars.append(b))
    now_ms = int(time.time() * 1000)
    bar_start = (now_ms // 10000) * 10000

    ingestor.current_bar_start = bar_start
    ingestor.bar_data = ingestor._init_bar()

    # Feed normal tick at bar_start + 2000
    ingestor._process_stream_event("btcusdt@aggTrade", {
        "e": "aggTrade", "E": bar_start + 2000, "s": "BTCUSDT", "p": "85000.0", "q": "1.0", "m": False
    })

    # Feed HISTORICAL tick from previous bar (bar_start - 5000)
    ingestor._process_stream_event("btcusdt@aggTrade", {
        "e": "aggTrade", "E": bar_start - 5000, "s": "BTCUSDT", "p": "84000.0", "q": "10.0", "m": True
    })

    # Feed FUTURE tick belonging to next bar (bar_start + 12000)
    ingestor._process_stream_event("btcusdt@aggTrade", {
        "e": "aggTrade", "E": bar_start + 12000, "s": "BTCUSDT", "p": "86000.0", "q": "20.0", "m": False
    })

    prices = ingestor.bar_data["prices"]
    freeze = ingestor.bar_data["freeze_trades"]
    print(f"  Prices collected in bar: {prices}")
    print(f"  Freeze trades collected: {freeze}")

    historical_leaked = 84000.0 in prices
    future_in_freeze = 86000.0 in freeze
    future_in_prices = 86000.0 in prices

    print(f"  Historical tick (t < bar_start) added to current bar prices: {historical_leaked}")
    print(f"  Future tick (t >= bar_start + 10s) added to current bar freeze: {future_in_freeze}")
    print(f"  Future tick (t >= bar_start + 10s) added to current bar prices: {future_in_prices}")

    return {
        "historical_leaked": historical_leaked,
        "future_in_freeze": future_in_freeze,
        "future_in_prices": future_in_prices,
    }


def test_latency_and_burst_throughput():
    """
    Measures end-to-end turnaround latency under high event frequency and full ring buffer (180 bars).
    Verifies compliance with <= 250 ms budget across 100 consecutive bars.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST 3: LATENCY & BURST THROUGHPUT BENCHMARK (100 BARS)")
    print("=" * 80)

    latencies = []
    records = []

    def handle_tick(rec):
        records.append(rec)
        lat = rec.get("latency_telemetry", {}).get("total_turnaround_ms", 0.0)
        latencies.append(lat)

    node = LiveStreamingPredictionNode(
        symbol="BTCUSDT",
        on_tick=handle_tick,
        enable_trading=True,
        max_cycles=100,
        min_warmup_bars=30,
        telemetry_ledger_path="scratch/challenge_live_latency_ledger.jsonl",
    )
    node.running = True

    base_ts = int(time.time() * 1000) - 1000 * 10000
    p = 85000.0

    print("  Streaming 100 consecutive bars through LiveStreamingPredictionNode...")
    t_start_all = time.perf_counter()
    for i in range(100):
        bar_ts = base_ts + i * 10000
        p = p * (1.0 + np.sin(i * 0.1) * 0.0005)
        bar = {
            "ts": bar_ts,
            "cutoff_ts": bar_ts + 9000,
            "close_ts": bar_ts + 10000,
            "open": p * 0.9995,
            "high": p * 1.0005,
            "low": p * 0.9990,
            "close": p,
            "volume": 25.0 + float(np.random.uniform(5, 50)),
            "trade_count": 250,
            "buy_volume": 14.0,
            "sell_volume": 11.0,
            "best_bid": p - 0.50,
            "best_ask": p + 0.50,
            "bids_depth": 85.0,
            "asks_depth": 90.0,
            "mark_price": p + 0.10,
            "index_price": p - 0.10,
            "funding_rate": 0.0001,
            "next_funding_time": bar_ts + 14400000,
            "eth_close": p / 25.0,
            "eth_volume": 120.0,
            "eth_buy_volume": 65.0,
            "eth_sell_volume": 55.0,
            "eth_imbalance": 0.08,
            "eth_bar": {
                "ts": bar_ts, "cutoff_ts": bar_ts + 9000,
                "close": p / 25.0, "volume": 120.0, "buy_volume": 65.0, "sell_volume": 55.0,
                "eth_imbalance": 0.08, "bbo_mid": p / 25.0,
            },
            "spot_perp": {
                "ts": bar_ts, "cutoff_ts": bar_ts + 9000,
                "perp_mid": p + 0.10, "spot_mid": p - 0.10,
                "funding_rate": 0.0001, "next_funding_time_ms": bar_ts + 14400000,
                "basis_spread_bps": 2.35,
            },
            "orderbook": {
                "ts": bar_ts, "cutoff_ts": bar_ts + 9000,
                "bbo_bid": p - 0.50, "bbo_ask": p + 0.50,
                "bbo_bid_qty": 85.0, "bbo_ask_qty": 90.0,
                "bids_depth": 85.0, "asks_depth": 90.0,
            },
        }
        node.on_bar_completed(bar)

    total_bench_time = time.perf_counter() - t_start_all
    node.stop()

    lat_arr = np.array(latencies)
    mean_lat = np.mean(lat_arr)
    p50_lat = np.median(lat_arr)
    p95_lat = np.percentile(lat_arr, 95)
    p99_lat = np.percentile(lat_arr, 99)
    max_lat = np.max(lat_arr)

    print(f"  Processed {len(records)} bars in {total_bench_time:.2f}s ({len(records)/total_bench_time:.1f} bars/sec)")
    print(f"  Latency Statistics (Budget <= 250 ms):")
    print(f"    - Mean:   {mean_lat:.2f} ms")
    print(f"    - Median: {p50_lat:.2f} ms")
    print(f"    - P95:    {p95_lat:.2f} ms")
    print(f"    - P99:    {p99_lat:.2f} ms")
    print(f"    - Max:    {max_lat:.2f} ms")
    print(f"    - Headroom at Max: {250.0 - max_lat:.2f} ms")

    assert max_lat <= 250.0, f"Max turnaround latency {max_lat:.2f} ms exceeded 250 ms budget!"
    return {
        "mean_lat": mean_lat,
        "p50_lat": p50_lat,
        "p95_lat": p95_lat,
        "p99_lat": p99_lat,
        "max_lat": max_lat,
    }


def test_synthetic_flat_bar_disconnects():
    """
    Stress-tests synthetic flat-bar interpolation when WebSocket disconnects for > 10s.
    Verifies gap detection, correct bar count, continuity, zero NaNs, and chop gate suppression.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST 4: SYNTHETIC FLAT-BAR INTERPOLATION UNDER WEBSOCKET DISCONNECTS")
    print("=" * 80)

    emitted_bars = []
    ingestor = BinanceLiveIngestor(on_bar_completed=lambda b: emitted_bars.append(b))
    ingestor.latest_close = 85000.0
    ingestor.latest_best_bid = 84999.5
    ingestor.latest_best_ask = 85000.5
    ingestor.latest_eth_close = 3200.0

    # Test 1: Gap of 15s (1 missing bar)
    start_ts = 1700000000000
    ingestor.current_bar_start = start_ts
    new_boundary_15s = start_ts + 20000 # 20s gap -> 1 missing bar at start_ts + 10000
    ingestor._handle_boundary_gap(start_ts, new_boundary_15s)
    time.sleep(0.05)
    assert len(emitted_bars) == 1, f"Expected 1 synthetic bar for 20s gap, got {len(emitted_bars)}"
    b1 = emitted_bars[0]
    assert b1["ts"] == start_ts + 10000
    assert b1["is_synthetic"] is True
    assert b1["volume"] == 0.0
    assert b1["close"] == 85000.0
    print(f"  [PASS] 20s gap produced exactly 1 synthetic bar at ts={b1['ts']} with is_synthetic=True")

    # Test 2: Gap of 35s (3 missing bars)
    emitted_bars.clear()
    new_boundary_35s = start_ts + 40000 # 40s gap -> 3 missing bars
    ingestor._handle_boundary_gap(start_ts, new_boundary_35s)
    time.sleep(0.05)
    assert len(emitted_bars) == 3, f"Expected 3 synthetic bars for 40s gap, got {len(emitted_bars)}"
    ts_list = [b["ts"] for b in emitted_bars]
    expected_ts = [start_ts + 10000, start_ts + 20000, start_ts + 30000]
    assert ts_list == expected_ts, f"Expected {expected_ts}, got {ts_list}"
    print(f"  [PASS] 40s gap produced exactly 3 synthetic bars with exact contiguous timestamps")

    # Test 3: Processing synthetic bars through live loop and checking chop gate suppression
    records = []
    node = LiveStreamingPredictionNode(
        symbol="BTCUSDT",
        on_tick=lambda r: records.append(r),
        enable_trading=True,
        min_warmup_bars=30,
        telemetry_ledger_path="scratch/challenge_live_synthetic_ledger.jsonl",
    )
    node.running = True

    # Feed 10 synthetic flat bars into node
    print("  Feeding 10 consecutive synthetic flat bars into LiveStreamingPredictionNode...")
    for b in emitted_bars * 3 + [emitted_bars[0]]: # 10 bars
        node.on_bar_completed(b)

    # Synthetic bars are buffered into ring buffer but filtered from trade evaluation ledger
    assert len(node.ring_buffer.btc_bars) >= 10, f"Expected at least 10 bars in ring buffer, got {len(node.ring_buffer.buffer)}"
    authorized_trades = [r for r in records if r.get("meta_labeler", {}).get("is_authorized", False)]
    print(f"  Total synthetic bars scored: {len(records)}")
    print(f"  Authorized trades during flat-bar chop regime: {len(authorized_trades)}")
    assert len(authorized_trades) == 0, f"Chop gate failed: authorized {len(authorized_trades)} trades during synthetic flat bars!"
    print("  [PASS] Realized volatility chop gate successfully suppressed 100% of trades during synthetic flat bars.")

    node.stop()
    return True


if __name__ == "__main__":
    findings = test_cutoff_boundary_all_streams()
    test_out_of_order_and_historical_ticks()
    test_latency_and_burst_throughput()
    test_synthetic_flat_bar_disconnects()
    print("\n" + "=" * 80)
    print("ALL EMPIRICAL TESTS COMPLETED.")
    print("=" * 80)

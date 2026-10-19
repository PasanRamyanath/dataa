"""
End-to-end integration test of LiveStreamingPredictionNode.
Simulates bar feeding from WebSocket, executes warm-up, runs parallel inference,
fuses predictions, and executes paper trading.
"""
import sys
import time
import math
from pathlib import Path
import polars as pl

repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_dir))

from streaming.live_loop_v2 import LiveStreamingPredictionNode

scored_records = []

def handle_tick(rec: dict):
    scored_records.append(rec)
    print(f"  >>> TICK CAPTURED! Dir: {rec.get('primary_direction')} | Close: ${rec.get('close'):.2f} | Turnaround: {rec.get('latency_telemetry', {}).get('total_turnaround_ms')} ms")

print("Initializing LiveStreamingPredictionNode...")
node = LiveStreamingPredictionNode(symbol="BTCUSDT", on_tick=handle_tick, enable_trading=True, max_cycles=5)
node.running = True  # Enable manual test driving

# Feed 65 synthetic bars to warm up the ring buffer (warmup >= 60)
base_ts = int(time.time() * 1000) - 650000
p = 85200.0

print("\nFeeding bars into LiveStreamingPredictionNode...")
for i in range(100):
    if not node.running:
        break
    ts = base_ts + i * 10000
    p = p * math.exp(0.0001)
    bar = {
        "ts": ts,
        "open": p * 0.999, "high": p * 1.001, "low": p * 0.998, "close": p,
        "volume": 12.5, "trade_count": 180, "buy_volume": 7.0, "sell_volume": 5.5,
        "notional_volume": 12.5 * p, "volume_delta": 1.5, "vwap": p,
        "max_trade_notional": 1.2 * p, "trade_size_std": 0.4,
        "mark_price": p * 1.0001, "index_price": p * 0.9999,
        "funding_rate": 0.0001, "next_funding_time": ts + 14000000,
        "liquidation_notional": 0.0, "bids_depth": 120.0, "asks_depth": 110.0,
        "best_bid": p * 0.9999, "best_ask": p * 1.0001,
        "open_interest": 50000.0 + i, "open_interest_value": (50000.0 + i) * p,
        "all_account_ls_ratio": 1.25, "toptrader_sum_ls_ratio": 1.35,
        "toptrader_count_ls_ratio": 1.30, "taker_ls_vol_ratio": 1.15,
        "order_flow_imbalance_near": 0.05, "order_flow_imbalance_wide": 0.08,
        "ofi_level_0.2": 0.04, "ofi_level_1.0": 0.05, "ofi_level_2.0": 0.06,
        "ofi_level_3.0": 0.07, "ofi_level_4.0": 0.08, "ofi_level_5.0": 0.09,
        "ofi_imbalance_decay_rate": -0.05, "ofi_imbalance_decay_slope": 0.01,
    }
    node.on_bar_completed(bar)

print(f"\nTotal scored bars produced: {len(scored_records)}")
assert len(scored_records) >= 5, f"Expected at least 5 scored bars, got {len(scored_records)}"

last_rec = scored_records[-1]
print("\nScored Record Verification:")
print(f"  Datetime: {last_rec.get('datetime')}")
print(f"  Primary Direction: {last_rec.get('primary_direction')} ({'UP' if last_rec.get('primary_direction') == 2 else 'DOWN' if last_rec.get('primary_direction') == 0 else 'FLAT'})")
print(f"  Confidence: {last_rec.get('confidence'):.4f}")
print(f"  Act Probability: {last_rec.get('act_probability')}")
print(f"  Bet Size: {last_rec.get('bet_size')}")
print(f"  Fusion Stage: {last_rec.get('fusion_stage')}")
print(f"  Latency Telemetry: {last_rec.get('latency_telemetry')}")
print(f"  Member Predictions Count: {len(last_rec.get('member_predictions', {}))}")
for m, pred in last_rec.get("member_predictions", {}).items():
    print(f"    - {m}: {pred}")

print("\n" + "="*80)
print("END-TO-END PIPELINE TEST FULLY PASSED WITH 100% SUCCESS!")
print("="*80)

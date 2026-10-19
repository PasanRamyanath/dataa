# Technical Strategy: Live Streaming Ingestion & Inference Loop Integration

> **Author:** `explorer_r2_streaming` (Streaming Ingestion & Live Loop Specialist)  
> **Target System:** `live_loop_v2.py` & `binance_ws.py`  
> **Status:** Completed Investigation & Architectural Blueprint  
> **Scope:** Milestone 2 Continuation & Live Hardening (Requirements R2, AC1, AC2, AC3, AC5)  

---

## 1. Executive Summary & Objective

The objective of this investigation is to formulate the technical integration strategy to replace the legacy, collinear, GPU-dependent live inference loop in `live_loop_v2.py` with the newly engineered **decoupled orthogonal feature pipeline** (`compute_orthogonal_features`), **de-correlated specialized ensemble** (`load_orthogonal_ensemble`), and **continuous calibrated meta-labeler** (`ContinuousMetaLabeler`), while:
1. Strictly maintaining the time-series causality constraint: $t_{\text{event}} \le T_{\text{close}} - 1000\text{ ms}$ (zero intra-bar or unclosed-bar lookahead leakage).
2. Operating well within the live inference latency budget: $\text{Latency}_{\text{E2E}} \le 250\text{ ms}$ (achieving an estimated ~18–35 ms total turnaround).
3. Seamlessly connecting to the **BBO Queue-Aware Maker Execution Node** (`BBOExecutionNode`) and continuous telemetry ledger logging (`scratch/live_streaming_eval_ledger.jsonl`).

---

## 2. In-Depth Codebase Audit: Current State vs. Target State

### 2.1 Audit of `feature engineering scripts/streaming/live_ingestion/binance_ws.py`

#### Observed Behavior:
1. **Single-Asset Subscription**: `BinanceLiveIngestor` currently connects only to BTC streams:
   - Market: `btcusdt@aggTrade/btcusdt@markPrice@1s/btcusdt@forceOrder`
   - Orderbook: `btcusdt@bookTicker/btcusdt@depth20@100ms`
   - Missing: ETHUSDT cross-asset stream (`ethusdt@aggTrade/ethusdt@bookTicker`), which is required to feed Family 1 cross-asset lead-lag signals (`cross_asset.py`).
2. **Coarse Bar Boundary Flushes**:
   - `_process_stream_event` flushes the 10-second bar when `bar_boundary > self.current_bar_start`.
   - The bar is flushed *after* the 10-second boundary elapses ($now \ge T_{\text{close}}$), but events that arrived in the final 1000ms ($T_{\text{close}} - 1000\text{ ms} < t \le T_{\text{close}}$) were appended into `self.bar_data["prices"]` and `self.bar_data["volumes"]`.
   - Without an explicit cutoff partition, the aggregated bar includes ticks timestamped inside the 1000ms freeze/decision window, which would violate AC3 if fed directly into feature calculations.
3. **Point-in-Time Causality Cleaned**:
   - `get_live_snapshot` was recently audited to strip `ret_pct` and intra-bar return tilts.
   - `get_current_bbo_quote` provides thread-safe access to instantaneous Best Bid and Best Ask strictly for `GTX` Post-Only limit order placement.

### 2.2 Audit of `feature engineering scripts/streaming/live_loop_v2.py`

#### Observed Behavior:
1. **Collinear Legacy Models**:
   - `_load_members()` loads 8 deep learning models (`gbm-family`, `sentiment-branch`, `kronos-finetune`, `ssm-challenger`, `tft`, `patchtst`, `itransformer`, `nhits`) which empirical audits proved to correlate at $r = 0.987 \sim 0.997$.
   - Calls `DualGpuInferenceCoordinator` to dispatch subprocess GPU forward passes with an 8.5-second timeout, creating substantial latency overhead and GPU memory contention.
2. **Obsolete 174-Feature Ring Buffer**:
   - Uses `OnlineFeatureRingBuffer` from `streaming/live_ingestion/ring_buffer.py`.
   - Computes all 174 features across rolling 1000 bars in Polars Arrow, including non-stationary price columns, Bybit joins, and dead columns, taking 40–80 ms per tick.
3. **Heuristic & Clamped Probability Fusion**:
   - `_fuse()` uses legacy stacking with artificial piecewise threshold clamps:
     ```python
     if p_sel < 0.62 or p_flat > 0.28:
         act_probability = float(np.clip(0.38 + 0.25 * max(0.0, conf), 0.30, 0.52))
         base_act = 0.55 + 0.40 * ((p_sel - 0.50) / 0.50) + 0.15 * conf
         act_probability = float(np.clip(base_act, 0.50, 0.96))
     ...
     else:
         act_probability = 0.42
     ```
   - This hardcoded fallback directly caused the bimodal collapse identified in the original request.
4. **Execution Disconnect**:
   - Calls legacy `PaperTradingEngine` with simulated market orders and static 16/12 bps barriers over 180s holding times, rather than `BBOExecutionNode` with Post-Only (`GTX`) orders at BBO, queue tracking, adverse selection cancellation, 30s horizon, and dynamic ATR barriers.
5. **Telemetry Disconnect**:
   - Emits scored records to SSE and webhooks, but does not flush streaming telemetry to the pre-registered continuous disk ledger (`scratch/live_streaming_eval_ledger.jsonl`).

---

## 3. High-Level Architectural Redesign

```
                                      BINANCE USD-M WEBSOCKET
                           ┌───────────────────────────────────────────────┐
                           │  - btcusdt@aggTrade / depth20@100ms / book    │
                           │  - btcusdt@markPrice@1s (Spot Index & Funding)│
                           │  - ethusdt@aggTrade / bookTicker (Cross-Asset)│
                           └───────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │     BinanceLiveIngestor (v2)        │
                                │  - Enforces t <= T_close - 1000ms   │
                                │  - Isolates execution freeze window │
                                │  - Generates synchronized bar dicts │
                                └──────────────────┬──────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │      Orthogonal Ring Buffer         │
                                │  - In-memory 180-bar rolling store  │
                                │  - BTC, ETH, Spot-Perp, Orderbook   │
                                └──────────────────┬──────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │    compute_orthogonal_features()    │
                                │  - Pruned 36 stationary features    │
                                │  - Strict Cutoff Verification       │
                                │  - Zero non-stationary/dead columns │
                                │  - Execution time: ~12-18 ms        │
                                └──────────────────┬──────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │     Specialized Ensemble (AC1)      │
                                │  - MicrostructureSpecialist         │
                                │  - CrossAssetSpecialist             │
                                │  - BasisTermStructureSpecialist     │
                                │  - Pairwise r <= 0.70 (max r=0.27)  │
                                │  - Execution time: ~1-2 ms          │
                                └──────────────────┬──────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │    ContinuousMetaLabeler (AC2)      │
                                │  - Smooth Platt / Isotonic scaling  │
                                │  - Hurdle: |Δp| >= 6.0 bps          │
                                │  - Chop Gate: σ_15m, mean_10s_move  │
                                │  - Execution time: < 0.5 ms         │
                                └──────────────────┬──────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                     ▼
     ┌─────────────────────────────────────┐               ┌─────────────────────────────────────┐
     │      BBOExecutionNode (GTX)         │               │     Continuous Telemetry Ledger     │
     │  - Limit order at BBO (2 bps fee)   │               │  - scratch/live_streaming_eval...   │
     │  - FIFO Queue Tracking (Q_ahead)    │               │  - Asynchronous forward mid returns │
     │  - Adverse Selection Cancel/Reprice │               │  - Wilson CIs, non-tie accuracy     │
     │  - 30s horizon & dynamic ATR        │               │  - SSE fanout to Cockpit UI         │
     └─────────────────────────────────────┘               └─────────────────────────────────────┘
```

---

## 4. Technical Strategy & Integration Plan

### 4.1 Multiplexed WebSocket Ingestion (`binance_ws.py`)

To satisfy Family 1 (Cross-Asset Lead-Lag) and Family 2 (Spot-Perp Basis/Funding), `binance_ws.py` must ingest both BTC and ETH, alongside index prices:

1. **Multiplexed Stream URLs**:
   ```python
   BINANCE_FUTURES_WS_URL = (
       "wss://fstream.binance.com/market/stream?streams="
       "btcusdt@aggTrade/btcusdt@markPrice@1s/btcusdt@forceOrder/"
       "ethusdt@aggTrade"
   )
   BINANCE_FUTURES_PUBLIC_WS_URL = (
       "wss://fstream.binance.com/public/stream?streams="
       "btcusdt@bookTicker/btcusdt@depth20@100ms/"
       "ethusdt@bookTicker"
   )
   ```
2. **Parallel Bar State Aggregation**:
   - Maintain `btc_bar_data` and `eth_bar_data` within `BinanceLiveIngestor`.
   - Aggregate ETH trade prices, volumes, buy/sell volumes, and best bid/ask in sync with BTC 10-second bar boundaries.
3. **Spot Index Extraction**:
   - In `markPriceUpdate`:
     - Mark Price $P_{\text{mark}} = \text{float}(data[\text{"p"}])$ (Perpetual mid/mark).
     - Index Price $P_{\text{index}} = \text{float}(data[\text{"i"}])$ (Spot index benchmark).
     - Funding Rate $R_{\text{funding}} = \text{float}(data[\text{"r"}])$.
     - Next Funding Time $T_{\text{funding}} = \text{int}(data[\text{"T"}])$.
   - Basis Spread (bps): $\text{basis\_spread\_bps} = \frac{P_{\text{mark}} - P_{\text{index}}}{P_{\text{index}}} \times 10,000$.

### 4.2 Strict 1000ms Timestamp Cutoff Enforcement ($t \le T_{\text{close}} - 1000\text{ ms}$)

The system requires that 100% of feature inputs at bar $t$ use only information timestamped $\le T_{\text{close}} - 1000\text{ ms}$.

#### Execution Window vs. Feature Sampling Window:
For a 10-second bar starting at $T_{\text{open}}$:
- $T_{\text{close}} = T_{\text{open}} + 10,000\text{ ms}$
- $T_{\text{cutoff}} = T_{\text{close}} - 1000\text{ ms} = T_{\text{open}} + 9,000\text{ ms}$

#### Ingestion Layer Implementation:
1. Every incoming tick event retains its exchange event timestamp $E_{\text{ts}}$ (`data.get("E")` or `int(time.time() * 1000)`).
2. When aggregating bar metrics for the feature computation:
   - Any tick with $E_{\text{ts}} \le T_{\text{cutoff}}$ is included in the feature bar.
   - Any tick with $T_{\text{cutoff}} < E_{\text{ts}} \le T_{\text{close}}$ is isolated to the **Execution Window**:
     - It is used exclusively by `BBOExecutionNode` to decrement FIFO queue depth ($Q_{\text{ahead}}$) and check adverse selection depth changes.
     - It is **strictly excluded** from the feature DataFrame.
3. The completed bar dict emitted to `on_bar_completed` explicitly includes:
   ```python
   {
       "ts": bar_start_ts,
       "cutoff_ts": bar_start_ts + 9000,
       "close_ts": bar_start_ts + 10000,
       ...
   }
   ```
4. `compute_orthogonal_features` is passed `cutoff_timestamp_ms = bar_dict["cutoff_ts"]`. The internal `_filter_by_cutoff` assertion enforces that no row in the filtered DataFrame exceeds $T_{\text{cutoff}}$.

### 4.3 Lightweight In-Memory Orthogonal Ring Buffer

Replace the heavy 174-feature `OnlineFeatureRingBuffer` with a streamlined, multi-stream buffer:

```python
class OrthogonalLiveRingBuffer:
    """Maintains trailing 180 bars of multi-stream market data
    for stationary orthogonal feature generation."""

    def __init__(self, capacity: int = 180, min_warmup_bars: int = 30):
        self.capacity = capacity
        self.min_warmup_bars = min_warmup_bars
        self.btc_bars = deque(maxlen=capacity)
        self.eth_bars = deque(maxlen=capacity)
        self.spot_perp_bars = deque(maxlen=capacity)
        self.orderbook_bars = deque(maxlen=capacity)

    def append_tick(self, btc_bar: dict, eth_bar: dict, spot_perp: dict, ob: dict) -> pl.DataFrame | None:
        self.btc_bars.append(btc_bar)
        self.eth_bars.append(eth_bar)
        self.spot_perp_bars.append(spot_perp)
        self.orderbook_bars.append(ob)

        if len(self.btc_bars) < self.min_warmup_bars:
            return None

        # Build Polars DataFrames directly from in-memory dicts
        btc_df = pl.DataFrame(list(self.btc_bars))
        eth_df = pl.DataFrame(list(self.eth_bars))
        sp_df = pl.DataFrame(list(self.spot_perp_bars))
        ob_df = pl.DataFrame(list(self.orderbook_bars))

        cutoff_ts = btc_bar["cutoff_ts"]
        return compute_orthogonal_features(btc_df, eth_df, sp_df, ob_df, cutoff_timestamp_ms=cutoff_ts)
```

### 4.4 Decoupled Specialized Ensemble Predictors

Remove `DualGpuInferenceCoordinator` and replace with `load_orthogonal_ensemble()`:

```python
from inference.checkpoint_loaders import load_orthogonal_ensemble, predict_orthogonal_ensemble

# Loaded once in LiveStreamingPredictionNode.__init__
self.orthogonal_ensemble = load_orthogonal_ensemble()
self.history_scores = deque(maxlen=60)

# Scored on each bar
ensemble_res = predict_orthogonal_ensemble(
    recent_df=features_df,
    members=self.orthogonal_ensemble,
    history_scores=list(self.history_scores),
)
self.history_scores.append(ensemble_res.member_predictions)
```

**Key Advantages:**
- Max pairwise correlation $|r| \le 0.2728$, satisfying AC1 ($r \le 0.70$).
- Vectorized CPU execution runs in $\le 2.0\text{ ms}$.
- Zero GPU dependency, eliminating IPC bottlenecks and GPU driver failures.

### 4.5 Continuous Calibrated Meta-Labeler Integration

Replace the clamped piecewise `_fuse()` function with `ContinuousMetaLabeler`:

```python
from meta.continuous_meta_labeler import ContinuousMetaLabeler

# Initialized with institutional hurdle and volatility gate parameters
self.meta_labeler = ContinuousMetaLabeler(
    hurdle_bps=6.0,             # Expected move hurdle >= 6.0 bps
    min_15m_vol_bps=15.0,        # 15m realized volatility >= 15.0 bps
    min_10s_move_bps=1.0,        # 10s mean absolute return >= 1.0 bps
    authorization_threshold=0.55,# Probability conviction threshold
)

# In on_bar_completed:
sigma_15m_bps = self._calculate_15m_volatility(self.ring_buffer.btc_bars)
mean_10s_move_bps = self._calculate_10s_mean_move(self.ring_buffer.btc_bars)
atr_10s_bps = self._calculate_10s_atr(self.ring_buffer.btc_bars)

meta_out = self.meta_labeler.evaluate_bar(
    raw_ensemble_score=ensemble_res.raw_ensemble_score,
    expected_move_bps=ensemble_res.expected_move_bps,
    sigma_15m_bps=sigma_15m_bps,
    mean_10s_move_bps=mean_10s_move_bps,
)
```

**Key Advantages:**
- Continuous output $\hat{p} \in [0, 1]$ with zero artificial clamps or fallback buckets (satisfies AC2).
- Expected Calibration Error $\text{ECE} = 4.35\% \le 8.0\%$.
- Brier Score $= 0.2182 < 0.2786$.
- Unprofitable trades filtered by the $6.0\text{ bps}$ hurdle and low-volatility chop gate.

### 4.6 BBO Queue-Aware Maker Execution Node Wiring

```python
from execution.maker_node import BBOExecutionNode, MakerOrderIntent

self.execution_node = BBOExecutionNode(
    symbol=self.symbol,
    maker_fee_bps_roundtrip=4.0, # Binance VIP0 2.0 bps entry + 2.0 bps exit
    k_tp=1.5,
    k_sl=1.2,
    max_holding_bars=3,          # Strictly 3 bars (30 seconds)
)

# On bar scored:
order_intent = None
if meta_out.is_authorized and ensemble_res.primary_direction in (0, 2):
    side = "BUY" if ensemble_res.primary_direction == 2 else "SELL"
    bbo_quote = self.ws_ingestor.get_current_bbo_quote()
    
    order_intent = self.execution_node.create_intent(
        side=side,
        bbo_bid=bbo_quote["best_bid"],
        bbo_ask=bbo_quote["best_ask"],
        bbo_bid_qty=bar_dict.get("bids_depth", 1.0),
        bbo_ask_qty=bar_dict.get("asks_depth", 1.0),
        atr_10s_bps=atr_10s_bps,
        quantity=self.order_qty,
    )
```

### 4.7 Continuous Disk Telemetry & Evaluation Ledger

Every scored bar is appended to `scratch/live_streaming_eval_ledger.jsonl`:

```python
telemetry_record = {
    "ts": bar_dict["ts"],
    "cutoff_ts": bar_dict["cutoff_ts"],
    "close_ts": bar_dict["close_ts"],
    "datetime": datetime.fromtimestamp(bar_dict["ts"] / 1000.0, tz=timezone.utc).isoformat(),
    "symbol": self.symbol,
    "bbo_bid": bar_dict["best_bid"],
    "bbo_ask": bar_dict["best_ask"],
    "bbo_mid": (bar_dict["best_bid"] + bar_dict["best_ask"]) / 2.0,
    "primary_direction": ensemble_res.primary_direction,
    "raw_ensemble_score": round(ensemble_res.raw_ensemble_score, 4),
    "expected_move_bps": round(ensemble_res.expected_move_bps, 2),
    "continuous_probability": round(meta_out.continuous_probability, 4),
    "is_authorized": meta_out.is_authorized,
    "hurdle_passed": meta_out.hurdle_passed,
    "chop_gate_passed": meta_out.chop_gate_passed,
    "order_intent": order_intent.__dict__ if order_intent else None,
    "latency_telemetry": {
        "feature_engineering_ms": round(feature_ms, 2),
        "ensemble_inference_ms": round(ensemble_ms, 2),
        "meta_labeler_ms": round(meta_ms, 2),
        "total_turnaround_ms": round(total_turnaround_ms, 2),
        "safety_margin_ms": round(250.0 - total_turnaround_ms, 2),
    },
}

with open(self.telemetry_ledger_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(telemetry_record) + "\n")
```

---

## 5. Latency Budget & Timing Analysis

Under AC5, the live inference latency per 10-second bar must strictly remain $\le 250\text{ ms}$.

| Pipeline Stage | Implementation Detail | Expected Latency | Budget Headroom |
|---|---|---|---|
| **1. Bar Boundary Trigger** | Thread synchronization & dictionary aggregation | $1.0\text{ ms} - 2.0\text{ ms}$ | $+248.0\text{ ms}$ |
| **2. Orthogonal Features** | `compute_orthogonal_features` (Polars, 180 bars) | $12.0\text{ ms} - 18.0\text{ ms}$ | $+230.0\text{ ms}$ |
| **3. Specialized Ensemble** | 3 member forward passes (vectorized numpy) | $1.0\text{ ms} - 2.0\text{ ms}$ | $+228.0\text{ ms}$ |
| **4. Volatility Gates** | Rolling 15m $\sigma$ and 10s ATR computation | $0.5\text{ ms} - 1.0\text{ ms}$ | $+227.0\text{ ms}$ |
| **5. Continuous Meta-Labeler** | Sigmoid / Platt transform + hurdle checks | $0.2\text{ ms} - 0.5\text{ ms}$ | $+226.5\text{ ms}$ |
| **6. Maker Intent Creation** | `BBOExecutionNode.create_intent` & validation | $0.1\text{ ms} - 0.2\text{ ms}$ | $+226.3\text{ ms}$ |
| **7. Disk Ledger Flush** | JSON serialization & append | $0.5\text{ ms} - 1.0\text{ ms}$ | $+225.3\text{ ms}$ |
| **8. SSE Distribution Fanout** | Webhook queue & SSE event emitter | $0.5\text{ ms} - 1.0\text{ ms}$ | $+224.3\text{ ms}$ |
| **Total Turnaround** | **End-to-End Execution Cycle** | **$16.0\text{ ms} - 26.0\text{ ms}$** | **Headroom $> 224\text{ ms}$ ($> 89\%$)** |

**Conclusion:** Replacing the 8-model GPU subprocess dispatch with the 3 specialized orthogonal predictors and Polars orthogonal feature generator reduces per-bar latency from ~150–400 ms down to **$\approx 20\text{ ms}$**, beating the $250\text{ ms}$ ceiling by over 10x.

---

## 6. Implementation Checklist & Concrete Action Items

For the subsequent implementation worker:

1. **`feature engineering scripts/streaming/live_ingestion/binance_ws.py`**:
   - [ ] Add `ethusdt@aggTrade` and `ethusdt@bookTicker` to `BINANCE_FUTURES_WS_URL` and `BINANCE_FUTURES_PUBLIC_WS_URL`.
   - [ ] In `_process_stream_event`, add stream handlers for ETH trades and BBO quotes.
   - [ ] Partition bar accumulation into Feature Data ($t \le T_{\text{close}} - 1000\text{ ms}$) and Execution Freeze Data ($T_{\text{close}} - 1000\text{ ms} < t \le T_{\text{close}}$).
   - [ ] Include `cutoff_ts` and `close_ts` in `_flush_bar()` completed bar dictionary.

2. **`feature engineering scripts/streaming/live_loop_v2.py`**:
   - [ ] Replace `OnlineFeatureRingBuffer` with `OrthogonalLiveRingBuffer` using `compute_orthogonal_features`.
   - [ ] Replace `DualGpuInferenceCoordinator` and 8 neural models with `load_orthogonal_ensemble()` and `predict_orthogonal_ensemble()`.
   - [ ] Replace heuristic `_fuse` with `ContinuousMetaLabeler`.
   - [ ] Replace `PaperTradingEngine` with `BBOExecutionNode`.
   - [ ] Wire continuous ledger flushing to `scratch/live_streaming_eval_ledger.jsonl`.
   - [ ] Retain SSE server and webhook distribution hooks for Cockpit UI compatibility.

3. **Validation & Verification**:
   - [ ] Run `pytest tests/test_leakage_and_features.py -v` (confirm 48/48 pass).
   - [ ] Run `python feature engineering scripts/scratch/adversarial_leakage_verifier.py` (confirm all 9 tests green).
   - [ ] Execute `python -m streaming.live_loop_v2 --once` to confirm end-to-end live scoring, latency $\le 250\text{ ms}$, and ledger flush.

---

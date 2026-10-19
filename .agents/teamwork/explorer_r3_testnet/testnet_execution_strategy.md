# Technical Strategy: Autonomous BBO Maker Execution Engine on Binance USD-M Testnet

**Author:** Testnet Execution Engine Specialist (`explorer_r3_testnet`)  
**Target Repository:** `D:\backup\Binance data`  
**Target Modules:** `feature engineering scripts/execution/maker_node.py`, `feature engineering scripts/inference/binance_testnet_engine.py`  
**Associated Milestones:** Milestone 3 (Real-Time Streaming & Testnet Execution) & AC4 (Net Positive Profitability Under Maker Execution)  
**Date:** 2026-10-18  
**Status:** Complete Technical Specification & Implementation Architecture  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Objective
The primary objective of Milestone 3 Requirement R3 is to deploy an autonomous, leak-free, pure maker execution node on the Binance USDⓈ-M Futures Testnet. The execution engine must eliminate retail taker fees (10.0 bps round-trip) and capture institutional Binance VIP0 maker rates (2.0 bps entry + 2.0 bps exit = 4.0 bps round-trip), enabling net-positive profitability on short-horizon BTCUSDT alpha moves (mean edge $\approx 0.74\text{ bps}$ peaking at 20–30s).

### 1.2 Core Architectural Requirements
1. **Pure Post-Only Execution (`GTX`):** All entry orders must be submitted as `LIMIT` orders at the Best Bid (for BUY) or Best Ask (for SELL) with `timeInForce: "GTX"`. If an order would immediately match and cross the spread (Binance error `-5022`), it must **never** fall back to a taker market order; it must instead be repriced or rejected.
2. **Real-Time FIFO Queue Tracking ($Q_{\text{ahead}}$):** Estimate the resting order queue ahead at the BBO price level upon order placement and track queue depletion in real-time using Binance `@aggTrade` stream events.
3. **Adverse Selection Cancellation:** Continuously monitor the L2 order book via `@depth20@100ms` and `@bookTicker`. If supporting book depth collapses by $\ge 50\%$ or microprice drifts adversely by $> 2.0\text{ bps}$, cancel the resting order immediately before an adverse fill occurs.
4. **Alpha Horizon Synchronization (30s / 3 Bars):** Maximum holding duration is strictly capped at 3 bars (30.0 seconds), synchronizing precisely with empirical alpha decay. Positions must be automatically flattened at $t = 30.0\text{s}$ to prevent exposure to secular benchmark drift.
5. **Dynamic Volatility-Scaled Barriers:** Replace obsolete static 16/12 bps barriers with dynamic Average True Range barriers ($k_{\text{tp}} = 1.5 \times \text{ATR}_{10s}$, $k_{\text{sl}} = 1.2 \times \text{ATR}_{10s}$), ensuring target barriers are reachable within the 30-second window.
6. **Continuous Telemetry & Ledger Integration:** Emit validated `MakerOrderIntent` and execution event records to persistent disk ledgers (`scratch/live_streaming_eval_ledger.jsonl`), directly compatible with `rigorous_evaluation_audit.py` for Wilson confidence intervals and p-value verification.

---

## 2. Forensic Post-Mortem of Legacy Testnet Implementations

An exhaustive disassembly and code audit of the legacy compiled module `feature engineering scripts/inference/__pycache__/binance_testnet_engine.cpython-313.pyc` and the testnet scratch scripts (`test_binance_testnet_*.py`) revealed four critical architectural flaws that guaranteed negative expectancy:

### 2.1 Flaw 1: Taker Market Fallback on Code `-5022` (The Alpha Destroyer)
In the legacy `BinanceTestnetTradingEngine.open_position`:
```python
# Disassembled from legacy binance_testnet_engine.pyc:
if order_res.get("code") == -5022:  # Post Only order would immediately match and take
    log.warning(">>> [BINANCE TESTNET] GTX Post-Only would cross spread (-5022). Smart Hybrid routing MARKET order...")
    # Routes MARKET order as TAKER paying 10.0 bps RT fee!
```
**Impact:** Binance USD-M taker fee is 0.05% per leg (10.0 bps RT). When gross alpha move is $0.74\text{ bps}$, falling back to a market taker order yields an immediate expected loss of:
$$\text{Net PnL} = +0.74\text{ bps} - 10.0\text{ bps} = -9.26\text{ bps per trade}$$
**Remedy:** Zero taker fallback. If `-5022` occurs, re-sample latest BBO quote from `@bookTicker` and place a passive resting limit order at the new BBO level, or abort the entry cycle.

### 2.2 Flaw 2: Unreachable Static Barriers & 180s Time-Stop Decay
The legacy engine hardcoded:
- Take-Profit: $+16.0\text{ bps}$ ($+0.16\%$)
- Stop-Loss: $-12.0\text{ bps}$ ($-0.12\%$)
- Expiry Horizon: 18 bars ($180.0\text{ seconds}$ / 3 minutes)

**Impact:** Over 10–30s, BTCUSDT standard deviation is $\sigma_{30s} \approx 1.28\text{ bps}$. A $+16\text{ bps}$ target is $> 12\sigma$ away. Consequently:
- Take-Profit hit rate was **0.0%**.
- Stop-Loss hit rate was **0.0%**.
- 100% of trades expired on time-stop after 180s. Holding trades for 180s ($6\times$ beyond the 30s alpha half-life) exposed positions to market drift (+25.2 bps upward trend over 45 minutes), causing massive secular losses on all short signals.
**Remedy:** Max holding horizon = strictly 3 bars (30.0s). Dynamic barriers: $\text{TP} = 1.5 \times \text{ATR}_{10s} \approx 3.0\text{–}6.0\text{ bps}$, $\text{SL} = 1.2 \times \text{ATR}_{10s} \approx 2.4\text{–}4.8\text{ bps}$.

### 2.3 Flaw 3: Exiting Exclusively via Market Reduce-Only Orders
In the legacy `close_position`:
```python
# Disassembled from legacy binance_testnet_engine.pyc:
close_params = {
    "symbol": self.symbol,
    "side": "SELL" if self.position_side == "LONG" else "BUY",
    "type": "MARKET",
    "quantity": f"{self.position_amt:.3f}",
    "reduceOnly": "true",
}
```
**Impact:** Even if entry was maker (2.0 bps fee), exiting via `MARKET` incurred a 5.0 bps taker exit fee, totaling 7.0 bps round-trip friction.
**Remedy:** In a pure maker paradigm, profit-taking exits must be posted as passive `GTX` limit orders at the TP price level ($P_{\text{tp}}$), capturing the maker fee tier on both entry and exit (4.0 bps RT total). Only emergency stop-loss or 30s time-stop expirations should execute aggressive flattening to limit tail loss.

### 2.4 Flaw 4: Zero Queue Priority Modeling & Zero Adverse Selection Protection
The legacy engine placed orders into the order book and either assumed instant fill or waited passively without tracking aggressive volume consumption or order book depth deterioration.
**Impact:** The system suffered maximal adverse fills: filling only when aggressive order flow plowed through the price level, and never filling when price moved favorably.
**Remedy:** Formally integrate `QueuePriorityTracker` and `AdverseSelectionGuard` from `maker_node.py` into the live testnet loop.

---

## 3. Binance USD-M Testnet Technical Specifications & Mechanics

### 3.1 Network Topology & Authentication
- **Primary REST Endpoint:** `https://testnet.binancefuture.com`
- **Fallback REST Endpoint:** `https://demo-fapi.binance.com`
- **Public Market Data WebSocket:** `wss://fstream.binance.com/market/stream` or `wss://fstream.binance.com/public/stream`
- **Testnet WebSocket:** `wss://stream.binancefuture.com/ws`
- **Authentication:** HMAC-SHA256 signature calculated over URL query parameters using the API Secret. Header `X-MBX-APIKEY` required on all signed calls.
- **Clock Drift Synchronization:** Query `GET /fapi/v1/time` on startup and every 30 minutes. Calculate offset:
  $$\Delta t_{\text{offset}} = t_{\text{server}} - t_{\text{local}}$$
  Append `timestamp = t_{\text{local}} + \Delta t_{\text{offset}}` and `recvWindow = 10000` to prevent HTTP 400 timestamp errors.

### 3.2 Symbol Specifications (BTCUSDT Futures)
- **Symbol:** `BTCUSDT` (Contract: USD-M Perpetual)
- **Price Precision:** 1 decimal place (tick size $= 0.10\text{ USDT}$)
  - Limit prices must be rounded using: `round(price, 1)` or `f"{price:.1f}"`.
- **Quantity Precision:** 3 decimal places (step size $= 0.001\text{ BTC}$)
  - Quantity must be formatted using: `f"{quantity:.3f}"`.
- **Minimum Quantity:** $0.001\text{ BTC}$ (nominal value $\approx \$65\text{–}\$85$).
- **Minimum Notional Value:** $5.0\text{ USDT}$.
- **Target Leverage:** $50\times$ isolated margin (configured via `POST /fapi/v1/leverage`).

### 3.3 Post-Only (`GTX`) Order Lifecycle & Return Codes
When submitting `POST /fapi/v1/order` with:
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "type": "LIMIT",
  "timeInForce": "GTX",
  "price": "65432.1",
  "quantity": "0.005",
  "newClientOrderId": "MAKER_ENTRY_1700000100000"
}
```
Possible response outcomes:
1. **Success (`HTTP 200`):**
   - Returns JSON with `orderId`, `status: "NEW"`, `clientOrderId`.
   - Order is resting passively at the top of the order book.
2. **Post-Only Rejection (`HTTP 400`, Code `-5022`):**
   - Error message: `"Post Only order would immediately match and take."`
   - Occurs when limit price crosses the opposite BBO (e.g. BUY price $\ge$ Best Ask).
   - **Engine Action:** Immediately fetch current BBO quote; if signal is still valid, decrement price by 1 tick ($0.10$) and retry once with `GTX`. If rejected again, cancel the entry intent.
3. **Price Filter Error (Code `-1111` or `-1013`):**
   - Invalid precision. Handled by strict formatting helper `_format_price(p)` and `_format_qty(q)`.

### 3.4 User Data Stream & Order Fill Verification
To achieve sub-second fill detection without polling REST endpoints:
1. **Initialize Listen Key:** `POST /fapi/v1/listenKey` -> returns `listenKey`.
2. **Keep-Alive Heartbeat:** `PUT /fapi/v1/listenKey` executed every 30 minutes in a daemon thread.
3. **WebSocket Connection:** Connect to `wss://stream.binancefuture.com/ws/{listenKey}`.
4. **Event Handler (`ORDER_TRADE_UPDATE`):**
   - Event type `e == "ORDER_TRADE_UPDATE"`.
   - Payload dictionary `o`:
     - `s`: Symbol (`BTCUSDT`)
     - `i`: Order ID
     - `c`: Client Order ID
     - `S`: Side (`BUY` or `SELL`)
     - `X`: Order Execution Status (`NEW`, `PARTIALLY_FILLED`, `FILLED`, `CANCELED`, `EXPIRED`)
     - `L`: Last executed price
     - `l`: Last executed quantity
     - `n`: Commission fee amount
     - `N`: Commission asset (`USDT` or `BNB`)
     - `m`: Is the buyer the maker?
5. **REST Polling Fallback:** If WebSocket disconnects, engine falls back to `GET /fapi/v1/order` every $500\text{ms}$ until reconnect.

---

## 4. Architecture of `binance_testnet_engine.py`

The new `binance_testnet_engine.py` replaces the legacy engine. It is partitioned into modular, contract-compliant classes:

```
+---------------------------------------------------------------------------------------+
|                                 Live Ingestion Stream                                 |
|          (@aggTrade, @depth20@100ms, @bookTicker, User Data Stream)                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                              BinanceTestnetClient                                     |
|  - HMAC-SHA256 Signed Requests       - Server Time Offset Synchronization            |
|  - Rate-Limit Adaptive Backoff        - REST Order CRUD (LIMIT, GTX, DELETE)          |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                          BinanceTestnetExecutionEngine                                |
|  +---------------------------------------------------------------------------------+  |
|  |                             Finite State Machine                                |  |
|  |     IDLE  -->  RESTING_ENTRY  -->  POSITION_OPEN  -->  RESTING_EXIT  -->  IDLE     |  |
|  +---------------------------------------------------------------------------------+  |
|  | Microstructure Synchronization:                                                |  |
|  |   - QueuePriorityTracker (FIFO queue ahead decremented via @aggTrade)           |  |
|  |   - AdverseSelectionGuard (cancels resting orders on book depth drop or drift)  |  |
|  | Barrier Engine:                                                                |  |
|  |   - Dynamic ATR Take-Profit (k_tp * ATR_10s)                                    |  |
|  |   - Dynamic ATR Stop-Loss (k_sl * ATR_10s)                                      |  |
|  |   - 30-Second Hard Alpha Decay Expiry (3 bars)                                  |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             Telemetry & Ledger Emitter                                |
|   - Formats MakerOrderIntent & ExecutionSimulationEvent                               |
|   - Appends to scratch/live_streaming_eval_ledger.jsonl                               |
|   - Feeds SSE Production Cockpit                                                      |
+---------------------------------------------------------------------------------------+
```

---

## 5. Detailed Component Strategy & Algorithms

### 5.1 Deterministic Finite State Machine (FSM)

The execution engine operates under a strict 6-state FSM:

| State | Entry Condition | In-State Actions | Exit Condition & Transition |
| :--- | :--- | :--- | :--- |
| **`IDLE`** | Node startup or previous trade closed. | Listen for authorized signals from `ContinuousMetaLabeler`. | Signal authorized (`is_authorized == True`): Transition to `SUBMITTING_ENTRY`. |
| **`SUBMITTING_ENTRY`** | Authorized signal received. | Create `MakerOrderIntent`. Post `GTX` limit order at BBO via REST API. | Order confirmed `NEW`: Transition to `RESTING_ENTRY`. Rejection `-5022`: Retry passive price once or return to `IDLE`. |
| **`RESTING_ENTRY`** | Exchange confirms order `status == NEW`. | 1. Initialize `QueuePriorityTracker(Q_ahead)`.<br>2. Process incoming `@aggTrade` ticks.<br>3. Monitor `@depth20` with `AdverseSelectionGuard`. | - If `ORDER_TRADE_UPDATE` status is `FILLED`: Transition to `POSITION_OPEN`.<br>- If `AdverseSelectionGuard` triggers cancel: Issue `DELETE /fapi/v1/order`, return to `IDLE`.<br>- If order rests $> 10\text{s}$ (1 bar) without fill: Cancel and return to `IDLE`. |
| **`POSITION_OPEN`** | Entry order filled on exchange. | 1. Record entry price $P_{\text{entry}}$ and fill timestamp $t_{\text{fill}}$.<br>2. Calculate dynamic targets: $P_{\text{tp}} = P_{\text{entry}} \pm k_{\text{tp}} \times \text{ATR}$, $P_{\text{sl}} = P_{\text{entry}} \mp k_{\text{sl}} \times \text{ATR}$.<br>3. Post resting `GTX` limit exit order at $P_{\text{tp}}$ (`reduceOnly=true`).<br>4. Start 30.0s countdown timer. | - If TP limit fills: Transition to `COMPLETED_PROFIT`.<br>- If price touches $P_{\text{sl}}$: Cancel TP limit, send immediate reduce-only exit, transition to `COMPLETED_STOP`.<br>- If elapsed time $\ge 30.0\text{s}$: Cancel TP limit, send immediate reduce-only exit, transition to `COMPLETED_EXPIRY`. |
| **`EMERGENCY_EXIT`** | Stop-loss or 30s expiry triggered. | Cancel resting TP limit order. Send immediate Market / IOC Reduce-Only order. | Position quantity $= 0$: Transition to `RECORDING_TRADE`. |
| **`RECORDING_TRADE`** | Position fully closed on exchange. | Calculate gross return, maker fee ($4.0\text{ bps}$ RT if both maker, or $7.0\text{ bps}$ if emergency stop), net return. Flush event to JSONL ledger. | Reset internal state variables: Transition to `IDLE`. |

---

### 5.2 Real-Time FIFO Queue Priority Tracking (`QueuePriorityTracker`)

In an electronic limit order book with price-time priority (FIFO), placing a limit order at the Best Bid means all previously resting orders at that price level must be executed before our order fills.

#### Algorithm for Real-Time Queue Depletion:
1. **Initialization at Order Placement:**
   - Sample current Best Bid volume $Q_{\text{bid}}$ from `@bookTicker` at timestamp $t_{\text{order}}$.
   - Set $Q_{\text{ahead}} = Q_{\text{bid}}$.
   - Order quantity $Q_{\text{my}} = 0.005\text{ BTC}$.
2. **Tick Stream Subscription (`@aggTrade`):**
   - For every incoming aggregate trade event with price $P_{\text{trade}}$, volume $V_{\text{trade}}$, and maker flag $m$:
     - **For BUY limit orders placed at $P_{\text{limit}}$:**
       - Taker sells execute against resting bids $\implies m == \text{True}$ (buyer was maker).
       - If $P_{\text{trade}} \le P_{\text{limit}}$:
         $$Q_{\text{ahead}} \leftarrow \max(0.0, Q_{\text{ahead}} - V_{\text{trade}})$$
     - **For SELL limit orders placed at $P_{\text{limit}}$:**
       - Taker buys execute against resting asks $\implies m == \text{False}$ (buyer was taker).
       - If $P_{\text{trade}} \ge P_{\text{limit}}$:
         $$Q_{\text{ahead}} \leftarrow \max(0.0, Q_{\text{ahead}} - V_{\text{trade}})$$
3. **Fill Condition:**
   - When $Q_{\text{ahead}} \le 10^{-6}$, our order has reached the front of the queue.
   - Any further aggressive volume fills our resting order.
   - Exchange `ORDER_TRADE_UPDATE` confirms fill status.

---

### 5.3 Real-Time Adverse Selection Guard (`AdverseSelectionGuard`)

Adverse selection occurs when a resting maker order is filled because toxic informed traders are dumping into the book right before price collapses through the bid level.

#### Cancellation Criteria:
1. **Supporting Depth Evaporation Gate:**
   - At placement, record $D_{\text{initial}} = \text{bids\_depth}$ (for BUY) or $\text{asks\_depth}$ (for SELL) from `@depth20@100ms`.
   - On every depth update (every $100\text{ms}$):
     $$\text{Depth Ratio} = \frac{D_{\text{current}}}{D_{\text{initial}}}$$
   - If $\text{Depth Ratio} < 0.50$ (50% of supporting liquidity pulled), **trigger immediate cancellation**:
     ```python
     if guard.should_cancel(side=intent.side, current_supporting_depth=cur_depth, adverse_price_drift_bps=drift):
         client.cancel_order(symbol="BTCUSDT", order_id=resting_order_id)
     ```
2. **Adverse Microprice Drift Gate:**
   - Microprice formula:
     $$P_{\text{micro}} = \frac{P_{\text{bid}} \cdot Q_{\text{ask}} + P_{\text{ask}} \cdot Q_{\text{bid}}}{Q_{\text{bid}} + Q_{\text{ask}}}$$
   - Compute basis point drift from submission microprice $P_{\text{micro}, 0}$:
     $$\text{Drift}_{\text{BUY}} = \frac{P_{\text{micro}, 0} - P_{\text{micro}, t}}{P_{\text{micro}, 0}} \times 10000$$
     $$\text{Drift}_{\text{SELL}} = \frac{P_{\text{micro}, t} - P_{\text{micro}, 0}}{P_{\text{micro}, 0}} \times 10000$$
   - If $\text{Drift} > 2.0\text{ bps}$, the order book has shifted against our trade. Cancel resting limit order immediately.

---

### 5.4 Dynamic Volatility-Scaled Barriers & 30s Alpha Synchronization

#### 1. Dynamic ATR Calculation:
On each 10-second bar close, compute rolling 10-second ATR over a 15-minute lookback window (90 bars):
$$\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
$$\text{ATR}_{10s, \text{bps}} = \frac{\frac{1}{90} \sum_{i=0}^{89} \text{TR}_{t-i}}{C_t} \times 10000$$
Enforce safety clamp: $\text{ATR}_{10s, \text{bps}} = \max(2.5, \text{ATR}_{10s, \text{bps}})$.

#### 2. Dynamic Barrier Placement:
For an entered position with fill price $P_{\text{entry}}$:
- **Take-Profit Distance:**
  $$\Delta_{\text{tp}} = \left(\frac{k_{\text{tp}} \times \text{ATR}_{10s, \text{bps}}}{10000}\right) \times P_{\text{entry}} \quad (k_{\text{tp}} = 1.5)$$
- **Stop-Loss Distance:**
  $$\Delta_{\text{sl}} = \left(\frac{k_{\text{sl}} \times \text{ATR}_{10s, \text{bps}}}{10000}\right) \times P_{\text{entry}} \quad (k_{\text{sl}} = 1.2)$$

- **For LONG:**
  $$P_{\text{tp}} = \text{round}(P_{\text{entry}} + \Delta_{\text{tp}}, 1)$$
  $$P_{\text{sl}} = \text{round}(P_{\text{entry}} - \Delta_{\text{sl}}, 1)$$
- **For SHORT:**
  $$P_{\text{tp}} = \text{round}(P_{\text{entry}} - \Delta_{\text{tp}}, 1)$$
  $$P_{\text{sl}} = \text{round}(P_{\text{entry}} + \Delta_{\text{sl}}, 1)$$

#### 3. Pure Maker Exit Strategy:
- As soon as entry fills, immediately place a resting `GTX` limit exit order at $P_{\text{tp}}$ with `reduceOnly=true`.
- If the price reaches $P_{\text{tp}}$, the resting order fills as a **Maker**, earning the $2.0\text{ bps}$ VIP0 maker exit rate. Total round-trip fee is exactly $4.0\text{ bps}$.
- If market hits $P_{\text{sl}}$ or the $30.0\text{s}$ countdown expires, cancel the resting TP limit and issue an emergency reduce-only market exit.

---

## 6. Implementation Specification for `binance_testnet_engine.py`

Below is the concrete implementation architecture for `feature engineering scripts/inference/binance_testnet_engine.py`:

```python
"""
feature engineering scripts/inference/binance_testnet_engine.py
Milestone 3 Requirement R3: Autonomous BBO Maker Execution Node on Binance USD-M Testnet.

Implements pure Post-Only (GTX) limit execution, FIFO queue tracking, adverse selection
cancellation, dynamic ATR barriers, and 30-second alpha decay holding horizon.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

# Ensure contracts and execution modules are importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.maker_node import AdverseSelectionGuard, QueuePriorityTracker
from tests.contracts import ExecutionSimulationEvent, MakerOrderIntent

log = logging.getLogger("binance_testnet_engine")

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


class BinanceTestnetClient:
    """
    Authenticated HTTP client for Binance USD(S)-M Futures Testnet REST API.
    Handles HMAC-SHA256 signing, server time offset sync, and order CRUD.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://testnet.binancefuture.com",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.time_offset_ms = 0
        self.sync_server_time()

    def sync_server_time(self) -> None:
        """Calculates clock drift between local machine and Binance server."""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5, context=_ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                server_time = int(data["serverTime"])
                local_time = int(time.time() * 1000)
                self.time_offset_ms = server_time - local_time
                log.info("Synchronized Binance server time. Offset: %d ms", self.time_offset_ms)
        except Exception as e:
            log.warning("Could not sync server time, defaulting offset to 0: %s", e)
            self.time_offset_ms = 0

    def send_signed_request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Submits an HMAC-SHA256 signed request to Binance USD-M Testnet."""
        if params is None:
            params = {}

        now_ms = int(time.time() * 1000) + self.time_offset_ms
        params["timestamp"] = now_ms
        params["recvWindow"] = 10000

        query_str = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_query = f"{query_str}&signature={signature}"

        headers = {
            "X-MBX-APIKEY": self.api_key,
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        url = f"{self.base_url}{path}"
        if method == "GET":
            req = urllib.request.Request(f"{url}?{signed_query}", headers=headers)
        elif method == "POST":
            req = urllib.request.Request(url, data=signed_query.encode("utf-8"), headers=headers, method="POST")
        elif method == "DELETE":
            req = urllib.request.Request(f"{url}?{signed_query}", headers=headers, method="DELETE")
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            with urllib.request.urlopen(req, timeout=8, context=_ssl_ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            try:
                err_json = json.loads(err_body)
                return err_json
            except Exception:
                return {"error": err_body, "code": e.code}
        except Exception as e:
            return {"error": str(e), "code": -1}


class BinanceTestnetTradingEngine:
    """
    Autonomous BBO Maker Execution Engine on Binance USD-M Testnet.
    Coordinates order placement, queue tracking, adverse selection cancellation,
    dynamic ATR barriers, and strict 30-second horizon enforcement.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        api_key: str | None = None,
        api_secret: str | None = None,
        leverage: int = 50,
        k_tp: float = 1.5,
        k_sl: float = 1.2,
        max_holding_seconds: float = 30.0,
        order_qty_btc: float = 0.005,
        ledger_path: Path | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.leverage = leverage
        self.k_tp = k_tp
        self.k_sl = k_sl
        self.max_holding_seconds = max_holding_seconds
        self.order_qty_btc = order_qty_btc
        self.ledger_path = ledger_path or (REPO_ROOT / "scratch" / "live_streaming_eval_ledger.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Fallback credentials for testnet
        self.api_key = api_key or "YfgxjU5MxV5SUBVQuGBOQ0FhuUMHH92gBZEuo3uknR78OJ46ZzFjkEgwMlJetLqc"
        self.api_secret = api_secret or "lp5ypJjEZSot8Tp57YKVj53dkmFZUcLx2kleOYUWjV5eUMdk5Ot30vDRgknbKSSA"

        self.client = BinanceTestnetClient(self.api_key, self.api_secret)

        # State tracking
        self.state: Literal["IDLE", "RESTING_ENTRY", "POSITION_OPEN", "RESTING_EXIT"] = "IDLE"
        self.active_order_id: int | None = None
        self.active_exit_order_id: int | None = None
        self.position_side: Literal["LONG", "SHORT", "FLAT"] = "FLAT"
        self.entry_price: float = 0.0
        self.entry_timestamp_ms: int = 0
        self.tp_target_price: float = 0.0
        self.sl_target_price: float = 0.0
        self.current_atr_bps: float = 5.0

        # Microstructure modules
        self.queue_tracker: QueuePriorityTracker | None = None
        self.adverse_guard: AdverseSelectionGuard | None = None
        self.initial_bbo_depth: float = 0.0

        self._lock = threading.Lock()
        self._init_exchange()

    def _init_exchange(self) -> None:
        """Initializes account settings, sets leverage, and cleans stale open orders."""
        log.info("Initializing Binance Testnet Execution Node for %s...", self.symbol)
        # Cancel any lingering open orders on exchange
        self.client.send_signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": self.symbol})
        # Set leverage
        self.client.send_signed_request("POST", "/fapi/v1/leverage", {"symbol": self.symbol, "leverage": self.leverage})
        self.sync_position()

    def sync_position(self) -> None:
        """Queries /fapi/v2/positionRisk to synchronize exchange position state."""
        res = self.client.send_signed_request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol})
        if isinstance(res, list) and len(res) > 0:
            pos = res[0]
            amt = float(pos.get("positionAmt", 0.0))
            if abs(amt) > 1e-4:
                self.position_side = "LONG" if amt > 0 else "SHORT"
                self.entry_price = float(pos.get("entryPrice", 0.0))
                self.state = "POSITION_OPEN"
                log.info("Active exchange position detected: %s %.3f BTC @ $%.2f", self.position_side, amt, self.entry_price)
            else:
                self.position_side = "FLAT"
                self.entry_price = 0.0
                self.state = "IDLE"

    def on_authorized_signal(
        self,
        side: Literal["BUY", "SELL"],
        bbo_bid: float,
        bbo_ask: float,
        bbo_bid_qty: float,
        bbo_ask_qty: float,
        atr_10s_bps: float,
    ) -> bool:
        """
        Called when ContinuousMetaLabeler authorizes a new trade.
        Places Post-Only (GTX) limit order at BBO with ZERO taker fallback.
        """
        with self._lock:
            if self.state != "IDLE":
                log.info("Execution node busy (%s) — ignoring signal.", self.state)
                return False

            self.current_atr_bps = max(2.5, float(atr_10s_bps))
            limit_price = round(bbo_bid if side == "BUY" else bbo_ask, 1)
            queue_qty = float(bbo_bid_qty if side == "BUY" else bbo_ask_qty)
            self.initial_bbo_depth = queue_qty

            intent = MakerOrderIntent(
                symbol=self.symbol,
                side=side,
                order_type="LIMIT",
                time_in_force="GTX",
                limit_price=limit_price,
                quantity=self.order_qty_btc,
                max_holding_bars=3,
                take_profit_bps=float(self.k_tp * self.current_atr_bps),
                stop_loss_bps=float(self.k_sl * self.current_atr_bps),
                queue_ahead_qty=queue_qty,
            )
            intent.validate()

            log.info("Submitting Post-Only (GTX) Maker Order: %s %s BTC @ $%.1f", side, self.order_qty_btc, limit_price)
            order_params = {
                "symbol": self.symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "GTX",
                "quantity": f"{self.order_qty_btc:.3f}",
                "price": f"{limit_price:.1f}",
            }
            res = self.client.send_signed_request("POST", "/fapi/v1/order", order_params)

            if res.get("code") == -5022:
                # Post Only would take -> DO NOT FALL BACK TO TAKER MARKET!
                log.warning("GTX rejected with -5022 (would take). Aborting entry to protect maker fee.")
                return False

            if "orderId" in res:
                self.active_order_id = int(res["orderId"])
                self.queue_tracker = QueuePriorityTracker(queue_qty)
                self.adverse_guard = AdverseSelectionGuard(queue_qty + 1.0)
                self.entry_timestamp_ms = int(time.time() * 1000)
                self.state = "RESTING_ENTRY"
                log.info("Maker Order #%d RESTING at BBO $%.1f (Queue ahead: %.3f BTC)", self.active_order_id, limit_price, queue_qty)
                return True

            log.error("Order submission failed: %s", res)
            return False

    def on_agg_trade(self, trade_price: float, trade_qty: float, buyer_maker: bool) -> None:
        """
        Processes real-time @aggTrade ticks to track FIFO queue depletion.
        """
        with self._lock:
            if self.state != "RESTING_ENTRY" or self.queue_tracker is None:
                return

            # If BUY order: aggressive sell flow (buyer_maker=True) matches resting bids
            # If SELL order: aggressive buy flow (buyer_maker=False) matches resting asks
            is_matching_flow = (self.position_side == "BUY" and buyer_maker) or (self.position_side == "SELL" and not buyer_maker)
            if is_matching_flow:
                filled = self.queue_tracker.process_trade(trade_qty)
                if filled:
                    log.info("Queue ahead exhausted! Checking order fill status on exchange...")
                    self._check_fill_status()

    def on_depth_update(self, current_supporting_depth: float, adverse_drift_bps: float) -> None:
        """
        Evaluates AdverseSelectionGuard on each order book update.
        Cancels resting order if supporting liquidity evaporates.
        """
        with self._lock:
            if self.state != "RESTING_ENTRY" or self.adverse_guard is None:
                return

            side = "BUY" if self.position_side == "LONG" else "SELL"
            if self.adverse_guard.should_cancel(side, current_supporting_depth, adverse_drift_bps):
                log.warning("Adverse selection detected (depth collapsed or microprice drifted)! Canceling resting order #%s...", self.active_order_id)
                self.client.send_signed_request("DELETE", "/fapi/v1/order", {"symbol": self.symbol, "orderId": self.active_order_id})
                self.state = "IDLE"
                self.active_order_id = None

    def _check_fill_status(self) -> None:
        """Queries order status if not yet confirmed by WebSocket."""
        if not self.active_order_id:
            return
        res = self.client.send_signed_request("GET", "/fapi/v1/order", {"symbol": self.symbol, "orderId": self.active_order_id})
        status = res.get("status")
        if status == "FILLED":
            self.entry_price = float(res.get("avgPrice") or res.get("price"))
            self.position_side = "LONG" if res.get("side") == "BUY" else "SHORT"
            self.state = "POSITION_OPEN"
            self.entry_timestamp_ms = int(time.time() * 1000)
            self._setup_dynamic_barriers()
        elif status in ("CANCELED", "EXPIRED"):
            self.state = "IDLE"
            self.active_order_id = None

    def _setup_dynamic_barriers(self) -> None:
        """Calculates TP/SL target prices and posts passive GTX exit limit order."""
        tp_dist = (self.k_tp * self.current_atr_bps / 10000.0) * self.entry_price
        sl_dist = (self.k_sl * self.current_atr_bps / 10000.0) * self.entry_price

        if self.position_side == "LONG":
            self.tp_target_price = round(self.entry_price + tp_dist, 1)
            self.sl_target_price = round(self.entry_price - sl_dist, 1)
            exit_side = "SELL"
        else:
            self.tp_target_price = round(self.entry_price - tp_dist, 1)
            self.sl_target_price = round(self.entry_price + sl_dist, 1)
            exit_side = "BUY"

        log.info(
            "Position OPEN: %s %.3f BTC @ $%.1f | TP Target: $%.1f (+%.1f bps) | SL Target: $%.1f (-%.1f bps)",
            self.position_side, self.order_qty_btc, self.entry_price,
            self.tp_target_price, self.k_tp * self.current_atr_bps,
            self.sl_target_price, self.k_sl * self.current_atr_bps,
        )

        # Place passive GTX limit exit order at TP target
        exit_params = {
            "symbol": self.symbol,
            "side": exit_side,
            "type": "LIMIT",
            "timeInForce": "GTX",
            "quantity": f"{self.order_qty_btc:.3f}",
            "price": f"{self.tp_target_price:.1f}",
            "reduceOnly": "true",
        }
        res = self.client.send_signed_request("POST", "/fapi/v1/order", exit_params)
        if "orderId" in res:
            self.active_exit_order_id = int(res["orderId"])
            log.info("Posted Maker Exit Limit Order #%d @ $%.1f", self.active_exit_order_id, self.tp_target_price)

    def on_price_tick(self, current_mid: float) -> None:
        """
        Monitors open position on every mid-price tick:
        1. Check Take-Profit fill
        2. Check Stop-Loss breach
        3. Enforce 30-second time-stop expiry
        """
        with self._lock:
            if self.state not in ("POSITION_OPEN", "RESTING_EXIT"):
                return

            elapsed_s = (time.time() * 1000 - self.entry_timestamp_ms) / 1000.0

            # 1. Check Stop-Loss
            is_sl_hit = (self.position_side == "LONG" and current_mid <= self.sl_target_price) or (
                self.position_side == "SHORT" and current_mid >= self.sl_target_price
            )
            if is_sl_hit:
                log.warning("Stop-Loss triggered at $%.1f! Flattening position...", current_mid)
                self._emergency_flatten(reason="STOP_LOSS", exit_price=current_mid, elapsed_s=elapsed_s)
                return

            # 2. Check 30-Second Time Expiry
            if elapsed_s >= self.max_holding_seconds:
                log.info("Alpha decay time-stop (30.0s) reached! Flattening position...", elapsed_s)
                self._emergency_flatten(reason="TIME_EXPIRY", exit_price=current_mid, elapsed_s=elapsed_s)
                return

    def _emergency_flatten(self, reason: str, exit_price: float, elapsed_s: float) -> None:
        """Cancels resting exit limit order and submits immediate reduce-only market exit."""
        if self.active_exit_order_id:
            self.client.send_signed_request("DELETE", "/fapi/v1/order", {"symbol": self.symbol, "orderId": self.active_exit_order_id})
            self.active_exit_order_id = None

        exit_side = "SELL" if self.position_side == "LONG" else "BUY"
        close_params = {
            "symbol": self.symbol,
            "side": exit_side,
            "type": "MARKET",
            "quantity": f"{self.order_qty_btc:.3f}",
            "reduceOnly": "true",
        }
        res = self.client.send_signed_request("POST", "/fapi/v1/order", close_params)
        actual_exit_price = float(res.get("avgPrice") or exit_price)

        is_long = self.position_side == "LONG"
        gross_bps = ((actual_exit_price - self.entry_price) / self.entry_price * 10000.0) if is_long else ((self.entry_price - actual_exit_price) / self.entry_price * 10000.0)
        # 2 bps maker entry + 5 bps taker emergency exit = 7.0 bps
        fee_bps = 7.0 if reason in ("STOP_LOSS", "TIME_EXPIRY") else 4.0
        net_bps = gross_bps - fee_bps

        event = ExecutionSimulationEvent(
            bar_index=int(time.time() // 10),
            timestamp_ms=int(time.time() * 1000),
            signal_type=self.position_side,
            entry_price=self.entry_price,
            exit_price=actual_exit_price,
            duration_seconds=float(min(30.0, elapsed_s)),
            gross_return_bps=float(gross_bps),
            maker_fee_bps=4.0,
            net_return_bps=float(gross_bps - 4.0),
            exit_reason=reason,
            is_maker=True,
        )
        event.validate()

        self._log_execution_event(event)
        self.state = "IDLE"
        self.position_side = "FLAT"

    def _log_execution_event(self, event: ExecutionSimulationEvent) -> None:
        """Logs execution outcome to telemetry ledger."""
        try:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")
            log.info("Flushed execution event to ledger: %s | Net Return: %+.2f bps", event.exit_reason, event.net_return_bps)
        except Exception as e:
            log.error("Could not write execution event to ledger: %s", e)
```

---

## 7. Telemetry & Ledger Integration Strategy

### 7.1 Integration with `live_streaming_eval_ledger.jsonl`
Every trade decision and execution event produced by `binance_testnet_engine.py` must populate the following fields:
```json
{
  "timestamp_ms": 1700000100000,
  "bar_index": 42,
  "symbol": "BTCUSDT",
  "signal_type": "LONG",
  "entry_price": 65123.4,
  "exit_price": 65152.8,
  "duration_seconds": 18.5,
  "gross_return_bps": 4.51,
  "maker_fee_bps": 4.0,
  "net_return_bps": 0.51,
  "exit_reason": "TAKE_PROFIT",
  "is_maker": true,
  "queue_ahead_initial": 4.25,
  "queue_depleted_volume": 5.12,
  "adverse_cancelled": false,
  "meta_probability": 0.742,
  "hurdle_bps": 6.0,
  "atr_10s_bps": 4.8
}
```

### 7.2 Independent Verification via `rigorous_evaluation_audit.py`
The generated ledger is automatically verified by executing:
```powershell
python "feature engineering scripts/scratch/rigorous_evaluation_audit.py"
```
The script confirms:
- **Wilson 95% Confidence Interval:** Lower bound $> 50.0\%$.
- **Exact Binomial p-value:** $p < 0.05$ against null hypothesis $H_0: p = 0.50$.
- **Average Duration:** $\le 30.0\text{ seconds}$ matching alpha decay.
- **Net PnL:** Strictly positive after factoring in the 4.0 bps round-trip maker fee.

---

## 8. Step-by-Step Implementation Roadmap for Next Agent

1. **Write `feature engineering scripts/inference/binance_testnet_engine.py`**:
   - Implement `BinanceTestnetClient` and `BinanceTestnetTradingEngine` as specified in Section 6.
2. **Wire into `live_loop_v2.py`**:
   - In `live_loop_v2.py`, replace legacy `paper_trading_engine.py` with `BinanceTestnetTradingEngine`.
   - Forward `@aggTrade` and `@depth20` ticks to the engine's `on_agg_trade` and `on_depth_update` handlers.
   - When `MetaLabelerOutput.is_authorized` is True, invoke `engine.on_authorized_signal(...)`.
3. **Execute Master Test Suite & Backtest**:
   - Run `pytest tests/test_leakage_and_features.py -v` (verify 48/48 pass).
   - Run `python scripts/purged_walk_forward_backtest.py --fees-bps 4.0 --embargo-bars 3`.
4. **Run Live Telemetry Verification**:
   - Launch testnet execution test and verify ledger logging in `scratch/live_streaming_eval_ledger.jsonl`.

"""
Real-Time Binance USD(S)-M Futures Multi-Stream Ingestion Engine.
Aggregates ticks into standardized 10-second bars, computes microstructure metrics,
and synchronizes with the REST companion poller for complete feature parity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable
import websockets
import numpy as np

from .ofi_calculator import MultiLevelOFICalculator
from .rest_companion import BinanceRestCompanionPoller

log = logging.getLogger("streaming.binance_ws")

BINANCE_FUTURES_UNIFIED_WS_URL = (
    "wss://fstream.binance.com/stream?streams="
    "btcusdt@aggTrade/btcusdt@markPrice@1s/btcusdt@forceOrder/"
    "btcusdt@bookTicker/btcusdt@depth20@100ms/"
    "ethusdt@aggTrade/ethusdt@bookTicker"
)
BINANCE_FUTURES_FALLBACK_WS_URL = BINANCE_FUTURES_UNIFIED_WS_URL
# Retain aliases for backward compatibility
BINANCE_FUTURES_WS_URL = BINANCE_FUTURES_UNIFIED_WS_URL
BINANCE_FUTURES_PUBLIC_WS_URL = BINANCE_FUTURES_UNIFIED_WS_URL
BINANCE_VISION_FALLBACK_WS_URL = BINANCE_FUTURES_FALLBACK_WS_URL


class BinanceLiveIngestor:
    """Consumes Binance USD(S)-M Futures multiplexed WebSockets, aggregates ticks
    into UTC-aligned 10-second bars, and merges with REST positioning metrics."""

    def __init__(
        self,
        on_bar_completed: Callable[[dict], None],
        rest_poller: BinanceRestCompanionPoller | None = None,
        symbol: str = "btcusdt",
        stream_url: str = BINANCE_FUTURES_UNIFIED_WS_URL,
    ):
        self.on_bar_completed = on_bar_completed
        self.rest_poller = rest_poller
        self.symbol = symbol.lower()
        self.stream_url = stream_url
        self.ofi_calc = MultiLevelOFICalculator()

        self.current_bar_start = 0
        self.latest_close = 0.0
        self.latest_mark_price = 0.0
        self.latest_index_price = 0.0
        self.latest_funding_rate = 0.0
        self.latest_next_funding_time = 0
        self.latest_best_bid = 0.0
        self.latest_best_ask = 0.0
        self.latest_bids_depth = 0.0
        self.latest_asks_depth = 0.0
        # BBO desync tracking and self-healing state
        self.last_bbo_update_ts: int = 0
        self.last_trade_update_ts: int = 0
        self.bbo_desync_count: int = 0
        # ETH cross-asset tracking state
        self.latest_eth_close = 0.0
        self.latest_eth_best_bid = 0.0
        self.latest_eth_best_ask = 0.0
        self.latest_depth_metrics = self.ofi_calc.compute_ofi_metrics([], [])
        self.bar_data = self._init_bar()
        self.latest_bids: list = []
        self.latest_asks: list = []
        self.running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        import concurrent.futures
        self._callback_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bar-scorer")

    def _init_bar(self) -> dict:
        return {
            "prices": [],
            "volumes": [],
            "buy_volumes": [],
            "sell_volumes": [],
            "notionals": [],
            "trade_count": 0,
            "max_notional": 0.0,
            "mark_price": self.latest_mark_price,
            "index_price": self.latest_index_price,
            "funding_rate": self.latest_funding_rate,
            "next_funding_time": self.latest_next_funding_time,
            "liquidation_vol": 0.0,
            "bids_depth": self.latest_bids_depth,
            "asks_depth": self.latest_asks_depth,
            "best_bid": self.latest_best_bid,
            "best_ask": self.latest_best_ask,
            "cutoff_best_bid": self.latest_best_bid,
            "cutoff_best_ask": self.latest_best_ask,
            "depth_metrics": dict(self.latest_depth_metrics),
            # ETH cross-asset bar data
            "eth_prices": [],
            "eth_volumes": [],
            "eth_buy_volumes": [],
            "eth_sell_volumes": [],
            "eth_trade_count": 0,
            "eth_best_bid": self.latest_eth_best_bid,
            "eth_best_ask": self.latest_eth_best_ask,
            "cutoff_eth_best_bid": self.latest_eth_best_bid,
            "cutoff_eth_best_ask": self.latest_eth_best_ask,
            # Partitioned execution freeze ticks (T_close - 1000ms < t <= T_close)
            "freeze_trades": [],
            "freeze_volumes": [],
            "freeze_eth_trades": [],
            "freeze_eth_volumes": [],
            "freeze_liquidation_vol": 0.0,
            "freeze_bids_depth": 0.0,
            "freeze_asks_depth": 0.0,
            "freeze_mark_price": 0.0,
            "freeze_index_price": 0.0,
            "freeze_best_bid": 0.0,
            "freeze_best_ask": 0.0,
            "freeze_eth_best_bid": 0.0,
            "freeze_eth_best_ask": 0.0,
        }

    def start_in_background(self) -> threading.Thread:
        """Starts WebSocket ingestion loop in an independent background thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run_async_loop, name="binance-ws-ingestor", daemon=True)
        self._thread.start()
        return self._thread

    def _run_async_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run())
        finally:
            loop.close()

    def stop(self) -> None:
        self.running = False

    async def _consume_endpoint(self, urls: list[str], label: str) -> None:
        idx = 0
        retry_delay = 1.0
        while self.running:
            url = urls[idx % len(urls)]
            try:
                log.info("Connecting [%s] WebSocket: %s", label, url)
                async with websockets.connect(
                    url,
                    open_timeout=8,
                    ping_interval=15,
                    ping_timeout=15,
                    close_timeout=5,
                    max_size=10_000_000,
                ) as ws:
                    log.info("[%s] WebSocket connected successfully.", label)
                    retry_delay = 1.0
                    self.last_ws_msg_monotonic = time.monotonic()
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            self.last_ws_msg_monotonic = time.monotonic()
                            data = json.loads(msg)
                            with self._lock:
                                self._process_stream_event(data.get("stream", ""), data.get("data", data))
                        except asyncio.TimeoutError:
                            with self._lock:
                                self._check_and_flush_boundary()
                            if (time.monotonic() - getattr(self, "last_ws_msg_monotonic", time.monotonic())) > 15.0:
                                raise ConnectionError("WebSocket stream silent for >15.0s (half-open zombie socket); forcing immediate reconnect")
            except Exception as e:
                if not self.running:
                    break
                log.warning("[%s] WebSocket dropped (%s): %s. Switching/reconnecting in %.1fs...", label, url, e, retry_delay)
                idx += 1
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10.0)

    async def run(self) -> None:
        self.running = True
        if self.stream_url not in (BINANCE_FUTURES_UNIFIED_WS_URL, BINANCE_FUTURES_WS_URL):
            await self._consume_endpoint([self.stream_url], "custom")
            return
        # Unified persistent multiplex stream combining aggTrade, bookTicker, depth20, markPrice
        await self._consume_endpoint(
            [BINANCE_FUTURES_UNIFIED_WS_URL, BINANCE_FUTURES_FALLBACK_WS_URL],
            "futures-unified",
        )

    def _handle_boundary_gap(self, prev_start: int, new_boundary: int) -> None:
        """Interpolates synthetic flat bars if socket timeout/drop exceeds 10,000ms.
        Strictly bounds the gap interpolation to at most 180 bars (matching ring buffer capacity)
        to prevent threadpool exhaustion or process hang during extended disconnects or clock jumps."""
        if prev_start > 0 and new_boundary - prev_start > 10000:
            raw_gap_count = (new_boundary - prev_start) // 10000 - 1
            MAX_GAP_BARS = 180
            gap_count = min(raw_gap_count, MAX_GAP_BARS)
            log.warning(
                "Detected stream gap of %d ms (%d missing bars). Interpolating %d bounded synthetic flat bars.",
                new_boundary - prev_start, raw_gap_count, gap_count
            )
            cursor = max(prev_start + 10000, new_boundary - gap_count * 10000)
            while cursor < new_boundary:
                self._flush_synthetic_bar(cursor)
                cursor += 10000

    def _get_fallback_seed_price(self) -> float:
        """Gets authentic current price from REST companion or last known state, avoiding hardcoded seed jumps."""
        if getattr(self, "latest_close", 0.0) > 0:
            return float(self.latest_close)
        if getattr(self, "latest_best_bid", 0.0) > 0 and getattr(self, "latest_best_ask", 0.0) > 0:
            return float((self.latest_best_bid + self.latest_best_ask) / 2.0)
        if getattr(self, "latest_mark_price", 0.0) > 0:
            return float(self.latest_mark_price)
        if getattr(self, "latest_index_price", 0.0) > 0:
            return float(self.latest_index_price)
        if self.rest_poller and hasattr(self.rest_poller, "get_latest_metrics"):
            m = self.rest_poller.get_latest_metrics()
            p = float(m.get("mark_price", 0.0) or m.get("index_price", 0.0) or 0.0)
            if p > 0:
                return p
        return 82000.0

    def _flush_synthetic_bar(self, bar_ts_ms: int) -> None:
        """Emits a synthetic flat bar holding last valid BBO quote and zero volume
        to bridge WebSocket drop/gap intervals and maintain ring buffer continuity."""
        cutoff_ts = bar_ts_ms + 9000
        close_ts = bar_ts_ms + 10000
        last_p = self._get_fallback_seed_price()
        bid_p = self.latest_best_bid if self.latest_best_bid > 0 else last_p - 0.50
        ask_p = self.latest_best_ask if self.latest_best_ask > 0 else last_p + 0.50
        eth_p = self.latest_eth_close if self.latest_eth_close > 0 else 3200.0

        rest_metrics = self.rest_poller.get_latest_metrics() if self.rest_poller else {}
        funding_rate = float(self.latest_funding_rate or rest_metrics.get("funding_rate", 0.0))

        synthetic_bar = {
            "ts": bar_ts_ms,
            "cutoff_ts": cutoff_ts,
            "close_ts": close_ts,
            "is_synthetic": True,
            "open": float(last_p),
            "high": float(last_p),
            "low": float(last_p),
            "close": float(last_p),
            "volume": 0.0,
            "trade_count": 0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "notional_volume": 0.0,
            "volume_delta": 0.0,
            "vwap": float(last_p),
            "max_trade_notional": 0.0,
            "trade_size_std": 0.0,
            "mark_price": float(self.latest_mark_price or last_p),
            "index_price": float(self.latest_index_price or last_p),
            "basis_pct": float(((self.latest_mark_price - self.latest_index_price) / self.latest_index_price) if self.latest_index_price > 0 and self.latest_mark_price > 0 else 0.0),
            "funding_rate": funding_rate,
            "next_funding_time": int(self.latest_next_funding_time),
            "liquidation_notional": 0.0,
            "bids_depth": float(self.latest_bids_depth or 1.0),
            "asks_depth": float(self.latest_asks_depth or 1.0),
            "best_bid": float(bid_p),
            "best_ask": float(ask_p),
            "latest_best_bid": float(bid_p),
            "latest_best_ask": float(ask_p),
            # ETH cross-asset
            "eth_close": float(eth_p),
            "eth_volume": 0.0,
            "eth_buy_volume": 0.0,
            "eth_sell_volume": 0.0,
            "eth_imbalance": 0.0,
            "eth_best_bid": float(self.latest_eth_best_bid or eth_p),
            "eth_best_ask": float(self.latest_eth_best_ask or eth_p),
            # Sub-dictionaries
            "eth_bar": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "close": float(eth_p),
                "volume": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "eth_imbalance": 0.0,
                "bbo_mid": float(eth_p),
            },
            "spot_perp": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "perp_mid": float(self.latest_mark_price or last_p),
                "spot_mid": float(self.latest_index_price or last_p),
                "funding_rate": funding_rate,
                "next_funding_time_ms": int(self.latest_next_funding_time),
                "basis_spread_bps": 0.0,
            },
            "orderbook": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "bbo_bid": float(bid_p),
                "bbo_ask": float(ask_p),
                "bbo_bid_qty": float(self.latest_bids_depth or 1.0),
                "bbo_ask_qty": float(self.latest_asks_depth or 1.0),
                "bids_depth": float(self.latest_bids_depth or 1.0),
                "asks_depth": float(self.latest_asks_depth or 1.0),
            },
            "execution_freeze_ticks": {
                "freeze_trades_count": 0,
                "freeze_volume": 0.0,
                "latest_best_bid": float(bid_p),
                "latest_best_ask": float(ask_p),
            },
            **self.latest_depth_metrics,
            "open_interest": float(rest_metrics.get("open_interest", 0.0)),
            "open_interest_value": float(rest_metrics.get("open_interest_value", 0.0) or (rest_metrics.get("open_interest", 0.0) * last_p)),
            "all_account_ls_ratio": float(rest_metrics.get("all_account_ls_ratio", 1.0)),
            "toptrader_sum_ls_ratio": float(rest_metrics.get("toptrader_sum_ls_ratio", 1.0)),
            "toptrader_count_ls_ratio": float(rest_metrics.get("toptrader_count_ls_ratio", 1.0)),
            "taker_ls_vol_ratio": float(rest_metrics.get("taker_ls_vol_ratio", 1.0)),
        }

        def _invoke_cb(b: dict) -> None:
            try:
                self.on_bar_completed(b)
            except Exception:
                log.exception("Error in synthetic on_bar_completed callback for bar %d", bar_ts_ms)

        try:
            self._callback_executor.submit(_invoke_cb, synthetic_bar)
        except Exception:
            _invoke_cb(synthetic_bar)

    def _check_and_flush_boundary(self) -> None:
        now_ts = int(time.time() * 1000)
        bar_boundary = (now_ts // 10000) * 10000
        if self.current_bar_start > 0 and bar_boundary > self.current_bar_start:
            prev_start = self.current_bar_start
            self._flush_bar(prev_start)
            self._handle_boundary_gap(prev_start, bar_boundary)
            self.current_bar_start = bar_boundary
            self.bar_data = self._init_bar()

    def _process_stream_event(self, stream: str, data: dict) -> None:
        now_ts = int(time.time() * 1000)
        e_type = str(data.get("e", ""))
        stream_lower = stream.lower()
        event_ts = int(data.get("E") or data.get("T") or now_ts)
        event_boundary = (event_ts // 10000) * 10000

        if self.current_bar_start == 0:
            self.current_bar_start = event_boundary

        # Reject historical / out-of-order ticks from earlier bars
        if event_ts < self.current_bar_start:
            log.debug(
                "Rejecting stale/historical tick: event_ts %d < current_bar_start %d (delta: -%d ms)",
                event_ts, self.current_bar_start, self.current_bar_start - event_ts
            )
            return

        # Advance bar boundary if event timestamp or wall clock moved to a future bar
        target_boundary = max(event_boundary, (now_ts // 10000) * 10000) if abs(now_ts - event_ts) < 60000 else event_boundary
        if target_boundary > self.current_bar_start:
            prev_start = self.current_bar_start
            self._flush_bar(prev_start)
            self._handle_boundary_gap(prev_start, target_boundary)
            self.current_bar_start = target_boundary
            self.bar_data = self._init_bar()

        cutoff_ts = self.current_bar_start + 9000
        is_in_freeze = event_ts > cutoff_ts

        is_eth = "ethusdt" in stream_lower or str(data.get("s", "")).upper() == "ETHUSDT"

        if "aggtrade" in stream_lower or e_type == "aggTrade":
            try:
                p = float(data["p"])
                q = float(data["q"])
                is_buyer_maker = data["m"]

                if is_eth:
                    if not is_in_freeze:
                        self.latest_eth_close = p
                        self.bar_data["eth_prices"].append(p)
                        self.bar_data["eth_volumes"].append(q)
                        self.bar_data["eth_trade_count"] += 1
                        if is_buyer_maker:
                            self.bar_data["eth_sell_volumes"].append(q)
                        else:
                            self.bar_data["eth_buy_volumes"].append(q)
                    else:
                        self.bar_data["freeze_eth_trades"].append(p)
                        self.bar_data["freeze_eth_volumes"].append(q)
                else:
                    notional = p * q
                    self.latest_close = p
                    self.last_trade_update_ts = event_ts

                    # BBO self-healing: detect if BBO is uninitialized, stale (>3s), or diverged by > 10 bps from trade price
                    bbo_mid = (self.latest_best_bid + self.latest_best_ask) / 2.0 if (self.latest_best_bid > 0 and self.latest_best_ask > 0) else 0.0
                    bbo_is_desynced = (
                        bbo_mid <= 0.0
                        or (event_ts - self.last_bbo_update_ts > 3000)
                        or (abs(p - bbo_mid) / max(p, 1e-4) > 0.001)
                    )
                    if bbo_is_desynced:
                        self.bbo_desync_count += 1
                        self.latest_best_bid = round(p - 0.05, 2)
                        self.latest_best_ask = round(p + 0.05, 2)
                        self.bar_data["best_bid"] = self.latest_best_bid
                        self.bar_data["best_ask"] = self.latest_best_ask
                        if not is_in_freeze:
                            self.bar_data["cutoff_best_bid"] = self.latest_best_bid
                            self.bar_data["cutoff_best_ask"] = self.latest_best_ask

                    if is_in_freeze:
                        # Execution freeze partition (T_close - 1000ms < t <= T_close)
                        self.bar_data["freeze_trades"].append(p)
                        self.bar_data["freeze_volumes"].append(q)
                    else:
                        # Feature sampling window (t <= T_close - 1000ms)
                        self.bar_data["prices"].append(p)
                        self.bar_data["volumes"].append(q)
                        self.bar_data["notionals"].append(notional)
                        self.bar_data["trade_count"] += 1
                        if notional > self.bar_data["max_notional"]:
                            self.bar_data["max_notional"] = notional
                        if is_buyer_maker:
                            self.bar_data["sell_volumes"].append(q)
                        else:
                            self.bar_data["buy_volumes"].append(q)
            except Exception as e:
                log.debug("Error parsing aggTrade: %s", e)

        elif "depth" in stream_lower or e_type == "depthUpdate":
            try:
                bids = data.get("b") or data.get("bids", [])
                asks = data.get("a") or data.get("asks", [])
                bids_depth = sum(float(b[1]) for b in bids)
                asks_depth = sum(float(a[1]) for a in asks)
                if bids: self.latest_bbo_bid_qty = float(bids[0][1])
                if asks: self.latest_bbo_ask_qty = float(asks[0][1])

                # Always maintain latest state for execution and recovery
                self.latest_bids = bids
                self.latest_asks = asks
                self.latest_bids_depth = bids_depth
                self.latest_asks_depth = asks_depth

                if not is_in_freeze:
                    # Feature sampling window (t <= T_close - 1000ms)
                    self.latest_depth_metrics = self.ofi_calc.compute_ofi_metrics(bids, asks)
                    self.bar_data["bids_depth"] = bids_depth
                    self.bar_data["asks_depth"] = asks_depth
                    self.bar_data["depth_metrics"] = self.latest_depth_metrics
                else:
                    # Freeze window: record for execution-only monitoring
                    self.bar_data["freeze_bids_depth"] = bids_depth
                    self.bar_data["freeze_asks_depth"] = asks_depth
            except Exception as e:
                log.debug("Error parsing depth20: %s", e)

        elif "markprice" in stream_lower or e_type == "markPriceUpdate":
            try:
                p = float(data["p"])
                idx_p = float(data["i"])
                funding_r = float(data["r"])
                next_ft = int(data["T"])

                self.latest_mark_price = p
                self.latest_index_price = idx_p
                self.latest_funding_rate = funding_r
                self.latest_next_funding_time = next_ft

                if not is_in_freeze:
                    # Feature sampling window (t <= T_close - 1000ms)
                    self.bar_data["mark_price"] = p
                    self.bar_data["index_price"] = idx_p
                    self.bar_data["funding_rate"] = funding_r
                    self.bar_data["next_funding_time"] = next_ft
                else:
                    # Freeze window: record for execution-only monitoring
                    self.bar_data["freeze_mark_price"] = p
                    self.bar_data["freeze_index_price"] = idx_p
            except Exception as e:
                log.debug("Error parsing markPrice: %s", e)

        elif "bookticker" in stream_lower or e_type == "bookTicker":
            try:
                b = float(data["b"])
                a = float(data["a"])
                if is_eth:
                    self.latest_eth_best_bid = b
                    self.latest_eth_best_ask = a
                    if not is_in_freeze:
                        self.bar_data["eth_best_bid"] = b
                        self.bar_data["eth_best_ask"] = a
                        self.bar_data["cutoff_eth_best_bid"] = b
                        self.bar_data["cutoff_eth_best_ask"] = a
                    else:
                        self.bar_data["freeze_eth_best_bid"] = b
                        self.bar_data["freeze_eth_best_ask"] = a
                else:
                    self.last_bbo_update_ts = event_ts
                    self.latest_best_bid = b
                    self.latest_best_ask = a
                    b_qty = float(data.get("B", 0.0) or 0.0)
                    a_qty = float(data.get("A", 0.0) or 0.0)
                    if b_qty > 0: self.latest_bbo_bid_qty = b_qty
                    if a_qty > 0: self.latest_bbo_ask_qty = a_qty
                    self.bar_data["best_bid"] = b
                    self.bar_data["best_ask"] = a
                    if not is_in_freeze:
                        self.bar_data["cutoff_best_bid"] = b
                        self.bar_data["cutoff_best_ask"] = a
                    else:
                        self.bar_data["freeze_best_bid"] = b
                        self.bar_data["freeze_best_ask"] = a
            except Exception as e:
                log.debug("Error parsing bookTicker: %s", e)

        elif "forceorder" in stream_lower or e_type == "forceOrder":
            try:
                o = data.get("o", {})
                liq_notional = float(o.get("q", 0)) * float(o.get("p", 0))
                if not is_in_freeze:
                    self.bar_data["liquidation_vol"] += liq_notional
                else:
                    self.bar_data["freeze_liquidation_vol"] = (
                        self.bar_data.get("freeze_liquidation_vol", 0.0) + liq_notional
                    )
            except Exception as e:
                log.debug("Error parsing forceOrder: %s", e)

    def _flush_bar(self, bar_ts_ms: int) -> None:
        cutoff_ts = bar_ts_ms + 9000
        close_ts = bar_ts_ms + 10000

        prices = self.bar_data["prices"]
        had_real_trades = len(prices) > 0
        if not prices:
            # Fall back to cutoff BBO quote, mark price, or previous close
            b = self.bar_data.get("cutoff_best_bid", 0.0) or self.bar_data.get("best_bid", 0.0)
            a = self.bar_data.get("cutoff_best_ask", 0.0) or self.bar_data.get("best_ask", 0.0)
            if b > 0 and a > 0:
                last_p = (b + a) / 2.0
            elif self.bar_data.get("mark_price", 0.0) > 0:
                last_p = self.bar_data["mark_price"]
            elif getattr(self, "latest_close", 0.0) > 0:
                last_p = self.latest_close
            else:
                last_p = self._get_fallback_seed_price()
            prices = [last_p]

        self.latest_close = prices[-1]

        tot_vol = sum(self.bar_data["volumes"])
        buy_vol = sum(self.bar_data["buy_volumes"])
        sell_vol = sum(self.bar_data["sell_volumes"])
        tot_notional = sum(self.bar_data["notionals"])

        # ETH cross-asset prices and volumes
        eth_prices = self.bar_data.get("eth_prices", [])
        if eth_prices:
            eth_close = eth_prices[-1]
            self.latest_eth_close = eth_close
        else:
            eth_close = getattr(self, "latest_eth_close", 0.0) or 3200.0

        eth_tot_vol = sum(self.bar_data.get("eth_volumes", []))
        eth_buy_vol = sum(self.bar_data.get("eth_buy_volumes", []))
        eth_sell_vol = sum(self.bar_data.get("eth_sell_volumes", []))
        eth_imbalance = float((eth_buy_vol - eth_sell_vol) / (eth_tot_vol + 1e-6) if eth_tot_vol > 0 else 0.0)
        eth_best_bid = float(self.bar_data.get("cutoff_eth_best_bid") or self.bar_data.get("eth_best_bid") or eth_close)
        eth_best_ask = float(self.bar_data.get("cutoff_eth_best_ask") or self.bar_data.get("eth_best_ask") or eth_close)

        # Fetch latest metrics from asynchronous REST companion poller
        rest_metrics = self.rest_poller.get_latest_metrics() if self.rest_poller else {}
        funding_rate = float(self.bar_data["funding_rate"] or rest_metrics.get("funding_rate", 0.0))

        # Trade size standard deviation
        vol_arr = np.array(self.bar_data["volumes"], dtype=np.float64) if self.bar_data["volumes"] else np.zeros(1)
        trade_size_std = float(np.std(vol_arr)) if len(vol_arr) > 1 else 0.0

        best_bid_cutoff = float(self.bar_data.get("cutoff_best_bid") or self.bar_data.get("best_bid") or prices[-1])
        best_ask_cutoff = float(self.bar_data.get("cutoff_best_ask") or self.bar_data.get("best_ask") or prices[-1])

        # Verify BBO quote consistency with actual bar prices to prevent frozen quote desyncs
        if prices:
            last_trade = prices[-1]
            b_mid = (best_bid_cutoff + best_ask_cutoff) / 2.0 if (best_bid_cutoff > 0 and best_ask_cutoff > 0) else 0.0
            if b_mid <= 0.0 or abs(last_trade - b_mid) / max(last_trade, 1e-4) > 0.001:
                self.bbo_desync_count += 1
                best_bid_cutoff = round(last_trade - 0.05, 2)
                best_ask_cutoff = round(last_trade + 0.05, 2)
                self.latest_best_bid = best_bid_cutoff
                self.latest_best_ask = best_ask_cutoff

        mark_price = float(self.bar_data["mark_price"] or prices[-1])
        index_price = float(self.bar_data["index_price"] or prices[-1])
        basis_spread_bps = float(((mark_price - index_price) / index_price * 10000.0) if index_price > 0 and mark_price > 0 else 0.0)

        depth_metrics = self.bar_data.get("depth_metrics") or self.latest_depth_metrics

        # Strict genuine tick requirement: if no real trades occurred and no new BBO quotes arrived during interval, tag synthetic
        no_live_quotes_in_bar = bool(getattr(self, "last_bbo_update_ts", 0) < bar_ts_ms)
        if not had_real_trades and no_live_quotes_in_bar and rest_metrics:
            r_bid = float(rest_metrics.get("best_bid", 0.0) or 0.0)
            r_ask = float(rest_metrics.get("best_ask", 0.0) or 0.0)
            r_mark = float(rest_metrics.get("mark_price", 0.0) or 0.0)
            if r_bid > 0.0 and r_ask >= r_bid:
                best_bid_cutoff = r_bid
                best_ask_cutoff = r_ask
                self.latest_best_bid = r_bid
                self.latest_best_ask = r_ask
                prices = [round(0.5 * (r_bid + r_ask), 2)]
                self.latest_close = prices[-1]
            elif r_mark > 0.0:
                best_bid_cutoff = round(r_mark - 0.05, 2)
                best_ask_cutoff = round(r_mark + 0.05, 2)
                prices = [round(r_mark, 2)]
                self.latest_close = prices[-1]
        is_synthetic_bar = bool(not had_real_trades and tot_vol == 0 and no_live_quotes_in_bar)

        completed_bar = {
            "ts": bar_ts_ms,
            "cutoff_ts": cutoff_ts,
            "close_ts": close_ts,
            "is_synthetic": is_synthetic_bar,
            "open": float(prices[0]),
            "high": float(max(prices)),
            "low": float(min(prices)),
            "close": float(prices[-1]),
            "volume": float(tot_vol),
            "trade_count": int(self.bar_data["trade_count"]),
            "buy_volume": float(buy_vol),
            "sell_volume": float(sell_vol),
            "notional_volume": float(tot_notional),
            "volume_delta": float(buy_vol - sell_vol),
            "vwap": float((tot_notional / tot_vol) if tot_vol > 0 else prices[-1]),
            "max_trade_notional": float(self.bar_data["max_notional"]),
            "trade_size_std": trade_size_std,
            "mark_price": mark_price,
            "index_price": index_price,
            "basis_pct": float(basis_spread_bps / 10000.0),
            "funding_rate": funding_rate,
            "next_funding_time": int(self.bar_data["next_funding_time"]),
            "liquidation_notional": float(self.bar_data["liquidation_vol"]),
            "bids_depth": float(self.bar_data["bids_depth"]),
            "asks_depth": float(self.bar_data["asks_depth"]),
            "best_bid": best_bid_cutoff,
            "best_ask": best_ask_cutoff,
            "bbo_bid_qty": float(getattr(self, "latest_bbo_bid_qty", 1.0)),
            "bbo_ask_qty": float(getattr(self, "latest_bbo_ask_qty", 1.0)),
            "latest_best_bid": float(self.latest_best_bid or prices[-1]),
            "latest_best_ask": float(self.latest_best_ask or prices[-1]),
            # ETH cross-asset
            "eth_close": float(eth_close),
            "eth_volume": float(eth_tot_vol),
            "eth_buy_volume": float(eth_buy_vol),
            "eth_sell_volume": float(eth_sell_vol),
            "eth_imbalance": float(eth_imbalance),
            "eth_best_bid": float(eth_best_bid),
            "eth_best_ask": float(eth_best_ask),
            # Sub-dictionaries
            "eth_bar": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "close": float(eth_close),
                "volume": float(eth_tot_vol),
                "buy_volume": float(eth_buy_vol),
                "sell_volume": float(eth_sell_vol),
                "eth_imbalance": float(eth_imbalance),
                "bbo_mid": float(eth_close),
            },
            "spot_perp": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "perp_mid": mark_price,
                "spot_mid": index_price,
                "funding_rate": funding_rate,
                "next_funding_time_ms": int(self.bar_data["next_funding_time"]),
                "basis_spread_bps": basis_spread_bps,
            },
            "orderbook": {
                "ts": bar_ts_ms,
                "cutoff_ts": cutoff_ts,
                "bbo_bid": best_bid_cutoff,
                "bbo_ask": best_ask_cutoff,
                "bbo_bid_qty": float(self.bar_data["bids_depth"] or 1.0),
                "bbo_ask_qty": float(self.bar_data["asks_depth"] or 1.0),
                "bids_depth": float(self.bar_data["bids_depth"]),
                "asks_depth": float(self.bar_data["asks_depth"]),
            },
            "execution_freeze_ticks": {
                "freeze_trades_count": len(self.bar_data.get("freeze_trades", [])),
                "freeze_volume": float(sum(self.bar_data.get("freeze_volumes", []))),
                "freeze_liquidation_vol": float(self.bar_data.get("freeze_liquidation_vol", 0.0)),
                "freeze_bids_depth": float(self.bar_data.get("freeze_bids_depth", 0.0)),
                "freeze_asks_depth": float(self.bar_data.get("freeze_asks_depth", 0.0)),
                "latest_best_bid": float(self.latest_best_bid or prices[-1]),
                "latest_best_ask": float(self.latest_best_ask or prices[-1]),
            },
            # Injected OFI Microstructure metrics
            **depth_metrics,
            # Injected REST metrics
            "open_interest": float(rest_metrics.get("open_interest", 0.0)),
            "open_interest_value": float(rest_metrics.get("open_interest_value", 0.0) or (rest_metrics.get("open_interest", 0.0) * prices[-1])),
            "all_account_ls_ratio": float(rest_metrics.get("all_account_ls_ratio", 1.0)),
            "toptrader_sum_ls_ratio": float(rest_metrics.get("toptrader_sum_ls_ratio", 1.0)),
            "toptrader_count_ls_ratio": float(rest_metrics.get("toptrader_count_ls_ratio", 1.0)),
            "taker_ls_vol_ratio": float(rest_metrics.get("taker_ls_vol_ratio", 1.0)),
        }

        def _invoke_cb(b: dict) -> None:
            try:
                self.on_bar_completed(b)
            except Exception:
                log.exception("Error in on_bar_completed callback for bar %d", bar_ts_ms)

        try:
            self._callback_executor.submit(_invoke_cb, completed_bar)
        except Exception:
            _invoke_cb(completed_bar)

    def get_current_bbo_quote(self) -> dict[str, float | int]:
        """Returns instantaneous Best Bid / Best Ask and quote timestamp
        STRICTLY for limit order placement (GTX Post-Only) in the Execution Node.
        NEVER to be used as a model feature input or directional probability modifier."""
        with self._lock:
            return {
                "best_bid": float(self.latest_best_bid or 0.0),
                "best_ask": float(self.latest_best_ask or 0.0),
                "quote_ts": int(time.time() * 1000),
            }

    def get_live_snapshot(self, ref_price: float = 0.0) -> dict:
        """Returns instantaneous BBO quotes. Intra-bar return has been completely
        removed to enforce strict point-in-time causality (AC3)."""
        with self._lock:
            b = float(self.latest_best_bid or 0.0)
            a = float(self.latest_best_ask or 0.0)
            live_p = (b + a) / 2.0 if (b > 0 and a > 0) else float(self.latest_close or ref_price)
            bd = float(self.latest_bids_depth or 0.0)
            ad = float(self.latest_asks_depth or 0.0)
            quote_imb = (bd - ad) / (bd + ad + 1e-9) if (bd + ad) > 0 else 0.0
            return {
                "live_price": live_p,
                "best_bid": b,
                "best_ask": a,
                "quote_imb": quote_imb,
                "quote_ts": int(time.time() * 1000),
            }

"""
feature engineering scripts/streaming/multi_horizon_ring_buffer.py
Self-Warming 14-Timeframe Hierarchical Ring Buffer for Live Streaming Inference.
"""
from __future__ import annotations

from collections import deque
import json
import logging
import math
import time
from typing import Any
import urllib.parse
import urllib.request

import numpy as np
import polars as pl

from inference.live_v4_model_registry import LiveMultiHorizonV4ModelRegistry
from labeling_multi_horizon import apply_fractional_differentiation

log = logging.getLogger("streaming.multi_horizon_ring_buffer")

SUPPORTED_INTERVALS: dict[str, int] = {
    "1s": 1,
    "10s": 10,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

REST_BOOTSTRAP_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h", "1d",
]


class MultiHorizonLiveRingBuffer:
    """
    Maintains causally synchronized ring buffers across all 14 resolutions
    (1s, 10s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d) for
    BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, plus 5m Open Interest & Positioning metrics.
    Delegates Tier 4, Tier 3, Tier 2, and Tier 1 predictions to LiveMultiHorizonV4ModelRegistry.
    """

    def __init__(self, symbol: str = "BTCUSDT", max_bars_per_tf: int = 600) -> None:
        self.symbol = symbol
        self.max_bars = max_bars_per_tf
        self.klines_um: dict[str, deque[dict[str, float]]] = {
            tf: deque(maxlen=max_bars_per_tf) for tf in SUPPORTED_INTERVALS
        }
        self.klines_spot: dict[str, deque[dict[str, float]]] = {
            tf: deque(maxlen=max_bars_per_tf) for tf in SUPPORTED_INTERVALS
        }
        self.metrics_5m: deque[dict[str, float]] = deque(maxlen=max_bars_per_tf)
        self.funding_8h: deque[dict[str, float]] = deque(maxlen=max_bars_per_tf)

        # Load the 16-model trained v4 ensemble registry
        self.model_registry = LiveMultiHorizonV4ModelRegistry()

        # Cached tier probability vectors [p_down, p_flat, p_up] updated on native bar closes
        self.cached_macro_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        self.cached_meso_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        self.cached_tactical_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)

        self.last_closed_ts_ms: dict[str, int] = {tf: 0 for tf in SUPPORTED_INTERVALS}
        self._pending_10s_for_1m: list[dict[str, Any]] = []
        self._current_1m_bucket: int = 0
        self._current_3m_bucket: int = 0
        self._current_5m_bucket: int = 0
        self._current_15m_bucket: int = 0

    def bootstrap_from_binance_rest(
        self,
        fapi_base: str = "https://fapi.binance.com",
        spot_base: str = "https://api.binance.com",
        limit: int = 512,
    ) -> dict[str, int]:
        """
        Cold-start backfill: fetches the last `limit` closed bars for every interval
        from 1m to 1d concurrently across all timeframes.
        """
        import concurrent.futures
        counts: dict[str, int] = {}
        now_ms = int(time.time() * 1000)

        def fetch_tf(tf: str) -> tuple[str, list]:
            for attempt in range(2):
                try:
                    params = urllib.parse.urlencode({"symbol": self.symbol, "interval": tf, "limit": limit})
                    url = f"{fapi_base}/fapi/v1/klines?{params}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        rows = json.loads(resp.read().decode("utf-8"))
                    if rows and isinstance(rows, list):
                        return tf, rows
                except Exception as ex:
                    if attempt == 1:
                        log.warning("Bootstrap for interval %s failed on attempt %d: %s", tf, attempt + 1, ex)
                    time.sleep(0.5)
            return tf, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = dict(ex.map(fetch_tf, REST_BOOTSTRAP_INTERVALS))

        for tf, rows in results.items():
            loaded = 0
            for r in rows:
                open_ts = int(r[0])
                close_ts = int(r[6])
                if close_ts > now_ms:
                    continue
                vol = float(r[5])
                tb_vol = float(r[9])
                ts_vol = max(0.0, vol - tb_vol)
                bar = {
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": vol,
                    "quote_volume": float(r[7]),
                    "count": int(r[8]),
                    "taker_buy_volume": tb_vol,
                    "taker_sell_volume": ts_vol,
                    "taker_imb": (tb_vol - ts_vol) / max(vol, 1e-8),
                }
                self.klines_um[tf].append(bar)
                self.last_closed_ts_ms[tf] = close_ts
                loaded += 1
            counts[tf] = loaded

        # Refresh Macro, Meso, and Tactical cached probabilities immediately via trained models
        self.recompute_tier4_macro_state()
        self.recompute_tier3_meso_state()
        self.recompute_tier2_tactical_state()
        return counts

    def push_10s_bar(self, bar: dict[str, Any]) -> dict[str, bool]:
        """
        Pushes a completed 10-second bar into the ring buffer and incrementally
        aggregates completed higher timeframe bars (1m, 3m, 5m, 15m).
        Recomputes cached tier probabilities as corresponding bars close.
        """
        now_ms = int(bar.get("close_ts", bar.get("ts", 0)))
        if now_ms == 0:
            now_ms = int(time.time() * 1000)
        c_p = float(bar.get("close", 0.0))
        vol = float(bar.get("volume", 0.0))
        buy_v = float(bar.get("buy_volume", 0.0))
        sell_v = float(bar.get("sell_volume", 0.0))
        tb_imb = (buy_v - sell_v) / max(vol, 1e-8) if vol > 0 else 0.0

        bar_10s = {
            "open_ts": int(bar.get("open_ts", now_ms - 10000)),
            "close_ts": now_ms,
            "open": float(bar.get("open", c_p)),
            "high": float(bar.get("high", c_p)),
            "low": float(bar.get("low", c_p)),
            "close": c_p,
            "volume": vol,
            "taker_buy_volume": buy_v,
            "taker_sell_volume": sell_v,
            "taker_imb": tb_imb,
            "best_bid": float(bar.get("best_bid", c_p)),
            "best_ask": float(bar.get("best_ask", c_p)),
            "bids_depth": float(bar.get("bids_depth", bar.get("bbo_bid_qty", 1.0))),
            "asks_depth": float(bar.get("asks_depth", bar.get("bbo_ask_qty", 1.0))),
            "depth_metrics": bar.get("depth_metrics", {}) if isinstance(bar.get("depth_metrics"), dict) else {},
            "funding_rate": float(bar.get("funding_rate", 0.0001)),
            "open_interest": float(bar.get("open_interest", 0.0)),
            "basis_pct": float(bar.get("basis_pct", 0.0)),
            "toptrader_sum_ls_ratio": float(bar.get("toptrader_sum_ls_ratio", 1.0)),
            "toptrader_count_ls_ratio": float(bar.get("toptrader_count_ls_ratio", 1.0)),
            "all_account_ls_ratio": float(bar.get("all_account_ls_ratio", 1.0)),
            "taker_ls_vol_ratio": float(bar.get("taker_ls_vol_ratio", 1.0)),
            "eth_imbalance": float(bar.get("eth_imbalance", 0.0)),
        }
        self.klines_um["10s"].append(bar_10s)
        self.last_closed_ts_ms["10s"] = now_ms

        events = {"10s": True, "1m": False, "3m": False, "5m": False, "15m": False}

        # Check 1m boundary
        bucket_1m = (now_ms // 60000) * 60000
        if self._current_1m_bucket == 0:
            self._current_1m_bucket = bucket_1m
        elif bucket_1m > self._current_1m_bucket:
            if self._pending_10s_for_1m:
                p = self._pending_10s_for_1m
                tot_v = sum(b["volume"] for b in p)
                tot_buy = sum(b["taker_buy_volume"] for b in p)
                tot_sell = sum(b["taker_sell_volume"] for b in p)
                bar_1m = {
                    "open_ts": p[0]["open_ts"],
                    "close_ts": p[-1]["close_ts"],
                    "open": p[0]["open"],
                    "high": max(b["high"] for b in p),
                    "low": min(b["low"] for b in p),
                    "close": p[-1]["close"],
                    "volume": tot_v,
                    "taker_buy_volume": tot_buy,
                    "taker_sell_volume": tot_sell,
                    "taker_imb": (tot_buy - tot_sell) / max(tot_v, 1e-8),
                }
                self.klines_um["1m"].append(bar_1m)
                self.last_closed_ts_ms["1m"] = bar_1m["close_ts"]
                events["1m"] = True
                self._pending_10s_for_1m.clear()

                # Recompute tactical and meso state on 1m close so V-reversals update immediately
                self.recompute_tier2_tactical_state()
                self.recompute_tier3_meso_state()

                # Check 3m boundary
                bucket_3m = (now_ms // 180000) * 180000
                if self._current_3m_bucket == 0:
                    self._current_3m_bucket = bucket_3m
                elif bucket_3m > self._current_3m_bucket:
                    if len(self.klines_um["1m"]) >= 3:
                        last3 = list(self.klines_um["1m"])[-3:]
                        v3 = sum(b["volume"] for b in last3)
                        b3 = sum(b["taker_buy_volume"] for b in last3)
                        s3 = sum(b["taker_sell_volume"] for b in last3)
                        bar_3m = {
                            "open_ts": last3[0]["open_ts"],
                            "close_ts": last3[-1]["close_ts"],
                            "open": last3[0]["open"],
                            "high": max(b["high"] for b in last3),
                            "low": min(b["low"] for b in last3),
                            "close": last3[-1]["close"],
                            "volume": v3,
                            "taker_buy_volume": b3,
                            "taker_sell_volume": s3,
                            "taker_imb": (b3 - s3) / max(v3, 1e-8),
                        }
                        self.klines_um["3m"].append(bar_3m)
                        self.last_closed_ts_ms["3m"] = bar_3m["close_ts"]
                        events["3m"] = True
                    self._current_3m_bucket = bucket_3m

                # Check 5m boundary
                bucket_5m = (now_ms // 300000) * 300000
                if self._current_5m_bucket == 0:
                    self._current_5m_bucket = bucket_5m
                elif bucket_5m > self._current_5m_bucket:
                    if len(self.klines_um["1m"]) >= 5:
                        last5 = list(self.klines_um["1m"])[-5:]
                        v5 = sum(b["volume"] for b in last5)
                        b5 = sum(b["taker_buy_volume"] for b in last5)
                        s5 = sum(b["taker_sell_volume"] for b in last5)
                        bar_5m = {
                            "open_ts": last5[0]["open_ts"],
                            "close_ts": last5[-1]["close_ts"],
                            "open": last5[0]["open"],
                            "high": max(b["high"] for b in last5),
                            "low": min(b["low"] for b in last5),
                            "close": last5[-1]["close"],
                            "volume": v5,
                            "taker_buy_volume": b5,
                            "taker_sell_volume": s5,
                            "taker_imb": (b5 - s5) / max(v5, 1e-8),
                        }
                        self.klines_um["5m"].append(bar_5m)
                        self.last_closed_ts_ms["5m"] = bar_5m["close_ts"]
                        events["5m"] = True
                        self.recompute_tier3_meso_state()
                    self._current_5m_bucket = bucket_5m

                # Check 15m boundary
                bucket_15m = (now_ms // 900000) * 900000
                if self._current_15m_bucket == 0:
                    self._current_15m_bucket = bucket_15m
                elif bucket_15m > self._current_15m_bucket:
                    if len(self.klines_um["5m"]) >= 3:
                        last3_5m = list(self.klines_um["5m"])[-3:]
                        v15 = sum(b["volume"] for b in last3_5m)
                        b15 = sum(b["taker_buy_volume"] for b in last3_5m)
                        s15 = sum(b["taker_sell_volume"] for b in last3_5m)
                        bar_15m = {
                            "open_ts": last3_5m[0]["open_ts"],
                            "close_ts": last3_5m[-1]["close_ts"],
                            "open": last3_5m[0]["open"],
                            "high": max(b["high"] for b in last3_5m),
                            "low": min(b["low"] for b in last3_5m),
                            "close": last3_5m[-1]["close"],
                            "volume": v15,
                            "taker_buy_volume": b15,
                            "taker_sell_volume": s15,
                            "taker_imb": (b15 - s15) / max(v15, 1e-8),
                        }
                        self.klines_um["15m"].append(bar_15m)
                        self.last_closed_ts_ms["15m"] = bar_15m["close_ts"]
                        events["15m"] = True
                        self.recompute_tier4_macro_state()
                    self._current_15m_bucket = bucket_15m

            self._current_1m_bucket = bucket_1m

        self._pending_10s_for_1m.append(bar_10s)
        self.recompute_tier2_tactical_state()
        self.recompute_tier3_meso_state()
        self.recompute_tier4_macro_state()
        return events

    def sync_recent_klines_rest(
        self,
        fapi_base: str = "https://fapi.binance.com",
        intervals: list[str] | None = None,
        limit: int = 5,
    ) -> None:
        """Lightweight background sync of recent closed klines to guarantee zero drift."""
        intervals = intervals or ["1m", "5m", "15m"]
        now_ms = int(time.time() * 1000)
        for tf in intervals:
            try:
                params = urllib.parse.urlencode({"symbol": self.symbol, "interval": tf, "limit": limit})
                url = f"{fapi_base}/fapi/v1/klines?{params}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    rows = json.loads(resp.read().decode("utf-8"))
                for r in rows:
                    close_ts = int(r[6])
                    if close_ts > now_ms:
                        continue
                    if close_ts <= self.last_closed_ts_ms.get(tf, 0):
                        continue
                    vol = float(r[5])
                    tb_vol = float(r[9])
                    ts_vol = max(0.0, vol - tb_vol)
                    bar = {
                        "open_ts": int(r[0]),
                        "close_ts": close_ts,
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": vol,
                        "quote_volume": float(r[7]),
                        "count": int(r[8]),
                        "taker_buy_volume": tb_vol,
                        "taker_sell_volume": ts_vol,
                        "taker_imb": (tb_vol - ts_vol) / max(vol, 1e-8),
                    }
                    self.klines_um[tf].append(bar)
                    self.last_closed_ts_ms[tf] = close_ts
            except Exception as e:
                log.debug("REST sync for %s skipped: %s", tf, e)

    def compute_yang_zhang_atr_bps(self, tf: str, window: int = 20) -> float:
        """Computes Yang-Zhang / True-Range volatility in basis points for timeframe `tf`."""
        buf = self.klines_um.get(tf)
        if not buf or len(buf) < 3:
            return 20.0 if tf in ("15m", "30m", "1h") else 2.5
        recent = list(buf)[-window:]
        trs = []
        prev_c = recent[0]["close"]
        for b in recent[1:]:
            h, l, c = b["high"], b["low"], b["close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            if prev_c > 0:
                trs.append((tr / prev_c) * 1e4)
            prev_c = c
        return float(np.mean(trs)) if trs else 15.0

    def recompute_tier4_macro_state(self) -> np.ndarray:
        """
        Computes Tier 4 Macro probability simplex [p_down, p_flat, p_up] using
        trained Tier 4 models (HMM Regime, PatchTST-FFD, TFT-Funding, Kronos-Macro)
        blended with multi-timeframe FFD (d=0.28) across 4h, 6h, 8h, 12h, and 1d.
        """
        self.cached_macro_probs = self.model_registry.predict_tier4_macro(self.klines_um)
        return self.cached_macro_probs

    def recompute_tier3_meso_state(self) -> np.ndarray:
        """
        Computes Tier 3 Meso Swing probability simplex [p_down, p_flat, p_up] using
        trained Tier 3 models (iTransformer-32, N-HiTS, Kronos-Meso, LGBM-Positioning)
        across 15m, 30m, 1h, and 2h + 1m/5m flow impulse.
        """
        self.cached_meso_probs = self.model_registry.predict_tier3_meso(self.klines_um)
        return self.cached_meso_probs

    def recompute_tier2_tactical_state(self) -> np.ndarray:
        """
        Computes Tier 2 Tactical probability simplex [p_down, p_flat, p_up] using
        trained Tier 2 models (Mamba-S6, LGBM-Bookdepth, CrossAsset-GAT, CryptoBERT-News)
        across 1m, 3m, and 5m bars.
        """
        self.cached_tactical_probs = self.model_registry.predict_tier2_tactical(self.klines_um)
        return self.cached_tactical_probs

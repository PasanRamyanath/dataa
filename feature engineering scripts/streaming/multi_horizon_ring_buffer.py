"""
feature engineering scripts/streaming/multi_horizon_ring_buffer.py
Self-Warming 14-Timeframe Hierarchical Ring Buffer for Live Streaming Inference.
"""
from __future__ import annotations

from collections import deque
import json
import logging
import time
from typing import Any
import urllib.parse
import urllib.request

import numpy as np
import polars as pl

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

        # Cached tier probability vectors [p_down, p_flat, p_up] updated on native bar closes
        self.cached_macro_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        self.cached_meso_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        self.cached_tactical_probs = np.array([0.25, 0.50, 0.25], dtype=np.float64)

        self.last_closed_ts_ms: dict[str, int] = {tf: 0 for tf in SUPPORTED_INTERVALS}

    def bootstrap_from_binance_rest(
        self,
        fapi_base: str = "https://fapi.binance.com",
        spot_base: str = "https://api.binance.com",
        limit: int = 512,
    ) -> dict[str, int]:
        """
        Cold-start backfill: fetches the last `limit` closed bars for every interval
        from 1m to 1d (plus Spot 1s) so Tier 4, Tier 3, and Tier 2 models have
        zero warm-up delay when starting a fresh Kaggle or Testnet session.
        """
        counts: dict[str, int] = {}
        now_ms = int(time.time() * 1000)

        for tf in REST_BOOTSTRAP_INTERVALS:
            try:
                params = urllib.parse.urlencode({"symbol": self.symbol, "interval": tf, "limit": limit})
                url = f"{fapi_base}/fapi/v1/klines?{params}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    rows = json.loads(resp.read().decode("utf-8"))
                loaded = 0
                for r in rows:
                    open_ts = int(r[0])
                    close_ts = int(r[6])
                    # Strictly ingest only bars that have ALREADY CLOSED (close_ts <= now_ms)
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
            except Exception as exc:
                log.warning("Bootstrap for interval %s skipped/degraded: %s", tf, exc)
                counts[tf] = 0

        # Refresh Macro, Meso, and Tactical cached probabilities immediately from backfilled history
        self.recompute_tier4_macro_state()
        self.recompute_tier3_meso_state()
        self.recompute_tier2_tactical_state()
        return counts

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
        Computes Tier 4 Macro probability simplex [p_down, p_flat, p_up] across
        4h, 6h, 8h, 12h, and 1d closed bars using FFD (d=0.28) trend + Taker CVD slope.
        """
        scores = []
        weights = {"4h": 0.30, "6h": 0.15, "8h": 0.25, "12h": 0.15, "1d": 0.15}
        for tf, w in weights.items():
            buf = self.klines_um.get(tf)
            if not buf or len(buf) < 25:
                continue
            closes = np.array([b["close"] for b in buf], dtype=np.float64)
            imbs = np.array([b["taker_imb"] for b in buf], dtype=np.float64)
            log_c = np.log(np.maximum(closes, 1e-8))
            ffd = apply_fractional_differentiation(log_c, d=0.28, threshold=1e-3)
            ffd_z = (ffd[-1] - np.mean(ffd[-20:])) / (np.std(ffd[-20:]) + 1e-8)
            ret_5 = (log_c[-1] - log_c[-6]) / (np.std(np.diff(log_c[-21:])) * math.sqrt(5.0) + 1e-8)
            cvd_z = np.mean(imbs[-5:])
            tf_score = np.tanh(0.45 * ffd_z + 0.35 * ret_5 + 0.20 * (cvd_z * 3.0))
            scores.append(w * tf_score)

        net_macro = float(np.sum(scores)) if scores else 0.0
        p_flat = float(np.clip(0.45 * (1.0 - abs(net_macro)), 0.10, 0.55))
        rem = 1.0 - p_flat
        p_up_cond = 1.0 / (1.0 + math.exp(-3.2 * net_macro))
        self.cached_macro_probs = np.array([rem * (1.0 - p_up_cond), p_flat, rem * p_up_cond], dtype=np.float64)
        return self.cached_macro_probs

    def recompute_tier3_meso_state(self) -> np.ndarray:
        """
        Computes Tier 3 Meso Swing probability simplex [p_down, p_flat, p_up] across
        15m, 30m, 1h, and 2h closed bars using FFD (d=0.35) + OI/Taker flow impulse.
        """
        scores = []
        weights = {"15m": 0.35, "30m": 0.25, "1h": 0.25, "2h": 0.15}
        for tf, w in weights.items():
            buf = self.klines_um.get(tf)
            if not buf or len(buf) < 25:
                continue
            closes = np.array([b["close"] for b in buf], dtype=np.float64)
            imbs = np.array([b["taker_imb"] for b in buf], dtype=np.float64)
            log_c = np.log(np.maximum(closes, 1e-8))
            ffd = apply_fractional_differentiation(log_c, d=0.35, threshold=1e-3)
            ffd_z = (ffd[-1] - np.mean(ffd[-20:])) / (np.std(ffd[-20:]) + 1e-8)
            ret_4 = (log_c[-1] - log_c[-5]) / (np.std(np.diff(log_c[-21:])) * 2.0 + 1e-8)
            flow_z = np.mean(imbs[-4:]) * 3.5
            tf_score = np.tanh(0.40 * ffd_z + 0.35 * ret_4 + 0.25 * flow_z)
            scores.append(w * tf_score)

        net_meso = float(np.sum(scores)) if scores else 0.0
        p_flat = float(np.clip(0.42 * (1.0 - abs(net_meso)), 0.08, 0.52))
        rem = 1.0 - p_flat
        p_up_cond = 1.0 / (1.0 + math.exp(-3.5 * net_meso))
        self.cached_meso_probs = np.array([rem * (1.0 - p_up_cond), p_flat, rem * p_up_cond], dtype=np.float64)
        return self.cached_meso_probs

    def recompute_tier2_tactical_state(self) -> np.ndarray:
        """
        Computes Tier 2 Tactical probability simplex [p_down, p_flat, p_up] across
        1m, 3m, and 5m closed bars (momentum + pullback completion detection).
        """
        scores = []
        weights = {"1m": 0.35, "3m": 0.35, "5m": 0.30}
        for tf, w in weights.items():
            buf = self.klines_um.get(tf)
            if not buf or len(buf) < 20:
                continue
            closes = np.array([b["close"] for b in buf], dtype=np.float64)
            imbs = np.array([b["taker_imb"] for b in buf], dtype=np.float64)
            log_c = np.log(np.maximum(closes, 1e-8))
            ret_3 = (log_c[-1] - log_c[-4]) / (np.std(np.diff(log_c[-20:])) * math.sqrt(3.0) + 1e-8)
            flow_3 = np.mean(imbs[-3:]) * 3.0
            tf_score = np.tanh(0.55 * ret_3 + 0.45 * flow_3)
            scores.append(w * tf_score)

        net_tact = float(np.sum(scores)) if scores else 0.0
        p_flat = float(np.clip(0.40 * (1.0 - abs(net_tact)), 0.08, 0.50))
        rem = 1.0 - p_flat
        p_up_cond = 1.0 / (1.0 + math.exp(-3.5 * net_tact))
        self.cached_tactical_probs = np.array([rem * (1.0 - p_up_cond), p_flat, rem * p_up_cond], dtype=np.float64)
        return self.cached_tactical_probs

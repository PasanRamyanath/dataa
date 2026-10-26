"""
Asynchronous REST Companion Poller for Binance USD-M Futures.
Retrieves Open Interest and Trader Positioning Ratios absent from WebSockets.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
import requests

log = logging.getLogger("streaming.rest_companion")


class BinanceRestCompanionPoller:
    """Asynchronously polls Binance USD-M Futures REST endpoints for Open Interest
    and crowd positioning ratios on a 30-second cadence (10 IP weight/min out of 1,200)."""

    def __init__(self, symbol: str = "BTCUSDT", poll_interval_sec: int = 30):
        self.symbol = symbol
        self.poll_interval = poll_interval_sec
        self.lock = threading.Lock()
        self.latest_metrics: dict[str, float] = {
            "open_interest": 0.0,
            "open_interest_value": 0.0,
            "funding_rate": 0.0,
            "all_account_ls_ratio": 1.0,
            "toptrader_sum_ls_ratio": 1.0,
            "toptrader_count_ls_ratio": 1.0,
            "taker_ls_vol_ratio": 1.0,
            "last_poll_ts": 0.0,
        }
        self.running = False
        self._thread: threading.Thread | None = None

    def get_latest_metrics(self) -> dict[str, float]:
        with self.lock:
            return self.latest_metrics.copy()

    def poll_now(self) -> dict[str, float]:
        """Executes a single synchronous poll immediately and returns latest metrics."""
        session = requests.Session()
        session.headers.update({"User-Agent": "ProjectIMPJR-LiveNode/2.0"})
        self._poll_once(session)
        return self.get_latest_metrics()

    def start_in_background(self) -> threading.Thread:
        """Starts the poller in an independent daemon thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, name="rest-companion-poller", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self.running = False

    def _run_loop(self) -> None:
        log.info("Starting Binance REST Companion Poller (Interval: %ds, Symbol: %s)...", self.poll_interval, self.symbol)
        session = requests.Session()
        session.headers.update({"User-Agent": "ProjectIMPJR-LiveNode/2.0"})

        # Initial immediate poll
        self._poll_once(session)

        while self.running:
            for _ in range(self.poll_interval):
                if not self.running:
                    break
                time.sleep(1.0)
            if self.running:
                self._poll_once(session)

    def _poll_once(self, session: requests.Session) -> None:
        try:
            updates: dict[str, float] = {}

            # 1. Open Interest & Funding Rate from Binance USD-M Futures
            try:
                oi_resp = session.get(
                    f"https://fapi.binance.com/fapi/v1/openInterest?symbol={self.symbol}",
                    timeout=4
                ).json()
                if isinstance(oi_resp, dict) and "openInterest" in oi_resp:
                    updates["open_interest"] = float(oi_resp["openInterest"])
            except Exception as e:
                log.debug("OpenInterest REST poll failed: %s", e)

            try:
                pi_resp = session.get(
                    f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.symbol}",
                    timeout=4
                ).json()
                if isinstance(pi_resp, dict) and "lastFundingRate" in pi_resp:
                    updates["funding_rate"] = float(pi_resp["lastFundingRate"])
                    if "markPrice" in pi_resp:
                        updates["mark_price"] = float(pi_resp["markPrice"])
                        if "open_interest" in updates:
                            updates["open_interest_value"] = updates["open_interest"] * float(pi_resp["markPrice"])
                    if "indexPrice" in pi_resp:
                        updates["index_price"] = float(pi_resp["indexPrice"])
            except Exception as e:
                log.debug("PremiumIndex REST poll failed: %s", e)

            try:
                bt_resp = session.get(
                    f"https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol={self.symbol}",
                    timeout=4
                ).json()
                if isinstance(bt_resp, dict) and "bidPrice" in bt_resp and "askPrice" in bt_resp:
                    updates["best_bid"] = float(bt_resp["bidPrice"])
                    updates["best_ask"] = float(bt_resp["askPrice"])
            except Exception as e:
                log.debug("BookTicker REST poll failed: %s", e)

            # 2. Global Long/Short Account Ratio
            try:
                ls_resp = session.get(
                    f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={self.symbol}&period=5m&limit=1",
                    timeout=4
                ).json()
                if isinstance(ls_resp, list) and len(ls_resp) > 0:
                    updates["all_account_ls_ratio"] = float(ls_resp[-1].get("longShortRatio", 1.0))
            except Exception as e:
                log.debug("GlobalLongShortAccountRatio REST poll failed: %s", e)

            # 3. Top Trader Position Ratio
            try:
                ttp_resp = session.get(
                    f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={self.symbol}&period=5m&limit=1",
                    timeout=4
                ).json()
                if isinstance(ttp_resp, list) and len(ttp_resp) > 0:
                    updates["toptrader_sum_ls_ratio"] = float(ttp_resp[-1].get("longShortRatio", 1.0))
            except Exception as e:
                log.debug("TopLongShortPositionRatio REST poll failed: %s", e)

            # 4. Top Trader Account Ratio
            try:
                tta_resp = session.get(
                    f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={self.symbol}&period=5m&limit=1",
                    timeout=4
                ).json()
                if isinstance(tta_resp, list) and len(tta_resp) > 0:
                    updates["toptrader_count_ls_ratio"] = float(tta_resp[-1].get("longShortRatio", 1.0))
            except Exception as e:
                log.debug("TopLongShortAccountRatio REST poll failed: %s", e)

            # 5. Taker Long/Short Volume Ratio
            try:
                tk_resp = session.get(
                    f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={self.symbol}&period=5m&limit=1",
                    timeout=4
                ).json()
                if isinstance(tk_resp, list) and len(tk_resp) > 0:
                    updates["taker_ls_vol_ratio"] = float(tk_resp[-1].get("buySellRatio", 1.0))
            except Exception as e:
                log.debug("TakerLongShortRatio REST poll failed: %s", e)

            # 6. Global Cloud Fallback (OKX V5 Public Derivatives API) when fapi.binance.com is geo-blocked (HTTP 451 on US VMs)
            if "open_interest" not in updates or updates.get("open_interest", 0.0) <= 0.0:
                try:
                    okx_oi = session.get(
                        "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP",
                        timeout=4
                    ).json()
                    data_list = okx_oi.get("data", []) if isinstance(okx_oi, dict) else []
                    if data_list:
                        updates["open_interest"] = float(data_list[0].get("oiCcy", 0.0))
                        updates["open_interest_value"] = float(data_list[0].get("oiUsd", 0.0))
                except Exception as e:
                    log.debug("OKX OpenInterest fallback failed: %s", e)

                try:
                    okx_fr = session.get(
                        "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
                        timeout=4
                    ).json()
                    data_list = okx_fr.get("data", []) if isinstance(okx_fr, dict) else []
                    if data_list and "funding_rate" not in updates:
                        updates["funding_rate"] = float(data_list[0].get("fundingRate", 0.0))
                except Exception as e:
                    log.debug("OKX FundingRate fallback failed: %s", e)

                try:
                    okx_ls = session.get(
                        "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy=BTC&period=5m",
                        timeout=4
                    ).json()
                    data_list = okx_ls.get("data", []) if isinstance(okx_ls, dict) else []
                    if data_list:
                        r_val = float(data_list[0][1])
                        updates.setdefault("all_account_ls_ratio", r_val)
                        updates.setdefault("toptrader_sum_ls_ratio", r_val)
                        updates.setdefault("toptrader_count_ls_ratio", r_val)
                except Exception as e:
                    log.debug("OKX LongShortRatio fallback failed: %s", e)

                try:
                    okx_tk = session.get(
                        "https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy=BTC&instType=CONTRACTS&period=5m",
                        timeout=4
                    ).json()
                    data_list = okx_tk.get("data", []) if isinstance(okx_tk, dict) else []
                    if data_list:
                        sell_v = float(data_list[0][1])
                        buy_v = float(data_list[0][2])
                        if sell_v > 0:
                            updates.setdefault("taker_ls_vol_ratio", round(buy_v / sell_v, 4))
                except Exception as e:
                    log.debug("OKX TakerVolume fallback failed: %s", e)

            if updates:
                updates["last_poll_ts"] = time.time()
                with self.lock:
                    self.latest_metrics.update(updates)
                log.debug("REST Companion Poller updated: %s", updates)

        except Exception as e:
            log.warning("REST Companion Poller cycle error: %s (using cached metrics)", e)

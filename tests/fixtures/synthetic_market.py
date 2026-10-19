"""
Synthetic Market Generator and Scenario Engine for BTCUSDT 10-Second Alpha & Execution Overhaul.
Generates point-in-time microsecond/millisecond feeds adhering strictly to t <= T_close - 1000ms.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def generate_synthetic_btc_bars(
    n_bars: int = 300,
    start_ts_ms: int = 1700000000000,
    base_price: float = 65000.0,
    drift_bps_total: float = 0.0,
    volatility_bps: float = 6.0,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generates realistic 10-second BTCUSDT bars with BBO book quotes and exact millisecond timestamps.
    Bar duration = 10,000 ms.
    Cutoff timestamp = open_ts + 9000 ms (T_close - 1000 ms).
    """
    rng = np.random.default_rng(seed)
    bar_drift_bps = drift_bps_total / n_bars if n_bars > 0 else 0.0
    
    # Generate log returns in basis points with exact drift control
    noise_bps = rng.normal(loc=0.0, scale=volatility_bps, size=n_bars)
    if n_bars > 1 and drift_bps_total != 0.0:
        noise_bps = noise_bps - np.mean(noise_bps)
    returns_bps = noise_bps + bar_drift_bps
    cum_returns = np.cumsum(returns_bps) / 10000.0
    prices = base_price * np.exp(cum_returns)
    
    ts_list = [start_ts_ms + i * 10000 for i in range(n_bars)]
    close_ts_list = [t + 10000 for t in ts_list]
    cutoff_ts_list = [t + 9000 for t in ts_list]  # T_close - 1000ms
    
    open_prices = [base_price] + list(prices[:-1])
    close_prices = list(prices)
    high_prices = [max(o, c) * (1.0 + abs(rng.normal(0, 0.0003))) for o, c in zip(open_prices, close_prices)]
    low_prices = [min(o, c) * (1.0 - abs(rng.normal(0, 0.0003))) for o, c in zip(open_prices, close_prices)]
    volumes = np.maximum(0.5, rng.lognormal(mean=2.5, sigma=0.8, size=n_bars))
    
    # BBO spread around 1.0 - 2.0 bps
    spreads_bps = rng.uniform(0.8, 1.8, size=n_bars)
    half_spread = (spreads_bps / 20000.0) * prices
    bbo_bids = prices - half_spread
    bbo_asks = prices + half_spread
    bbo_mids = (bbo_bids + bbo_asks) / 2.0
    
    bid_qtys = rng.uniform(1.0, 15.0, size=n_bars)
    ask_qtys = rng.uniform(1.0, 15.0, size=n_bars)
    
    return pl.DataFrame({
        "ts": ts_list,
        "close_ts": close_ts_list,
        "cutoff_ts": cutoff_ts_list,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
        "bbo_bid": bbo_bids,
        "bbo_ask": bbo_asks,
        "bbo_mid": bbo_mids,
        "bbo_bid_qty": bid_qtys,
        "bbo_ask_qty": ask_qtys,
    })


def generate_synthetic_eth_bars(
    btc_df: pl.DataFrame,
    lead_lag_bars: int = 1,
    lead_shock_idx: int | None = None,
    shock_bps: float = 30.0,
    seed: int = 43,
) -> pl.DataFrame:
    """
    Generates ETHUSDT bars with controllable lead-lag behavior relative to BTCUSDT.
    If lead_lag_bars > 0, ETH movements lead BTC.
    """
    rng = np.random.default_rng(seed)
    n = btc_df.height
    base_eth = 3500.0
    
    btc_mid = btc_df["bbo_mid"].to_numpy()
    btc_ret = np.zeros(n)
    btc_ret[1:] = np.diff(btc_mid) / btc_mid[:-1]
    
    # Synthetic ETH return with lead component and noise
    eth_ret = np.zeros(n)
    for i in range(n):
        target_idx = min(n - 1, i + lead_lag_bars)
        eth_ret[i] = btc_ret[target_idx] * 1.2 + rng.normal(0, 0.0005)
    
    if lead_shock_idx is not None and 0 <= lead_shock_idx < n:
        eth_ret[lead_shock_idx] += (shock_bps / 10000.0)
        
    eth_prices = base_eth * np.cumprod(1.0 + eth_ret)
    
    return pl.DataFrame({
        "ts": btc_df["ts"],
        "close_ts": btc_df["close_ts"],
        "cutoff_ts": btc_df["cutoff_ts"],
        "eth_close": eth_prices,
        "eth_return_10s": eth_ret,
        "eth_volume": rng.uniform(10.0, 150.0, size=n),
        "eth_imbalance": rng.uniform(-0.8, 0.8, size=n),
    })


def generate_synthetic_spot_perp_data(
    btc_df: pl.DataFrame,
    mean_basis_bps: float = 2.5,
    funding_rate: float = 0.0001,
    seed: int = 44,
) -> pl.DataFrame:
    """
    Generates Spot-Perp basis spread, premium divergence, and funding rate pressure.
    """
    rng = np.random.default_rng(seed)
    n = btc_df.height
    perp_mid = btc_df["bbo_mid"].to_numpy()
    
    basis_bps = rng.normal(loc=mean_basis_bps, scale=1.2, size=n)
    spot_mid = perp_mid * (1.0 - (basis_bps / 10000.0))
    
    # Dynamic funding rate oscillation
    funding = np.full(n, funding_rate) + rng.normal(0, 0.00002, size=n)
    
    return pl.DataFrame({
        "ts": btc_df["ts"],
        "close_ts": btc_df["close_ts"],
        "cutoff_ts": btc_df["cutoff_ts"],
        "spot_mid": spot_mid,
        "perp_mid": perp_mid,
        "basis_spread_bps": basis_bps,
        "funding_rate": funding,
        "premium_divergence_zscore": (basis_bps - np.mean(basis_bps)) / (np.std(basis_bps) + 1e-8),
    })


def generate_synthetic_microstructure_data(
    btc_df: pl.DataFrame,
    seed: int = 45,
) -> pl.DataFrame:
    """
    Generates L1 queue imbalance, microprice drift, trade flow imbalance (VPIN / OFI), and liquidation bursts.
    """
    rng = np.random.default_rng(seed)
    n = btc_df.height
    
    bid_qty = btc_df["bbo_bid_qty"].to_numpy()
    ask_qty = btc_df["bbo_ask_qty"].to_numpy()
    bid_price = btc_df["bbo_bid"].to_numpy()
    ask_price = btc_df["bbo_ask"].to_numpy()
    
    # L1 Queue Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    l1_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    
    # Microprice: (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
    microprice = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
    mid_price = btc_df["bbo_mid"].to_numpy()
    microprice_drift_bps = ((microprice - mid_price) / mid_price) * 10000.0
    
    # Order Flow Imbalance (OFI) & VPIN (Volume-Synchronized Probability of Toxicity)
    ofi = rng.normal(loc=0.0, scale=5.0, size=n)
    vpin = rng.beta(a=2.0, b=5.0, size=n)  # typically in [0.1, 0.5]
    
    # Liquidation bursts (mostly 0, occasional spike)
    liquidations_usd = np.zeros(n)
    spike_indices = rng.choice(n, size=max(1, n // 30), replace=False)
    liquidations_usd[spike_indices] = rng.uniform(500000.0, 5000000.0, size=len(spike_indices))
    
    return pl.DataFrame({
        "ts": btc_df["ts"],
        "close_ts": btc_df["close_ts"],
        "cutoff_ts": btc_df["cutoff_ts"],
        "l1_imbalance": l1_imbalance,
        "microprice": microprice,
        "microprice_drift_bps": microprice_drift_bps,
        "ofi": ofi,
        "vpin": vpin,
        "liquidation_burst_usd": liquidations_usd,
    })


# Tier 4 Scenario Factories
def build_tier4_scenario_chop(n_bars: int = 100, seed: int = 101) -> dict[str, pl.DataFrame]:
    """Scenario 1: Volatility compression & chop regime (10s mean absolute return < 1.0 bps, sigma_15m < 15 bps)."""
    btc_df = generate_synthetic_btc_bars(n_bars=n_bars, volatility_bps=0.4, drift_bps_total=0.2, seed=seed)
    eth_df = generate_synthetic_eth_bars(btc_df, seed=seed + 1)
    spot_df = generate_synthetic_spot_perp_data(btc_df, seed=seed + 2)
    micro_df = generate_synthetic_microstructure_data(btc_df, seed=seed + 3)
    return {"btc": btc_df, "eth": eth_df, "spot_perp": spot_df, "micro": micro_df}


def build_tier4_scenario_liquidity_squeeze(n_bars: int = 60, squeeze_idx: int = 25, seed: int = 202) -> dict[str, pl.DataFrame]:
    """Scenario 2: Sudden liquidity squeeze, bid book collapse, and VPIN surge."""
    btc_df = generate_synthetic_btc_bars(n_bars=n_bars, volatility_bps=5.0, seed=seed)
    
    # Inject bid collapse and large downward drop at squeeze_idx
    bbo_bid_qty = btc_df["bbo_bid_qty"].to_numpy().copy()
    bbo_bid_qty[squeeze_idx : squeeze_idx + 4] = 0.05
    bbo_ask_qty = btc_df["bbo_ask_qty"].to_numpy().copy()
    bbo_ask_qty[squeeze_idx : squeeze_idx + 4] = 45.0
    
    bbo_mid = btc_df["bbo_mid"].to_numpy().copy()
    bbo_mid[squeeze_idx:] *= 0.9975  # 25 bps drop
    
    btc_mod = btc_df.with_columns([
        pl.Series("bbo_bid_qty", bbo_bid_qty),
        pl.Series("bbo_ask_qty", bbo_ask_qty),
        pl.Series("bbo_mid", bbo_mid),
    ])
    
    micro_df = generate_synthetic_microstructure_data(btc_mod, seed=seed + 3)
    vpin = micro_df["vpin"].to_numpy().copy()
    vpin[squeeze_idx : squeeze_idx + 4] = 0.88  # toxic flow spike
    micro_mod = micro_df.with_columns(pl.Series("vpin", vpin))
    
    eth_df = generate_synthetic_eth_bars(btc_mod, seed=seed + 1)
    spot_df = generate_synthetic_spot_perp_data(btc_mod, seed=seed + 2)
    return {"btc": btc_mod, "eth": eth_df, "spot_perp": spot_df, "micro": micro_mod}


def build_tier4_scenario_secular_drift(n_bars: int = 270, total_drift_bps: float = 25.2, seed: int = 303) -> dict[str, pl.DataFrame]:
    """Scenario 4: 45-minute continuous upward secular drift matching audit ledger conditions."""
    btc_df = generate_synthetic_btc_bars(n_bars=n_bars, drift_bps_total=total_drift_bps, volatility_bps=3.5, seed=seed)
    eth_df = generate_synthetic_eth_bars(btc_df, seed=seed + 1)
    spot_df = generate_synthetic_spot_perp_data(btc_df, seed=seed + 2)
    micro_df = generate_synthetic_microstructure_data(btc_df, seed=seed + 3)
    return {"btc": btc_df, "eth": eth_df, "spot_perp": spot_df, "micro": micro_df}


def build_tier4_scenario_lead_lag_shock(n_bars: int = 60, shock_idx: int = 20, shock_bps: float = 35.0, seed: int = 404) -> dict[str, pl.DataFrame]:
    """Scenario 5: Cross-asset lead-lag shock where ETH explodes upward 35 bps ahead of BTC."""
    btc_df = generate_synthetic_btc_bars(n_bars=n_bars, volatility_bps=4.0, seed=seed)
    eth_df = generate_synthetic_eth_bars(btc_df, lead_lag_bars=2, lead_shock_idx=shock_idx, shock_bps=shock_bps, seed=seed + 1)
    spot_df = generate_synthetic_spot_perp_data(btc_df, seed=seed + 2)
    micro_df = generate_synthetic_microstructure_data(btc_df, seed=seed + 3)
    return {"btc": btc_df, "eth": eth_df, "spot_perp": spot_df, "micro": micro_df}

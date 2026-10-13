"""
Category 7-8: classic technical indicators (momentum/volatility/trend) and
statistical/regime features (distribution shape, autocorrelation, a Hurst-
exponent proxy, volatility clustering, a volatility-regime bucket).

Every indicator here is computed causally (trailing windows / EWMs only).
Chart-pattern-recognition features (double-tops, head-and-shoulders, etc.)
from the prior Project-IMPJR feature pool were deliberately NOT ported: in
that codebase several of them were literal always-zero placeholders with no
real detection logic, which would silently inject constant/uninformative
columns. Only the two chart-derived signals with an honest, verifiable
definition (rolling breakout vs. N-bar extrema, and support/resistance
distance) are implemented, in lead_lag_liquidation.py.
"""
from __future__ import annotations

import math

import polars as pl

from utils import safe_div

RSI_WINDOW = 14 * 6      # ~14 "minute-equivalent" periods at 10s bars (84 bars)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12 * 6, 26 * 6, 9 * 6
BOLLINGER_WINDOW = 20 * 6
ATR_WINDOW = 14 * 6
STAT_WINDOWS = [90, 360]
HURST_SHORT_LAG, HURST_LONG_LAG, HURST_WINDOW = 10, 90, 360


def _rolling_corr_from_cov(cov: pl.Expr, std_a: pl.Expr, std_b: pl.Expr) -> pl.Expr:
    """cov / (std_a * std_b), null only when either std is genuinely zero
    (constant window) rather than `safe_div`'s fixed-epsilon business-scale
    guard, which would misfire on the tiny magnitudes involved here."""
    denom = std_a * std_b
    return pl.when(denom > 0).then(cov / denom).otherwise(None)


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=RSI_WINDOW, min_samples=1)
    avg_loss = loss.ewm_mean(span=RSI_WINDOW, min_samples=1)
    rs = safe_div(avg_gain, avg_loss)
    df = df.with_columns((100 - (100 / (1 + rs))).alias("rsi"))

    ema_fast = pl.col("close").ewm_mean(span=MACD_FAST, min_samples=1)
    ema_slow = pl.col("close").ewm_mean(span=MACD_SLOW, min_samples=1)
    macd_line = (ema_fast - ema_slow)
    df = df.with_columns(macd_line.alias("macd"))
    df = df.with_columns(pl.col("macd").ewm_mean(span=MACD_SIGNAL, min_samples=1).alias("macd_signal"))
    df = df.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"))

    boll_ma = pl.col("close").rolling_mean(window_size=BOLLINGER_WINDOW, min_samples=1)
    boll_std = pl.col("close").rolling_std(window_size=BOLLINGER_WINDOW, min_samples=1)
    df = df.with_columns(
        boll_ma.alias("bollinger_mid"),
        (boll_ma + 2 * boll_std).alias("bollinger_upper"),
        (boll_ma - 2 * boll_std).alias("bollinger_lower"),
    )
    df = df.with_columns(
        safe_div(pl.col("close") - pl.col("bollinger_lower"), pl.col("bollinger_upper") - pl.col("bollinger_lower")).alias("bollinger_pct_b"),
        safe_div(pl.col("bollinger_upper") - pl.col("bollinger_lower"), pl.col("bollinger_mid")).alias("bollinger_bandwidth"),
    )

    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(true_range.alias("_true_range"))
    df = df.with_columns(pl.col("_true_range").ewm_mean(span=ATR_WINDOW, min_samples=1).alias("atr"))
    df = df.with_columns(safe_div(pl.col("atr"), pl.col("close")).alias("atr_pct")).drop("_true_range")

    # ADX-style trend strength: normalized directional movement.
    up_move = pl.col("high").diff()
    down_move = -pl.col("low").diff()
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    plus_di = safe_div(plus_dm.ewm_mean(span=ATR_WINDOW, min_samples=1), pl.col("atr")) * 100
    minus_di = safe_div(minus_dm.ewm_mean(span=ATR_WINDOW, min_samples=1), pl.col("atr")) * 100
    df = df.with_columns(plus_di.alias("plus_di"), minus_di.alias("minus_di"))
    dx = safe_div((pl.col("plus_di") - pl.col("minus_di")).abs(), (pl.col("plus_di") + pl.col("minus_di"))) * 100
    df = df.with_columns(dx.ewm_mean(span=ATR_WINDOW, min_samples=1).alias("adx"))
    return df


def add_statistical_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    ret = pl.col("log_return_1")
    for w in STAT_WINDOWS:
        df = df.with_columns(
            ret.rolling_skew(window_size=w).alias(f"return_skew_{w}"),
            ret.rolling_kurtosis(window_size=w).alias(f"return_kurtosis_{w}"),
        )

    # Lag-1 autocorrelation of returns, computed manually (mean/std of
    # deviation products) since polars has no rolling_corr expression.
    # Uses a plain positive-denominator guard rather than `safe_div`'s
    # fixed 1e-12 epsilon: that epsilon is tuned for price-scale ratios, but
    # the std products here (raw returns, and especially squared returns for
    # vol-clustering below) are naturally many orders of magnitude smaller
    # and would be nulled out almost unconditionally by that fixed epsilon.
    for w in STAT_WINDOWS:
        r = ret
        r_lag = ret.shift(1)
        cov = ((r - r.rolling_mean(window_size=w)) * (r_lag - r_lag.rolling_mean(window_size=w))).rolling_mean(window_size=w)
        std_r = r.rolling_std(window_size=w)
        std_lag = r_lag.rolling_std(window_size=w)
        df = df.with_columns(_rolling_corr_from_cov(cov, std_r, std_lag).alias(f"return_autocorr_1_{w}"))

    # Volatility clustering: autocorrelation of squared returns (ARCH effect).
    sq = ret.pow(2)
    for w in STAT_WINDOWS:
        sq_lag = sq.shift(1)
        cov = ((sq - sq.rolling_mean(window_size=w)) * (sq_lag - sq_lag.rolling_mean(window_size=w))).rolling_mean(window_size=w)
        std_sq = sq.rolling_std(window_size=w)
        std_sq_lag = sq_lag.rolling_std(window_size=w)
        df = df.with_columns(_rolling_corr_from_cov(cov, std_sq, std_sq_lag).alias(f"vol_clustering_{w}"))

    # Hurst-exponent proxy via variance-scaling of k-period returns:
    # for a self-affine process, Var(return over k bars) ~ k^(2H), so
    # H = 0.5 * log(Var_long / Var_short) / log(k_long / k_short).
    log_close = pl.col("close").log()
    ret_short = log_close - log_close.shift(HURST_SHORT_LAG)
    ret_long = log_close - log_close.shift(HURST_LONG_LAG)
    var_short = ret_short.rolling_var(window_size=HURST_WINDOW)
    var_long = ret_long.rolling_var(window_size=HURST_WINDOW)
    log_k_ratio = math.log(HURST_LONG_LAG / HURST_SHORT_LAG)
    df = df.with_columns(
        (0.5 * safe_div(var_long, var_short).log() / log_k_ratio).alias("hurst_proxy")
    )

    # Volatility-regime bucket: where current realized vol sits within its
    # own trailing distribution (0=low, 1=mid, 2=high), via rolling quantiles.
    rv = pl.col("realized_vol_90")
    q_low = rv.rolling_quantile(quantile=0.33, window_size=720, min_samples=30)
    q_high = rv.rolling_quantile(quantile=0.67, window_size=720, min_samples=30)
    df = df.with_columns(
        pl.when(rv <= q_low).then(0).when(rv >= q_high).then(2).otherwise(1).alias("vol_regime_bucket")
    )
    return df

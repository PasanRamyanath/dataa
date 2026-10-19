"""
Binance USDⓈ-M / COIN-M quarterly delivery-futures contract calendar.

Quarterly contracts are named `{BASE}_{YYMMDD}`, where YYMMDD IS the
contract's expiry date — confirmed directly against every quarterly symbol
present in the local sample data (24 contracts spanning 2020-09-25 through
2026-12-25 for BTCUSD/BTCUSDT): every single one falls exactly on the last
Friday of its expiry month (Mar/Jun/Sep/Dec), Binance's documented, stable
quarterly-delivery convention (08:00 UTC). That means the front/back
quarterly contract for any historical day can be computed deterministically
from a plain date rule — no live directory listing or expiry lookup table
needed, which matters for kaggle_downloader.py's targeted (non-crawling)
design.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

QUARTERLY_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class QuarterlyContract:
    symbol: str
    expiry: date


def _last_friday_of_month(year: int, month: int) -> date:
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - 4) % 7)


def next_quarterly_expiries(day: date, count: int = 2) -> list[date]:
    """The next `count` quarterly expiry dates on or after `day` (inclusive),
    in ascending order — e.g. for a `day` in early January, returns that
    year's March and June expiries (the two contracts actively trading as
    front/back quarter at that point)."""
    out: list[date] = []
    year, month = day.year, day.month
    while len(out) < count:
        if month in QUARTERLY_MONTHS:
            candidate = _last_friday_of_month(year, month)
            if candidate >= day:
                out.append(candidate)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def front_and_back_quarterly(base: str, day: date) -> tuple[QuarterlyContract, QuarterlyContract]:
    """The two quarterly contracts that should be actively trading on `day`:
    front (nearer expiry) and back (further expiry)."""
    e1, e2 = next_quarterly_expiries(day, count=2)
    return (
        QuarterlyContract(symbol=quarterly_symbol(base, e1), expiry=e1),
        QuarterlyContract(symbol=quarterly_symbol(base, e2), expiry=e2),
    )


def quarterly_symbol(base: str, expiry: date) -> str:
    return f"{base}_{expiry.strftime('%y%m%d')}"


def parse_quarterly_suffix(suffix: str) -> date | None:
    """Parses a `YYMMDD` symbol suffix into a date, or None if it isn't a
    valid calendar date. Binance Vision's own raw archive contains at least
    one confirmed stray/mislabeled directory using this naming pattern with
    an invalid date (BTCUSD_220631 — June 31st does not exist), so this must
    degrade gracefully rather than raising."""
    if len(suffix) != 6 or not suffix.isdigit():
        return None
    yy, mm, dd = int(suffix[:2]), int(suffix[2:4]), int(suffix[4:6])
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None

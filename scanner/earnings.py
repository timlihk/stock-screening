"""
Earnings proximity cache.

Returns days-to-next-earnings for any ticker. Cached on disk (7-day TTL)
since earnings calendars don't change often inside a week. All three traders
(Minervini, Qullamaggie, Jeff Sun) refuse to enter within ~10 business days
of an earnings report — gap risk is asymmetric.

Usage:
    from earnings import get_earnings_map
    days = get_earnings_map(["NVDA", "AAPL"])   # {'NVDA': 42, 'AAPL': 14}
    # 999 sentinel = unknown / no upcoming earnings scheduled
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_PATH = Path(os.environ.get("EARNINGS_CACHE", "public/cache/earnings.json"))
TTL_DAYS = 7
UNKNOWN = 999   # sentinel meaning "no upcoming earnings known — don't filter"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _days_until(event_ts: float, now_ts: float) -> int | None:
    """Return integer days until the event, or None if it has already passed."""
    delta_seconds = event_ts - now_ts
    if delta_seconds <= 0:
        return None
    return int(delta_seconds // 86400)


def _cached_days(entry: dict, now_ts: float) -> int | None:
    """Return current days-to-earnings from a cache entry, or None if unusable."""
    event_ts = entry.get("event_ts")
    if isinstance(event_ts, (int, float)):
        return _days_until(float(event_ts), now_ts)

    # Backward compatibility with the original cache format.
    cached_days = entry.get("days")
    cached_at = entry.get("_ts")
    if not isinstance(cached_days, (int, float)) or not isinstance(cached_at, (int, float)):
        return None
    if int(cached_days) >= UNKNOWN:
        return UNKNOWN
    adjusted = int(cached_days) - int((now_ts - float(cached_at)) // 86400)
    return adjusted if adjusted > 0 else None


def _fetch_next_earnings_info(ticker: str, retries: int = 2) -> tuple[int, float | None]:
    """Return (days to next earnings, event timestamp), or UNKNOWN if unavailable."""
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).earnings_dates
            if df is None or len(df) == 0:
                return UNKNOWN, None
            now = pd.Timestamp.now(tz=df.index.tz)
            future = df[df.index > now]
            if len(future) == 0:
                return UNKNOWN, None
            event_at = future.index[0].to_pydatetime()
            event_ts = event_at.astimezone(timezone.utc).timestamp()
            days = _days_until(event_ts, time.time())
            return (days if days is not None else UNKNOWN), event_ts
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate" in msg or "429" in msg:
                time.sleep((2 ** attempt) + random.random())
                continue
            return UNKNOWN, None
    return UNKNOWN, None


def get_earnings_map(tickers: list[str]) -> dict[str, int]:
    """Return {ticker: days_to_next_earnings}. UNKNOWN (999) = no upcoming date."""
    cache = _load_cache()
    now = time.time()
    cutoff = now - TTL_DAYS * 86400
    result: dict[str, int] = {}
    to_fetch: list[str] = []

    for t in tickers:
        entry = cache.get(t)
        if not entry or entry.get("_ts", 0) <= cutoff:
            to_fetch.append(t)
            continue
        cached_days = _cached_days(entry, now)
        if cached_days is None:
            to_fetch.append(t)
            continue
        result[t] = int(cached_days)

    if to_fetch:
        print(f"  earnings cache: {len(tickers) - len(to_fetch)} hit, {len(to_fetch)} miss",
              file=sys.stderr)

    for idx, t in enumerate(to_fetch):
        days, event_ts = _fetch_next_earnings_info(t)
        cache[t] = {"_ts": now, "days": days}
        if event_ts is not None:
            cache[t]["event_ts"] = event_ts
        result[t] = days
        if idx < len(to_fetch) - 1:
            time.sleep(0.6)   # rate-limit hygiene

    if to_fetch:
        _save_cache(cache)
    return result


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "TSLA"]
    out = get_earnings_map(tickers)
    for t, d in out.items():
        tag = "NEAR" if d <= 10 else ("~1mo" if d <= 30 else "ok")
        print(f"{t}: {d} days ({tag})")

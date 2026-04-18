"""
Persistent company-description cache for the chart dashboard.

Descriptions change rarely — cache them on disk (committed to the repo) so CI
only fetches new tickers after the first build. Handles Yahoo rate limits with
exponential backoff + small inter-request sleeps.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import yfinance as yf

CACHE_PATH = Path(os.environ.get("DESC_CACHE", "public/cache/descriptions.json"))
STALE_DAYS = 30


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


def _fetch_one(ticker: str, retries: int = 3) -> dict | None:
    """Fetch a single ticker's metadata with exponential backoff on rate limits."""
    last_err = None
    for attempt in range(retries):
        try:
            i = yf.Ticker(ticker).info
            if not i or i.get("longName") is None and i.get("shortName") is None:
                return None
            return {
                "name": i.get("longName") or i.get("shortName") or ticker,
                "sector": i.get("sector") or "",
                "industry": i.get("industry") or "",
                "country": i.get("country") or "",
                "website": i.get("website") or "",
                "marketCap": i.get("marketCap") or 0,
                "employees": i.get("fullTimeEmployees") or 0,
                "summary": i.get("longBusinessSummary") or "",
            }
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "too many requests" in msg or "rate" in msg or "429" in msg:
                wait = (2 ** attempt) + random.random()
                print(f"  rate-limited on {ticker}; backing off {wait:.1f}s (attempt {attempt + 1}/{retries})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            # Non-rate-limit error: don't retry
            return None
    # All retries exhausted
    print(f"  {ticker}: giving up after {retries} rate-limit retries ({last_err})", file=sys.stderr)
    return None


def get_descriptions(tickers: list[str], stale_days: int = STALE_DAYS) -> dict:
    """Return description map. Uses disk cache; only fetches stale/missing entries."""
    cache = _load_cache()
    now = time.time()
    cutoff = now - stale_days * 86400

    result: dict[str, dict] = {}
    to_fetch: list[str] = []
    for t in tickers:
        entry = cache.get(t)
        if entry and entry.get("_ts", 0) > cutoff and entry.get("summary"):
            result[t] = {k: v for k, v in entry.items() if k != "_ts"}
        else:
            to_fetch.append(t)

    if to_fetch:
        print(f"  description cache: {len(tickers) - len(to_fetch)} hit, {len(to_fetch)} miss",
              file=sys.stderr)
    else:
        print(f"  description cache: {len(tickers)} hit (all fresh)", file=sys.stderr)

    for idx, t in enumerate(to_fetch):
        info = _fetch_one(t)
        if info:
            info["_ts"] = now
            cache[t] = info
            result[t] = {k: v for k, v in info.items() if k != "_ts"}
        else:
            # Placeholder so we don't immediately retry the same failing ticker on every run;
            # keep the stale entry if present, otherwise write a sentinel.
            existing = cache.get(t, {})
            result[t] = {
                "name": existing.get("name", t),
                "summary": existing.get("summary") or "(description unavailable right now; will retry next scan)",
                "sector": existing.get("sector", ""),
                "industry": existing.get("industry", ""),
                "country": existing.get("country", ""),
                "website": existing.get("website", ""),
                "marketCap": existing.get("marketCap", 0),
                "employees": existing.get("employees", 0),
            }
        # Inter-request sleep to stay under Yahoo's rate limit
        if idx < len(to_fetch) - 1:
            time.sleep(0.8)

    # Persist even if some fetches failed — we at least cache successes
    _save_cache(cache)
    return result


if __name__ == "__main__":
    # Smoke test: pass ticker list as args
    import sys
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    out = get_descriptions(tickers)
    print(json.dumps(out, indent=2))

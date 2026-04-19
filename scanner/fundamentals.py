"""
SimFin fundamentals cache.

Pulls the last ~6 quarters of P&L for each ticker, computes:
- Revenue YoY growth (latest Q vs same Q 4 quarters ago)
- Net-income YoY growth with sign-aware loss/turnaround handling
- Revenue QoQ growth (latest Q vs prior Q)
- Revenue acceleration (current revenue YoY > prior quarter revenue YoY)
- Earnings acceleration (current net-income YoY > prior quarter net-income YoY)

Disk-cached with a 30-day TTL, but invalidated early after a known earnings
event so pre-report fundamentals do not survive into the post-report window.

API reference: https://backend.simfin.com/api/v3/
Auth:         Authorization: api-key <KEY>   (hyphenated!)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

CACHE_PATH = Path(os.environ.get("FUNDAMENTALS_CACHE", "public/cache/fundamentals.json"))
EARNINGS_CACHE_PATH = Path(os.environ.get("EARNINGS_CACHE", "public/cache/earnings.json"))
TTL_DAYS = 30
BASE_URL = "https://backend.simfin.com/api/v3"

UNKNOWN = None   # sentinel; None means "no fundamental data available"


def _auth_header() -> dict[str, str]:
    key = os.environ.get("SIMFIN_API_KEY", "").strip()
    if not key:
        return {}
    return {"Authorization": f"api-key {key}"}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _load_earnings_cache() -> dict:
    if EARNINGS_CACHE_PATH.exists():
        try:
            return json.loads(EARNINGS_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _fetch_pl_quarters(ticker: str, fyears: list[int]) -> list[dict]:
    """Return a list of per-quarter dicts (oldest to newest) for a ticker.
    Each dict: {'fiscal_period': 'Q1', 'fiscal_year': 2024, 'revenue': ..., 'net_income': ...}
    Returns [] on failure or no data.
    """
    headers = _auth_header()
    if not headers:
        return []
    quarters: list[dict] = []
    for fy in sorted(fyears):
        params = {
            "ticker": ticker,
            "statements": "pl",
            "period": "q1,q2,q3,q4",
            "fyear": str(fy),
        }
        try:
            r = requests.get(f"{BASE_URL}/companies/statements/compact",
                             params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                # Rate limited — back off briefly
                time.sleep(2.0 + random.random())
                continue
            if r.status_code != 200:
                continue
            payload = r.json()
            if not payload or not isinstance(payload, list):
                continue
            statements = payload[0].get("statements", [])
            pl = next((s for s in statements if s.get("statement") == "PL"), None)
            if not pl:
                continue
            columns = pl.get("columns") or []
            data = pl.get("data") or []
            col = {name: i for i, name in enumerate(columns)}
            rev_idx = col.get("Revenue")
            ni_idx = col.get("Net Income")
            period_idx = col.get("Fiscal Period")
            year_idx = col.get("Fiscal Year")
            if None in (rev_idx, ni_idx, period_idx, year_idx):
                continue
            for row in data:
                if len(row) <= max(rev_idx, ni_idx):
                    continue
                quarters.append({
                    "fiscal_period": row[period_idx],
                    "fiscal_year": row[year_idx],
                    "revenue": row[rev_idx],
                    "net_income": row[ni_idx],
                })
        except requests.RequestException:
            continue
    # Sort chronologically: year asc, then quarter number asc
    def _q_ord(q):
        period = q.get("fiscal_period") or ""
        try:
            qn = int(period.lstrip("qQ"))
        except ValueError:
            qn = 0
        return (q.get("fiscal_year") or 0, qn)
    quarters.sort(key=_q_ord)
    return quarters


def _compute_metrics(quarters: list[dict]) -> dict:
    """Given a chronologically-sorted list of quarter dicts, compute growth rates."""
    out = {
        "revenue_yoy_growth":    None,
        "net_income_yoy_growth": None,
        "revenue_qoq_growth":    None,
        "net_income_qoq_growth": None,
        "revenue_accelerating":  None,
        "earnings_accelerating": None,
        "latest_period":         None,
        "sample_quarters":       len(quarters),
    }
    if len(quarters) < 5:
        return out
    latest = quarters[-1]
    prior_q = quarters[-2]
    year_ago = quarters[-5]
    out["latest_period"] = f"{latest.get('fiscal_period')} {latest.get('fiscal_year')}"

    def _pct(a, b):
        if a is None or b is None or b == 0:
            return None
        try:
            return (a / b - 1) * 100
        except Exception:
            return None

    def _profit_pct(a, b):
        if a is None or b is None:
            return None
        try:
            a = float(a)
            b = float(b)
        except Exception:
            return None
        if b == 0:
            if a > 0:
                return 1000.0
            if a < 0:
                return -1000.0
            return 0.0
        if b > 0:
            return (a / b - 1) * 100
        if a >= 0:
            return ((a + abs(b)) / abs(b)) * 100
        return ((abs(b) - abs(a)) / abs(b)) * 100

    out["revenue_yoy_growth"]    = _pct(latest.get("revenue"),    year_ago.get("revenue"))
    out["net_income_yoy_growth"] = _profit_pct(latest.get("net_income"), year_ago.get("net_income"))
    out["revenue_qoq_growth"]    = _pct(latest.get("revenue"),    prior_q.get("revenue"))
    out["net_income_qoq_growth"] = _profit_pct(latest.get("net_income"), prior_q.get("net_income"))

    # "Accelerating" — current YoY growth exceeds prior quarter's YoY growth.
    if len(quarters) >= 6:
        prior_year_ago = quarters[-6]
        prior_rev_yoy = _pct(prior_q.get("revenue"), prior_year_ago.get("revenue"))
        curr_rev_yoy = out["revenue_yoy_growth"]
        if prior_rev_yoy is not None and curr_rev_yoy is not None:
            out["revenue_accelerating"] = curr_rev_yoy > prior_rev_yoy

        prior_ni_yoy = _profit_pct(prior_q.get("net_income"), prior_year_ago.get("net_income"))
        curr_ni_yoy = out["net_income_yoy_growth"]
        if prior_ni_yoy is not None and curr_ni_yoy is not None:
            out["earnings_accelerating"] = curr_ni_yoy > prior_ni_yoy
    return out


def _score(metrics: dict) -> int:
    """Simple 0-5 fundamental_score:
       +1 revenue YoY >= 15%
       +1 revenue YoY >= 25%  (big grower bonus)
       +1 net income YoY >= 20%
       +1 net income YoY >= 50%
       +1 earnings accelerating
    """
    score = 0
    rev = metrics.get("revenue_yoy_growth")
    if rev is not None:
        if rev >= 15: score += 1
        if rev >= 25: score += 1
    ni = metrics.get("net_income_yoy_growth")
    if ni is not None:
        if ni >= 20: score += 1
        if ni >= 50: score += 1
    if metrics.get("earnings_accelerating"):
        score += 1
    return score


def get_fundamentals_map(tickers: list[str], stale_days: int = TTL_DAYS) -> dict[str, dict]:
    """Return {ticker: {metrics..., fundamental_score}}.
    Empty dict for tickers without accessible data (keeps scan robust)."""
    if not _auth_header():
        print("  SimFin: SIMFIN_API_KEY not set — skipping fundamentals", file=sys.stderr)
        return {}

    cache = _load_cache()
    earnings_cache = _load_earnings_cache()
    now_ts = time.time()
    cutoff = now_ts - stale_days * 86400
    current_year = datetime.now().year
    # Fetch 3 fiscal years so we always have enough quarters for YoY + accel.
    # Companies with offset fiscal calendars (NVDA, AAPL) need extra coverage
    # when the current_year hasn't reported all four quarters yet.
    fyears = [current_year - 2, current_year - 1, current_year]

    result: dict[str, dict] = {}
    to_fetch: list[str] = []
    for t in tickers:
        entry = cache.get(t)
        earnings_entry = earnings_cache.get(t) or {}
        earnings_event_ts = earnings_entry.get("event_ts")
        stale_after_report = (
            isinstance(earnings_event_ts, (int, float))
            and entry is not None
            and entry.get("_ts", 0) < float(earnings_event_ts) <= now_ts
        )
        if entry and entry.get("_ts", 0) > cutoff and "fundamental_score" in entry and not stale_after_report:
            result[t] = {k: v for k, v in entry.items() if k != "_ts"}
        else:
            to_fetch.append(t)

    if to_fetch:
        print(f"  fundamentals cache: {len(tickers) - len(to_fetch)} hit, {len(to_fetch)} miss",
              file=sys.stderr)

    for idx, t in enumerate(to_fetch):
        quarters = _fetch_pl_quarters(t, fyears)
        metrics = _compute_metrics(quarters)
        record = {**metrics, "fundamental_score": _score(metrics), "_ts": now_ts}
        cache[t] = record
        result[t] = {k: v for k, v in record.items() if k != "_ts"}
        if idx < len(to_fetch) - 1:
            time.sleep(0.6)   # ~2 req/sec rate-limit hygiene (SimFin free tier)

    if to_fetch:
        _save_cache(cache)
    return result


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "NVDA", "MSFT"]
    out = get_fundamentals_map(tickers)
    print(json.dumps(out, indent=2))

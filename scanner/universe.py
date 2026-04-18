"""
Universe providers for the SEPA/Qullamaggie scanner.

Sources:
1. SEC EDGAR `company_tickers_exchange.json` — authoritative US equity list
   with exchange metadata. One HTTP call, cached 7 days. SEC requires a
   descriptive User-Agent (use an email).
2. Finviz pre-filter — narrows the universe by price, avg volume, and above-SMA
   before the expensive yfinance OHLCV pull. Cuts typical universe 5-10x.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Iterable

import pandas as pd
import requests

CACHE_DIR = os.environ.get("SCREEN_CACHE_DIR", "/tmp/sepa-scan/cache")
# SEC requires a real contact email in the UA per their fair-use policy.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "stock-screening tim@timli.net")
SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# SEC EDGAR uses these exchange strings:
EDGAR_EXCHANGE_MAP = {
    "nasdaq":  {"Nasdaq"},
    "nyse":    {"NYSE"},
    "amex":    {"NYSE American"},
    "arca":    {"NYSE Arca"},
    "cboe":    {"CBOE"},
}

# -------------------------- SEC EDGAR ----------------------------------------

def fetch_sec_edgar_universe(
    exchanges: Iterable[str] = ("nasdaq", "nyse"),
    cache_days: int = 7,
) -> list[str]:
    """Return sorted ticker list from SEC EDGAR, filtered by exchange."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "sec_edgar_universe.json")

    age_secs = time.time() - os.path.getmtime(cache_path) if os.path.exists(cache_path) else float("inf")
    if age_secs < cache_days * 86400:
        with open(cache_path) as f:
            payload = json.load(f)
    else:
        print(f"  fetching SEC EDGAR universe (UA: {SEC_USER_AGENT})...", file=sys.stderr)
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
        r = requests.get(SEC_URL, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        with open(cache_path, "w") as f:
            json.dump(payload, f)

    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    wanted = set()
    for ex in exchanges:
        wanted |= EDGAR_EXCHANGE_MAP.get(ex.lower(), set())
    if not wanted:
        return []
    df = df[df["exchange"].isin(wanted)]

    # Clean tickers: uppercase, drop share-class dots (BRK.B), strip junk
    tickers = set()
    for t in df["ticker"].dropna():
        t = str(t).strip().upper()
        if not t or "." in t or "$" in t or " " in t or "/" in t:
            continue
        tickers.add(t)
    return sorted(tickers)

# -------------------------- Finviz pre-filter --------------------------------

FINVIZ_EXCHANGE_TOKENS = {
    "nasdaq": "exch_nasd",
    "nyse":   "exch_nyse",
    "amex":   "exch_amex",
}

FINVIZ_URL = "https://finviz.com/screener.ashx"
FINVIZ_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

def finviz_prefilter(
    exchanges: Iterable[str] = ("nasdaq", "nyse"),
    min_price: float = 5,
    min_avg_vol: int = 200_000,
    above_sma: int | None = 50,
    performance_26w_over: float | None = None,
    max_results: int = 5000,
) -> list[str]:
    """
    Scrape the Finviz free screener to narrow the universe before bulk OHLCV.

    Args:
        exchanges: any of nasdaq, nyse, amex
        min_price: Finviz "Price - Over $X" filter. Tokens: 5=o5, 7=o7, 10=o10, 15=o15, 20=o20.
        min_avg_vol: avg volume filter (200k = o200, 500k = o500, 1M = o1000).
        above_sma: 20, 50, or 200 — adds "Price above SMAN" filter.
        performance_26w_over: half-year performance filter (e.g. 30 = up >30% 6mo).

    Returns sorted ticker list.
    """
    tokens: list[str] = []

    # Exchange filter: Finviz's free screener is single-value for exchange,
    # so piping two exchange tokens yields 0 results. We skip this filter and
    # rely on the base universe (SEC EDGAR) to constrain exchanges upstream.

    # Price bucket
    price_map = {5: "sh_price_o5", 7: "sh_price_o7", 10: "sh_price_o10",
                 15: "sh_price_o15", 20: "sh_price_o20", 30: "sh_price_o30",
                 50: "sh_price_o50", 100: "sh_price_o100"}
    if min_price in price_map:
        tokens.append(price_map[min_price])
    elif min_price:
        # Fall back to nearest bucket above
        bucket = min((k for k in price_map if k <= min_price), default=5, key=lambda k: abs(k - min_price))
        tokens.append(price_map[bucket])

    # Avg volume bucket
    vol_map = {100_000: "sh_avgvol_o100", 200_000: "sh_avgvol_o200",
               300_000: "sh_avgvol_o300", 500_000: "sh_avgvol_o500",
               750_000: "sh_avgvol_o750", 1_000_000: "sh_avgvol_o1000",
               2_000_000: "sh_avgvol_o2000"}
    if min_avg_vol in vol_map:
        tokens.append(vol_map[min_avg_vol])
    elif min_avg_vol:
        closest = min(vol_map.keys(), key=lambda k: abs(k - min_avg_vol))
        tokens.append(vol_map[closest])

    # Above SMA
    if above_sma == 20:
        tokens.append("ta_sma20_pa")
    elif above_sma == 50:
        tokens.append("ta_sma50_pa")
    elif above_sma == 200:
        tokens.append("ta_sma200_pa")

    # Performance 26-week
    if performance_26w_over is not None:
        if performance_26w_over >= 50:
            tokens.append("ta_perf2_6m50o")
        elif performance_26w_over >= 30:
            tokens.append("ta_perf2_6m30o")
        elif performance_26w_over >= 20:
            tokens.append("ta_perf2_6m20o")
        elif performance_26w_over >= 10:
            tokens.append("ta_perf2_6m10o")

    f = ",".join(tokens)
    print(f"  Finviz filter: f={f}", file=sys.stderr)

    session = requests.Session()
    session.headers.update({"User-Agent": FINVIZ_UA, "Accept": "text/html"})

    tickers: set[str] = []
    tickers = set()
    page = 1
    start = 1
    while start <= max_results:
        params = {"v": "111", "f": f, "r": str(start)}
        r = session.get(FINVIZ_URL, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  Finviz returned {r.status_code} on page {page}; stopping", file=sys.stderr)
            break
        # Parse ticker anchors: <a ... class="screener-link-primary">TICKER</a>
        # Robust regex avoids a full HTML parser dep.
        import re
        found = re.findall(r'class="screener-link-primary"[^>]*>([A-Z][A-Z0-9\-\.]*)</a>', r.text)
        if not found:
            break
        tickers.update(found)
        start += 20
        page += 1
        # Finviz rate-limits aggressively; short sleep between pages.
        time.sleep(0.5)
    return sorted(tickers)

# -------------------------- Combined dispatcher ------------------------------

def resolve_universe(
    source: str = "sec_edgar",
    exchanges: Iterable[str] = ("nasdaq", "nyse"),
    use_finviz_prefilter: bool = False,
    min_price: float = 5,
    min_avg_vol: int = 200_000,
    above_sma: int | None = 50,
) -> list[str]:
    """Return the ticker list for scanning.

    source: 'sec_edgar' (default, authoritative) or 'nasdaq_trader' (fallback).
    use_finviz_prefilter: if True, intersect with Finviz's quick-filter output
        to cut the universe before expensive OHLCV pulls.
    """
    if source == "sec_edgar":
        try:
            base = fetch_sec_edgar_universe(exchanges=exchanges)
        except Exception as e:
            print(f"  SEC EDGAR failed ({e}); falling back to Nasdaq Trader", file=sys.stderr)
            from sepa_scan_universe import fetch_universe as legacy_fetch
            base = legacy_fetch(exchanges)
            source = "nasdaq_trader (fallback)"
    elif source == "nasdaq_trader":
        from sepa_scan_universe import fetch_universe as legacy_fetch
        base = legacy_fetch(exchanges)
    else:
        raise ValueError(f"unknown universe source: {source}")

    print(f"  base universe ({source}): {len(base)}", file=sys.stderr)

    if use_finviz_prefilter:
        try:
            fv = finviz_prefilter(
                exchanges=exchanges,
                min_price=min_price,
                min_avg_vol=min_avg_vol,
                above_sma=above_sma,
            )
        except Exception as e:
            print(f"  Finviz pre-filter failed ({e}); using full base universe", file=sys.stderr)
            return base
        print(f"  Finviz candidates: {len(fv)}", file=sys.stderr)
        if len(fv) < 50:
            # Suspiciously small — likely a broken filter or Finviz blocked us.
            # Fall back to the full base universe rather than crashing downstream.
            print(f"  too few Finviz candidates; using full base universe", file=sys.stderr)
            return base
        base_set = set(base)
        intersected = sorted(t for t in fv if t in base_set)
        print(f"  after Finviz intersection: {len(intersected)}", file=sys.stderr)
        if len(intersected) < 50:
            print(f"  intersection too small; using full base universe", file=sys.stderr)
            return base
        return intersected

    return base

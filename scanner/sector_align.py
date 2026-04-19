"""
Sector alignment: bonus/penalty based on ticker's sector ETF rank.

A stock in a top-quintile sector gets +bonus to its setup; a stock in a
bottom-quintile sector gets -penalty. Sector strength is blended across 1-, 3-,
and 6-month returns to avoid overreacting to a single quarter. "Fish where the
fish are."

Uses sectors.json (produced by sector_scan.py) for per-sector ranking and the
description cache (public/cache/descriptions.json) for per-ticker sector lookup.
Unknown tickers get neutral (zero) bonus.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SECTORS_JSON = Path(os.environ.get("SECTORS_JSON", "public/results/latest/sectors.json"))
DESC_CACHE = Path(os.environ.get("DESC_CACHE", "public/cache/descriptions.json"))
SECTOR_BLEND_WEIGHTS = {
    "ret_1m": 0.2,
    "ret_3m": 0.5,
    "ret_6m": 0.3,
}

# GICS sector names in SEC/Yahoo terminology -> our sector_scan.py ETF names
SECTOR_NAME_MAP = {
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Financials": "Financials",
    "Technology": "Technology",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Cyclical": "Consumer Discretionary",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
}


def _percentile_ranks(rows: list[dict], metric: str) -> dict[str, float]:
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return {}
    rows.sort(key=lambda r: r.get(metric, -999), reverse=True)
    n = len(rows)
    return {r["name"]: ((n - idx) / n) * 100 for idx, r in enumerate(rows)}


def load_sector_ranks() -> dict[str, float]:
    """Map {sector_display_name: blended percentile rank_0_to_100}."""
    if not SECTORS_JSON.exists():
        return {}
    with open(SECTORS_JSON) as f:
        data = json.load(f)
    sector_rows = [r for r in data.get("rows", []) if r.get("group") == "sector"]
    if not sector_rows:
        return {}

    blended = {r["name"]: 0.0 for r in sector_rows}
    total_weight = 0.0
    for metric, weight in SECTOR_BLEND_WEIGHTS.items():
        metric_ranks = _percentile_ranks(sector_rows, metric)
        if not metric_ranks:
            continue
        total_weight += weight
        for name, pct in metric_ranks.items():
            blended[name] = blended.get(name, 0.0) + weight * pct

    if total_weight > 0:
        return {name: round(score / total_weight, 2) for name, score in blended.items()}
    # Fallback if blended fields are missing.
    return _percentile_ranks(sector_rows, "ret_3m")


def load_description_cache() -> dict:
    if not DESC_CACHE.exists():
        return {}
    try:
        with open(DESC_CACHE) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def sector_bonus_for(ticker: str, sector_ranks: dict[str, float], desc_cache: dict) -> float:
    """Return bonus in [-1, +1] based on ticker's sector blended RS percentile.
    Unknown ticker => 0 (neutral)."""
    info = desc_cache.get(ticker, {})
    sector_raw = info.get("sector") or ""
    sector_mapped = SECTOR_NAME_MAP.get(sector_raw)
    if not sector_mapped:
        return 0.0
    pct = sector_ranks.get(sector_mapped)
    if pct is None:
        return 0.0
    # Center at 50 so median sector gets 0, top ~= +1, bottom ~= -1
    return round((pct - 50) / 50.0, 2)


def sector_name_for(ticker: str, desc_cache: dict) -> str | None:
    info = desc_cache.get(ticker, {})
    sector_raw = info.get("sector") or ""
    return SECTOR_NAME_MAP.get(sector_raw)


if __name__ == "__main__":
    import sys
    ranks = load_sector_ranks()
    desc = load_description_cache()
    tickers = sys.argv[1:] or list(desc.keys())[:10]
    for t in tickers:
        b = sector_bonus_for(t, ranks, desc)
        s = sector_name_for(t, desc)
        print(f"{t}: sector={s} bonus={b:+.2f}")

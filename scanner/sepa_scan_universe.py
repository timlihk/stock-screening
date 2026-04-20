"""
Parameterized universe SEPA scanner.

Pipeline:
  1. Fetch symbol directory from Nasdaq Trader (Nasdaq + other-listed).
  2. Filter by exchange, drop warrants/units/preferreds/rights, optional ETFs.
  3. Pass 1: batch yfinance (1mo), compute 20-day avg dollar volume,
     filter by min_price and min_adv_usd.
  4. Pass 2: batch yfinance (15mo) on survivors, run full SEPA trend template,
     VCP-lite signals, Deepvue/Minervini extensions, then overlay vendored
     Andy-Roger breakout / episodic-pivot QM picks.
  5. Write results CSV + print the shortlist.

Usage:
    python3 sepa_scan_universe.py \\
        --exchanges nasdaq,nyse \\
        --min-price 5 \\
        --min-adv-usd 50000000 \\
        --min-setup-score 3 \\
        --use-qullamaggie \\
        --date 2026-04-18
"""
import argparse, json, os, sys, time, warnings
import os.path
from io import StringIO

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from universe import resolve_universe
    HAVE_UNIVERSE_MODULE = True
except ImportError:
    HAVE_UNIVERSE_MODULE = False

try:
    from sector_align import (
        load_description_cache as load_sector_description_cache,
        load_sector_ranks,
        sector_bonus_for,
        sector_name_for,
    )
    HAVE_SECTOR_ALIGN = True
except ImportError:
    HAVE_SECTOR_ALIGN = False

try:
    from qm_strategies import apply_vendor_qm_scores
    HAVE_QM_VENDOR = True
except ImportError:
    HAVE_QM_VENDOR = False

BENCH = ["SPY", "QQQ", "IWM"]
OUT_DIR_DEFAULT = "/tmp/sepa-scan"

# Setup family calibration. Thresholds are the minimum score to "qualify" as
# that archetype. The QM families are sourced from Andy-Roger's 0-100 scanner
# scores and then mapped into this repo's 0-8 family frame so the regime gate
# and unified shortlist can still compare families coherently.
FAMILY_THRESHOLDS = {
    "sepa_vcp":        6,   # 0-8 scale
    "power_play":      6,   # 0-8
    "qm_breakout":     6,   # 0-8
    "qm_episodic_pivot": 6, # 0-8
}
FAMILY_MAX_SCORES = {
    "sepa_vcp":        8,
    "power_play":      8,
    "qm_breakout":     8,
    "qm_episodic_pivot": 8,
}

# ---------------------------- Universe ---------------------------------------

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

OTHER_EXCHANGE_CODES = {
    # Nasdaq Trader otherlisted.txt exchange codes
    "nyse":  ["N"],   # NYSE
    "amex":  ["A"],   # NYSE American (formerly AMEX)
    "arca":  ["P"],   # NYSE Arca
    "bats":  ["Z"],   # Cboe BZX
    "iex":   ["V"],
}

EXCLUDE_PATTERNS = "Warrant|Unit|Preferred|Depositary|Right|Note|Debenture|Subordinated|Bond"

def fetch_universe(exchanges, include_etf=False):
    """Return sorted list of tickers from chosen exchanges."""
    tickers = []

    # Nasdaq-listed file
    if "nasdaq" in exchanges:
        r = requests.get(NASDAQ_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep="|")
        df = df[df["Symbol"].notna() & (df["Test Issue"] == "N")].copy()
        if not include_etf and "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]
        df = df[~df["Security Name"].str.contains(EXCLUDE_PATTERNS, case=False, na=False, regex=True)]
        tickers.extend(df["Symbol"].tolist())

    # Other-listed file (NYSE, AMEX, ARCA, BATS)
    wanted_codes = []
    for ex in exchanges:
        wanted_codes.extend(OTHER_EXCHANGE_CODES.get(ex, []))
    if wanted_codes:
        r = requests.get(OTHER_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep="|")
        df = df[df["ACT Symbol"].notna() & (df["Test Issue"] == "N")].copy()
        df = df[df["Exchange"].isin(wanted_codes)]
        if not include_etf and "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]
        df = df[~df["Security Name"].str.contains(EXCLUDE_PATTERNS, case=False, na=False, regex=True)]
        tickers.extend(df["ACT Symbol"].tolist())

    # Normalize: uppercase, drop share-class dots (BRK.B) and $, dedupe, sort
    cleaned = set()
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper()
        if not t or "$" in t or "." in t or " " in t or "/" in t or t.endswith("W"):
            # ^W excludes many warrants but may drop a few genuine tickers — acceptable for now
            continue
        cleaned.add(t)
    return sorted(cleaned)

# ---------------------------- Pass 1 liquidity -------------------------------

def liquidity_pass(tickers, min_price, min_adv_usd, batch=400):
    """Batch-fetch 1-month OHLCV; return DataFrame of survivors with price + 20d ADV ($)."""
    survivors = []
    n_batches = (len(tickers) + batch - 1) // batch
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        bidx = i // batch + 1
        print(f"  liquidity batch {bidx}/{n_batches}: {len(chunk)} tickers", file=sys.stderr)
        try:
            data = yf.download(chunk, period="2mo", auto_adjust=True,
                               progress=False, threads=True, group_by="ticker")
        except Exception as e:
            print(f"    batch failed: {e}", file=sys.stderr)
            continue
        for t in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[t]
                else:
                    df = data
                close = df["Close"].dropna()
                vol = df["Volume"].dropna()
                if len(close) < 15:
                    continue
                price = close.iloc[-1]
                if price < min_price:
                    continue
                adv_usd = (close * vol).tail(20).mean()
                if not np.isfinite(adv_usd) or adv_usd < min_adv_usd:
                    continue
                survivors.append(dict(ticker=t, price=price, adv_usd=adv_usd))
            except Exception:
                continue
    return pd.DataFrame(survivors)

# ---------------------------- Pass 2 SEPA scan -------------------------------
# Reused logic, trimmed from sepa_scan_20260418.py. Keep in sync.

def pullback_sequence(close, lookback=84):
    c = close.iloc[-lookback:]
    if len(c) < 20:
        return []
    pullbacks, direction, ext, last_high = [], None, c.iloc[0], c.iloc[0]
    for v in c.iloc[1:]:
        if direction is None:
            if v > ext * 1.03:
                direction = "up"; last_high = v; ext = v
            elif v < ext * 0.97:
                direction = "down"; ext = v
        elif direction == "up":
            if v > ext:
                ext = v; last_high = v
            elif v < ext * 0.97:
                direction = "down"; ext = v
        elif direction == "down":
            if v < ext:
                ext = v
            elif v > ext * 1.03:
                pullbacks.append((last_high - ext) / last_high * 100)
                direction = "up"; last_high = v; ext = v
    if direction == "down":
        depth = (last_high - ext) / last_high * 100
        if depth > 0:
            pullbacks.append(depth)
    return pullbacks

def weekly_range_pct(close, high, low, weeks=6):
    df = pd.DataFrame({"Close": close, "High": high, "Low": low})
    w = df.resample("W-FRI").agg({"Close": "last", "High": "max", "Low": "min"}).dropna()
    if len(w) < weeks:
        return []
    last = w.iloc[-weeks:]
    return ((last["High"] - last["Low"]) / last["Close"] * 100).tolist()

def weekly_volume(vol, weeks=6):
    w = vol.resample("W-FRI").sum().dropna()
    if len(w) < weeks:
        return []
    return w.iloc[-weeks:].tolist()

def atr(high, low, close, window):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]

def hh_hl_stats(high, low, window=60):
    """Count sessions making higher high AND higher low over the last `window` days.
    Returns (count, pct). Orderly uptrends have high ratios (>=40%).
    Inspired by VladPetrariu/Qullamaggie-breakout-scanner's HH/HL structure score."""
    if len(high) < window + 1:
        return 0, 0.0
    h = high.iloc[-(window + 1):]
    l = low.iloc[-(window + 1):]
    hh = (h.values[1:] > h.values[:-1])
    hl = (l.values[1:] > l.values[:-1])
    count = int((hh & hl).sum())
    return count, count / window * 100

def candle_quality_stats(open_, high, low, close, window=21):
    """Average body/range ratio over the last `window` days.
    Vlad's tiers: 0.52+ ideal, 0.44+ ok, 0.38+ barcode, <0.38 poor.
    Filters out choppy 'barcode' price action.
    Also returns ABR% (average body range as % of price)."""
    if len(close) < window:
        return np.nan, np.nan
    o = open_.iloc[-window:]; h = high.iloc[-window:]
    l = low.iloc[-window:];   c = close.iloc[-window:]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    body_range_ratio = (body / rng).mean()
    abr_pct = (body / c).mean() * 100
    return float(body_range_ratio), float(abr_pct)


def close_in_range_stats(high, low, close, window=10):
    """Average close location within each day's range over the trailing window."""
    if len(close) < window:
        return np.nan
    h = high.iloc[-window:]
    l = low.iloc[-window:]
    c = close.iloc[-window:]
    rng = (h - l).replace(0, np.nan)
    loc = ((c - l) / rng).clip(lower=0, upper=1)
    return float(loc.mean())


def volume_pressure_stats(open_, close, vol, window=20):
    """Return up/down volume ratio over the trailing window."""
    if len(close) < window or len(vol) < window:
        return np.nan
    o = open_.iloc[-window:]
    c = close.iloc[-window:]
    v = vol.iloc[-window:]
    up_vol = v[c >= o].sum()
    down_vol = v[c < o].sum()
    if down_vol <= 0:
        return np.inf if up_vol > 0 else np.nan
    return float(up_vol / down_vol)


def distribution_day_count(close, vol, window=20):
    """Count high-volume down days, a simple proxy for institutional selling."""
    if len(close) < window + 1 or len(vol) < window + 1:
        return 0
    c = close.iloc[-(window + 1):]
    v = vol.iloc[-(window + 1):]
    down = c.pct_change() < -0.002
    heavier = v > v.shift(1)
    return int((down & heavier).iloc[1:].sum())


def quiet_down_weeks(close, vol, weeks=6):
    """Count down weeks that occur on lighter volume than the prior week."""
    df = pd.DataFrame({"Close": close, "Volume": vol})
    w = df.resample("W-FRI").agg({"Close": "last", "Volume": "sum"}).dropna()
    if len(w) < weeks + 1:
        return 0
    last = w.iloc[-(weeks + 1):]
    down = last["Close"].diff() < 0
    lighter = last["Volume"] < last["Volume"].shift(1)
    return int((down & lighter).iloc[1:].sum())


def recent_expansion_profile(close, high, low, vol, lookback=15):
    """Detect a recent expansion day followed by constructive digestion."""
    if len(close) < lookback + 21:
        return {"recent_expansion": False, "post_expansion_tight": False}

    avg_vol_20 = vol.iloc[-(lookback + 20):-lookback].mean()
    recent = pd.DataFrame({
        "close": close.iloc[-lookback:],
        "high": high.iloc[-lookback:],
        "low": low.iloc[-lookback:],
        "vol": vol.iloc[-lookback:],
    })
    day_ret = recent["close"].pct_change().fillna(0)
    vol_ratio = recent["vol"] / avg_vol_20 if avg_vol_20 > 0 else np.nan
    expansion_mask = (day_ret >= 0.05) & (vol_ratio >= 1.8)
    if not expansion_mask.any():
        return {"recent_expansion": False, "post_expansion_tight": False}

    anchor_idx = recent.index[expansion_mask][-1]
    post = recent.loc[anchor_idx:]
    ref_close = post["close"].iloc[0]
    drawdown_pct = (post["low"].min() / ref_close - 1) * 100
    rebound_range_pct = (post["high"].max() / post["low"].min() - 1) * 100
    return {
        "recent_expansion": True,
        "post_expansion_tight": drawdown_pct >= -10 and rebound_range_pct <= 18,
    }


def base_structure(close, high, low):
    """Pick the best recent base and return its pivot/shape metrics."""
    price = close.iloc[-1]
    best = None
    for length in (25, 35, 45, 55, 65):
        if len(close) < length:
            continue
        c = close.iloc[-length:]
        h = high.iloc[-length:]
        l = low.iloc[-length:]
        base_high = float(h.max())
        base_low = float(l.min())
        if base_high <= 0 or base_high <= base_low:
            continue
        depth_pct = (base_high - base_low) / base_high * 100
        if depth_pct < 6 or depth_pct > 40:
            continue
        last10_high = float(h.iloc[-10:].max())
        last10_low = float(l.iloc[-10:].min())
        tightness_pct = (last10_high - last10_low) / price * 100 if price > 0 else np.nan
        close_position = (price - base_low) / (base_high - base_low)
        pivot_pos = int(np.argmax(h.values))
        pivot_age = length - 1 - pivot_pos
        pct_to_pivot = (price / base_high - 1) * 100
        score = abs(depth_pct - 18) + tightness_pct + abs(min(pct_to_pivot, 0)) * 1.5
        if close_position < 0.6 or pivot_age < 3:
            score += 100
        candidate = {
            "base_length": length,
            "base_depth_pct": depth_pct,
            "base_tightness_pct": tightness_pct,
            "base_close_position": close_position,
            "pivot": base_high,
            "pivot_age": pivot_age,
            "pct_to_pivot": pct_to_pivot,
            "base_valid": close_position >= 0.6 and pivot_age >= 3 and tightness_pct <= 15,
            "base_score": score,
        }
        if best is None or candidate["base_score"] < best["base_score"]:
            best = candidate
    if best is None:
        return {
            "base_length": np.nan,
            "base_depth_pct": np.nan,
            "base_tightness_pct": np.nan,
            "base_close_position": np.nan,
            "pivot": np.nan,
            "pivot_age": np.nan,
            "pct_to_pivot": np.nan,
            "base_valid": False,
            "base_score": np.nan,
        }
    return best


def classify_setup_families(df, use_qullamaggie=True):
    """Recompute family membership after QM scores have been merged in."""
    if df.empty:
        return df

    stale_cols = [
        "sepa_vcp_qualifies",
        "power_play_qualifies",
        "qm_breakout_qualifies",
        "qm_episodic_pivot_qualifies",
        "qualified_count",
        "also_fits",
        "primary_setup",
        "primary_setup_score",
        "setup_score",
        "primary_setup_valid",
        "best_family_excess",
        "best_family_pct",
    ]
    df = df.drop(columns=[c for c in stale_cols if c in df.columns]).copy()

    def _classify(row):
        family_scores = {
            "sepa_vcp": int(row.get("sepa_vcp_score", 0) or 0),
            "power_play": int(row.get("power_play_score", 0) or 0),
            "qm_breakout": int(row.get("qm_breakout_score", 0) or 0) if use_qullamaggie else 0,
            "qm_episodic_pivot": int(row.get("qm_episodic_pivot_score", 0) or 0) if use_qullamaggie else 0,
        }
        family_excess = {
            name: family_scores[name] - FAMILY_THRESHOLDS[name]
            for name in FAMILY_THRESHOLDS
        }
        family_pct_max = {
            name: family_scores[name] / FAMILY_MAX_SCORES[name]
            for name in FAMILY_MAX_SCORES
        }
        qualifies_as = {
            name: family_scores[name] >= FAMILY_THRESHOLDS[name]
            for name in FAMILY_THRESHOLDS
        }
        qualified = [name for name, passes in qualifies_as.items() if passes]
        if qualified:
            primary_setup = max(
                qualified,
                key=lambda name: (family_excess[name], family_scores[name]),
            )
            also_fits = [name for name in qualified if name != primary_setup]
        else:
            primary_setup = "none"
            also_fits = []
        primary_setup_valid = primary_setup != "none"
        setup_score = family_scores[primary_setup] if primary_setup_valid else 0
        return pd.Series(
            {
                "sepa_vcp_qualifies": qualifies_as["sepa_vcp"],
                "power_play_qualifies": qualifies_as["power_play"],
                "qm_breakout_qualifies": qualifies_as["qm_breakout"],
                "qm_episodic_pivot_qualifies": qualifies_as["qm_episodic_pivot"],
                "qualified_count": len(qualified),
                "also_fits": ",".join(also_fits) if also_fits else "",
                "primary_setup": primary_setup,
                "primary_setup_score": setup_score,
                "setup_score": setup_score,
                "primary_setup_valid": primary_setup_valid,
                "best_family_excess": max(family_excess.values()),
                "best_family_pct": max(family_pct_max.values()),
            }
        )

    classified = df.apply(_classify, axis=1)
    return pd.concat([df, classified], axis=1)

def analyze(tkr, df, spy_ret_1y, use_qullamaggie=True):
    if df is None or df.empty:
        return None
    df = df.dropna(how="all")
    close = df["Close"].dropna(); high = df["High"].dropna()
    low = df["Low"].dropna();     vol  = df["Volume"].dropna()
    open_ = df["Open"].dropna() if "Open" in df.columns else close
    if len(close) < 220:
        return None

    price = close.iloc[-1]
    ma50  = close.rolling(50).mean().iloc[-1]
    ma150 = close.rolling(150).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    ma200_1m = close.rolling(200).mean().iloc[-22] if len(close) >= 222 else np.nan
    hi52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    lo52 = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    pct_from_hi = (price / hi52 - 1) * 100
    pct_above_lo = (price / lo52 - 1) * 100
    ret_1y = (price / close.iloc[-252] - 1) * 100 if len(close) >= 252 else np.nan
    rs_vs_spy = ret_1y - spy_ret_1y if not np.isnan(ret_1y) else np.nan
    avg_vol_20 = vol.iloc[-21:-1].mean()
    vol_today = vol.iloc[-1]
    vol_ratio = vol_today / avg_vol_20 if avg_vol_20 > 0 else np.nan

    # 8-point trend template
    c1 = price > ma150 and price > ma200
    c2 = ma150 > ma200
    c3 = not np.isnan(ma200_1m) and ma200 > ma200_1m
    c4 = ma50 > ma150 and ma50 > ma200
    c5 = price > ma50
    c6 = pct_above_lo >= 30
    c7 = pct_from_hi >= -25
    c8 = not np.isnan(rs_vs_spy) and rs_vs_spy > 0
    all_pass = all([c1, c2, c3, c4, c5, c6, c7, c8])

    # VCP-lite
    pullbacks = pullback_sequence(close, 84)
    wrange = weekly_range_pct(close, high, low, 6)
    wvol = weekly_volume(vol, 6)
    pb_count = len(pullbacks)
    contracting_pbs = pb_count >= 3 and pullbacks[-3] > pullbacks[-2] > pullbacks[-1]
    contracting_range = len(wrange) >= 5 and np.mean(wrange[-2:]) < np.mean(wrange[:3]) * 0.7
    vol_dry_up = len(wvol) >= 5 and wvol[-1] < np.mean(wvol[:-1]) * 0.8
    # ATR is computed once with a single window (14) and reused everywhere.
    # Previously: atr(10)/atr(50) for atr_ratio and atr(14) separately for
    # not_extended_50ma — inconsistent. Unified to atr(14) and atr(50).
    atr14 = atr(high, low, close, 14)
    atr50 = atr(high, low, close, 50)
    atr_ratio = (atr14 / atr50) if (not np.isnan(atr14) and atr50 > 0) else np.nan
    atr_compression = not np.isnan(atr_ratio) and atr_ratio <= 0.65
    if len(close) >= 5:
        r5_hi, r5_lo = high.iloc[-5:].max(), low.iloc[-5:].min()
        range5_pct = (r5_hi - r5_lo) / price * 100
    else:
        range5_pct = np.nan
    tight_range = not np.isnan(range5_pct) and range5_pct <= 8
    ret_6m = (price / close.iloc[-126] - 1) * 100 if len(close) >= 126 else np.nan
    ret_15d = (price / close.iloc[-16] - 1) * 100 if len(close) >= 16 else np.nan
    ma20 = close.rolling(20).mean().iloc[-1]

    # Structure-quality signals (Vlad-inspired)
    hh_hl_count, hh_hl_pct = hh_hl_stats(high, low, window=60)
    hh_hl_orderly = hh_hl_pct >= 40            # >=40% of last 60 sessions are HH/HL
    candle_quality, abr21_pct = candle_quality_stats(open_, high, low, close, window=21)
    candle_orderly = not np.isnan(candle_quality) and candle_quality >= 0.44  # Vlad's "ok" tier
    avg_close_in_range = close_in_range_stats(high, low, close, window=10)
    close_in_upper_range = not np.isnan(avg_close_in_range) and avg_close_in_range >= 0.60
    up_down_vol_ratio = volume_pressure_stats(open_, close, vol, window=20)
    accumulation_support = (not np.isnan(up_down_vol_ratio) and up_down_vol_ratio >= 1.20)
    distribution_days = distribution_day_count(close, vol, window=20)
    quiet_pullback_weeks = quiet_down_weeks(close, vol, weeks=6)
    quiet_pullback = quiet_pullback_weeks >= 2
    expansion = recent_expansion_profile(close, high, low, vol, lookback=15)
    base = base_structure(close, high, low)
    pivot = base["pivot"]
    pct_to_pivot = base["pct_to_pivot"]
    breakout_ready = not np.isnan(pct_to_pivot) and -3.0 <= pct_to_pivot <= 2.0

    # Jeff Sun extensions (reuse unified atr14 from above)
    # 1) ATR% extension from 50-MA — entry ceiling at ≤ 4× (anti-chase)
    atr50ma_ext = ((price - ma50) / atr14) if (not np.isnan(atr14) and atr14 > 0 and not np.isnan(ma50)) else np.nan
    not_extended_50ma = not np.isnan(atr50ma_ext) and atr50ma_ext <= 4.0
    # 2) 1-week compression — price range within ±5% over last 5 sessions
    ret_5d = (price / close.iloc[-6] - 1) * 100 if len(close) >= 6 else np.nan
    compression_1w = not np.isnan(ret_5d) and abs(ret_5d) <= 5

    # Consolidated tightness (0-3 ordinal): merges the three partly-redundant
    # "tight" signals into one human-readable measure. Each point is a distinct
    # dimension of tightness:
    #   +1 price range (5-day high-low as % of price)
    #   +1 ATR compression (atr14 / atr50 baseline)
    #   +1 directional compression (abs 5-day return)
    tightness_score = int(tight_range) + int(atr_compression) + int(compression_1w)

    # Leadership score (0-12) — tiered instead of flat 0/1 checks so that
    # *bigger* movers with *cleaner* structure separate from marginal leaders.
    # Each return-based dimension gets 0-3 based on which threshold bucket it
    # clears; structure signals stay binary.
    def _tier(value, bins):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        return sum(1 for b in bins if value >= b)

    leadership_score = (
        int(all_pass)                                    # gate: 0/1
        + _tier(ret_6m,     [30, 60, 100])               # 0-3: 30/60/100% 6-mo ret
        + _tier(pct_from_hi, [-12, -5, -1])              # 0-3: within 12/5/1% of 52wH
        + int(hh_hl_orderly)                             # 0/1
        + int(price > ma20)                              # 0/1
        + _tier(rs_vs_spy,  [10, 50, 100])               # 0-3: RS outperformance tiers
    )
    entry_score = sum([
        base["base_valid"],
        breakout_ready,
        quiet_pullback,
        close_in_upper_range,
        accumulation_support,
        distribution_days <= 2,
        not_extended_50ma,
    ])
    support_score = int(close_in_upper_range) + int(accumulation_support) + int(distribution_days <= 2)

    sepa_vcp_score = sum([
        all_pass,
        base["base_valid"],
        pb_count >= 2,
        contracting_pbs,
        contracting_range,
        quiet_pullback or vol_dry_up,
        breakout_ready,
        support_score >= 2,
    ])
    power_play_digest = quiet_pullback or vol_dry_up or expansion["post_expansion_tight"]
    power_play_score = sum([
        not np.isnan(ret_6m) and ret_6m >= 85,
        not np.isnan(ret_15d) and -12 <= ret_15d <= 8,
        pct_from_hi >= -12,
        price > ma20,
        tightness_score >= 2 or base["base_tightness_pct"] <= 12,
        power_play_digest,
        not_extended_50ma,
        support_score >= 2,
    ])

    out = dict(
        ticker=tkr, price=price,
        pct_from_hi=pct_from_hi, pct_above_lo=pct_above_lo,
        ret_1y=ret_1y, rs_vs_spy=rs_vs_spy,
        vol_ratio=vol_ratio, all_pass=all_pass,
        atr_ratio=atr_ratio, range5_pct=range5_pct,
        ret_6m=ret_6m, ret_15d=ret_15d,
        contracting_pbs=contracting_pbs, vol_dry_up=vol_dry_up,
        atr_compression=atr_compression, tight_range=tight_range,
        hh_hl_count=hh_hl_count, hh_hl_pct=hh_hl_pct, hh_hl_orderly=hh_hl_orderly,
        candle_quality=candle_quality, abr21_pct=abr21_pct, candle_orderly=candle_orderly,
        atr50ma_ext=atr50ma_ext, not_extended_50ma=not_extended_50ma,
        ret_5d=ret_5d, compression_1w=compression_1w,
        tightness_score=tightness_score, support_score=support_score,
        avg_close_in_range=avg_close_in_range, close_in_upper_range=close_in_upper_range,
        up_down_vol_ratio=up_down_vol_ratio, accumulation_support=accumulation_support,
        distribution_days=distribution_days, quiet_pullback_weeks=quiet_pullback_weeks,
        quiet_pullback=quiet_pullback,
        recent_expansion=expansion["recent_expansion"],
        post_expansion_tight=expansion["post_expansion_tight"],
        base_length=base["base_length"], base_depth_pct=base["base_depth_pct"],
        base_tightness_pct=base["base_tightness_pct"], base_close_position=base["base_close_position"],
        pivot_age=base["pivot_age"], breakout_ready=breakout_ready,
        leadership_score=leadership_score, entry_score=entry_score,
        sepa_vcp_score=sepa_vcp_score, power_play_score=power_play_score,
        pct_to_pivot=pct_to_pivot, pivot=pivot,
    )
    return out

def full_scan(tickers, spy_ret_1y, use_qullamaggie=True, batch=400):
    rows = []
    n_batches = (len(tickers) + batch - 1) // batch
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        bidx = i // batch + 1
        print(f"  scan batch {bidx}/{n_batches}: {len(chunk)} tickers", file=sys.stderr)
        try:
            data = yf.download(chunk, period="15mo", auto_adjust=True,
                               progress=False, threads=True, group_by="ticker")
        except Exception as e:
            print(f"    batch failed: {e}", file=sys.stderr)
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                r = analyze(t, df, spy_ret_1y, use_qullamaggie=use_qullamaggie)
                if r is not None:
                    rows.append(r)
            except Exception:
                continue
    return pd.DataFrame(rows)

# ---------------------------- Main -------------------------------------------

def env_stats(c, label):
    ma200 = c.rolling(200).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    price = c.iloc[-1]
    ret_1y = (price / c.iloc[-252] - 1) * 100
    return dict(label=label, price=price, ma50=ma50, ma200=ma200,
                above_ma200=price > ma200, above_ma50=price > ma50, ret_1y=ret_1y)


def market_regime_profile(spy, qqq, iwm):
    score = sum([
        spy["above_ma50"], spy["above_ma200"],
        qqq["above_ma50"], qqq["above_ma200"],
        iwm["above_ma50"],
        spy["ret_1y"] > 0, qqq["ret_1y"] > 0,
    ])
    # leadership_score now ranges 0-12 (tiered). Thresholds scale accordingly
    # from the prior 0-6 scale (risk_on 4 -> 6, risk_off 5 -> 8).
    if score >= 6:
        label = "risk_on"
        min_setup = 6
        min_entry = 4
        min_leadership = 6
    elif score >= 4:
        label = "mixed"
        min_setup = 6
        min_entry = 5
        min_leadership = 7
    else:
        label = "risk_off"
        min_setup = 7
        min_entry = 5
        min_leadership = 8
    return {
        "label": label,
        "score": score,
        "min_setup": min_setup,
        "min_entry": min_entry,
        "min_leadership": min_leadership,
    }

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exchanges", default="nasdaq,nyse",
                   help="Comma-separated: nasdaq, nyse, amex, arca, bats")
    p.add_argument("--universe-source", default="sec_edgar",
                   choices=["sec_edgar", "nasdaq_trader"])
    p.add_argument("--use-finviz-prefilter", action="store_true",
                   help="Pre-filter universe via Finviz (much faster)")
    p.add_argument("--finviz-above-sma", type=int, default=50, choices=[0, 20, 50, 200])
    p.add_argument("--institutional-buying", action="store_true",
                   help="Finviz: only include stocks with positive institutional transactions")
    p.add_argument("--exclude-biotech", action="store_true",
                   help="Post-scan: drop biotech names (uses description cache for sector/industry)")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-adv-usd", type=float, default=50e6,
                   help="Min 20-day avg dollar volume ($)")
    p.add_argument("--include-etf", action="store_true")
    p.add_argument("--min-setup-score", type=int, default=5,
                   help="Minimum family setup score (0-8). Default 5.")
    p.add_argument("--use-qullamaggie", action="store_true", default=True)
    p.add_argument("--no-qullamaggie", dest="use_qullamaggie", action="store_false")
    p.add_argument("--max-tickers", type=int, default=0,
                   help="Cap universe size for testing (0 = no cap)")
    p.add_argument("--output-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    return p.parse_args()

def main():
    args = parse_args()
    exchanges = [e.strip().lower() for e in args.exchanges.split(",") if e.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    date_tag = args.date.replace("-", "")
    t0 = time.time()

    print(f"[1/4] Fetching benchmark data...", file=sys.stderr)
    bench = yf.download(BENCH, period="15mo", auto_adjust=True,
                        progress=False, threads=True, group_by="ticker")
    spy = env_stats(bench["SPY"]["Close"].dropna(), "SPY")
    qqq = env_stats(bench["QQQ"]["Close"].dropna(), "QQQ")
    iwm = env_stats(bench["IWM"]["Close"].dropna(), "IWM")
    print("\n=== MARKET ENVIRONMENT ===")
    for e in (spy, qqq, iwm):
        print(f"{e['label']}: price={e['price']:.2f}  50MA={e['ma50']:.2f}  200MA={e['ma200']:.2f}  "
              f"above200={e['above_ma200']}  above50={e['above_ma50']}  1y={e['ret_1y']:+.1f}%")
    regime = market_regime_profile(spy, qqq, iwm)
    print(f"Regime: {regime['label']} (score={regime['score']}/7, "
          f"min_setup={regime['min_setup']}, min_entry={regime['min_entry']}, "
          f"min_leadership={regime['min_leadership']})")

    print(f"\n[2/4] Fetching universe (source={args.universe_source}, exchanges={', '.join(exchanges)})...",
          file=sys.stderr)
    if HAVE_UNIVERSE_MODULE:
        universe = resolve_universe(
            source=args.universe_source,
            exchanges=exchanges,
            use_finviz_prefilter=args.use_finviz_prefilter,
            min_price=args.min_price,
            min_avg_vol=200_000,
            above_sma=args.finviz_above_sma or None,
            institutional_buying=args.institutional_buying,
        )
    else:
        universe = fetch_universe(exchanges, include_etf=args.include_etf)
    if args.max_tickers:
        universe = universe[:args.max_tickers]
    print(f"  universe size: {len(universe)}", file=sys.stderr)

    print(f"\n[3/4] Liquidity pass (price >= ${args.min_price:.2f}, ADV >= ${args.min_adv_usd:,.0f})...",
          file=sys.stderr)
    if not universe:
        print("  ERROR: universe is empty; cannot continue", file=sys.stderr)
        # Write an empty results file so downstream steps can gracefully report no data.
        pd.DataFrame(columns=["ticker", "price", "all_pass", "setup_score"]).to_csv(
            f"{args.output_dir}/results_universe_{date_tag}.csv", index=False)
        sys.exit(2)

    liq = liquidity_pass(universe, args.min_price, args.min_adv_usd)
    liq.to_csv(f"{args.output_dir}/universe_liquid_{date_tag}.csv", index=False)
    print(f"  liquid survivors: {len(liq)}", file=sys.stderr)

    if liq.empty:
        print("  ERROR: no tickers passed liquidity filter", file=sys.stderr)
        pd.DataFrame(columns=["ticker", "price", "all_pass", "setup_score"]).to_csv(
            f"{args.output_dir}/results_universe_{date_tag}.csv", index=False)
        sys.exit(3)

    print(f"\n[4/4] Full SEPA scan on {len(liq)} survivors (Qullamaggie={args.use_qullamaggie})...",
          file=sys.stderr)
    df = full_scan(liq["ticker"].tolist(), spy["ret_1y"],
                   use_qullamaggie=args.use_qullamaggie)
    df = df.merge(liq[["ticker", "adv_usd"]], on="ticker", how="left")

    # Universe-relative RS: percentile rank (0-100) of ret_1y across all
    # analyzed tickers. Replaces plain "rs_vs_spy" as the cleaner momentum
    # ranking per IBD/Qullamaggie methodology.
    if "ret_1y" in df.columns:
        df["rs_pct_rank"] = df["ret_1y"].rank(pct=True, method="average") * 100
        df["rs_top_20"] = df["rs_pct_rank"] >= 80   # top quintile
    else:
        df["rs_pct_rank"] = np.nan
        df["rs_top_20"] = False

    if args.use_qullamaggie:
        if not HAVE_QM_VENDOR:
            print("  WARNING: vendored QM scanner unavailable; QM families disabled", file=sys.stderr)
        else:
            print("  overlaying vendored Andy-Roger QM breakout / EP scan", file=sys.stderr)
            df = apply_vendor_qm_scores(df, liq["ticker"].tolist())
    for col in ("qm_breakout_vendor_score", "qm_episodic_pivot_vendor_score"):
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)
    df = classify_setup_families(df, use_qullamaggie=args.use_qullamaggie and HAVE_QM_VENDOR)

    df["market_regime"] = regime["label"]
    df["regime_score"] = regime["score"]
    df["regime_eligible"] = (
        df["primary_setup_valid"]
        & (df["setup_score"] >= np.maximum(args.min_setup_score, regime["min_setup"]))
        & (df["entry_score"] >= regime["min_entry"])
        & (df["leadership_score"] >= regime["min_leadership"])
    )

    df.to_csv(f"{args.output_dir}/results_universe_{date_tag}.csv", index=False)

    passers = df[df["regime_eligible"]].copy()
    print(f"\n=== TREND-TEMPLATE PASS RATE: {int(df['all_pass'].sum())}/{len(df)} ===")
    print(f"=== REGIME-ELIGIBLE SETUPS: {len(passers)}/{len(df)} ===")
    if not passers.empty:
        family_counts = {}
        for arch in FAMILY_THRESHOLDS:
            qual_col = f"{arch}_qualifies"
            if qual_col in passers.columns:
                family_counts[arch] = int(passers[qual_col].sum())
        print(f"Eligible setup mix: {family_counts}")

    # Jeff Sun "no biotechs" — drop Healthcare/Biotechnology names using the
    # description cache's sector/industry fields. Unknown tickers pass through;
    # the cache fills organically so more get filtered on each subsequent run.
    if args.exclude_biotech:
        cache_path = os.environ.get("DESC_CACHE", "public/cache/descriptions.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                desc_cache = json.load(f)
            def _is_biotech(t):
                info = desc_cache.get(t, {})
                sector = (info.get("sector") or "").lower()
                industry = (info.get("industry") or "").lower()
                return sector == "healthcare" and "biotech" in industry
            before = len(passers)
            passers = passers[~passers["ticker"].apply(_is_biotech)].copy()
            print(f"  exclude-biotech: removed {before - len(passers)} names", file=sys.stderr)
        else:
            print(f"  exclude-biotech: description cache missing; no filter applied", file=sys.stderr)

    # Optional sector overlay for the local scanner summary. In the workflow,
    # sector_scan.py runs separately and the published shortlist always applies
    # this bonus. For standalone scanner runs we apply it only when a sector
    # dashboard JSON is already available.
    passers["sector_bonus"] = 0.0
    passers["sector_name"] = None
    if HAVE_SECTOR_ALIGN:
        sector_ranks = load_sector_ranks()
        if sector_ranks:
            desc_cache = load_sector_description_cache()
            passers["sector_name"] = passers["ticker"].apply(lambda t: sector_name_for(t, desc_cache))
            passers["sector_bonus"] = passers["ticker"].apply(
                lambda t: sector_bonus_for(t, sector_ranks, desc_cache)
            )
            print("  sector bonus applied to local shortlist summary", file=sys.stderr)
        else:
            print("  sector bonus skipped locally (sectors.json unavailable)", file=sys.stderr)

    # Unified ranking: sort by normalized family strength (excess above each
    # archetype's threshold) rather than raw max(family_scores) — the scores
    # are on different scales across families, so raw max would mechanically
    # favor higher-ceiling archetypes.
    shortlist = passers.copy()
    sort_cols = ["best_family_excess", "entry_score", "leadership_score",
                 "sector_bonus", "qualified_count",
                 "qm_breakout_vendor_score", "qm_episodic_pivot_vendor_score",
                 "rs_pct_rank"]
    shortlist = shortlist.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    cols = ["ticker", "primary_setup", "also_fits", "qualified_count",
            "best_family_excess", "best_family_pct",
            "price", "pct_from_hi", "pct_to_pivot",
            "base_length", "base_depth_pct", "base_tightness_pct",
            "tightness_score", "support_score",
            "leadership_score", "entry_score", "setup_score",
            "sector_bonus",
            "sepa_vcp_score", "power_play_score", "qm_breakout_score", "qm_episodic_pivot_score",
            "qm_breakout_vendor_score", "qm_episodic_pivot_vendor_score",
            "breakout_ready", "quiet_pullback_weeks", "distribution_days",
            "avg_close_in_range", "up_down_vol_ratio", "vol_ratio",
            "rs_vs_spy", "rs_pct_rank"]
    cols = [c for c in cols if c in shortlist.columns]
    print(f"\n=== BREAKOUT SHORTLIST ({regime['label']}, setup_score>={max(args.min_setup_score, regime['min_setup'])}): {len(shortlist)} ===\n")
    print(shortlist[cols].head(50).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Qullamaggie-focused cuts (independent of SEPA score)
    if args.use_qullamaggie and HAVE_QM_VENDOR:
        qm_breakouts = df[
            (df["qm_breakout_qualifies"] == True)
            & (df["entry_score"] >= regime["min_entry"])
        ].copy()
        qm_breakouts = qm_breakouts.sort_values(
            ["qm_breakout_vendor_score", "qm_breakout_score", "leadership_score", "rs_pct_rank"],
            ascending=[False, False, False, False],
        )
        print(f"\n=== QULLAMAGGIE BREAKOUT CANDIDATES: {len(qm_breakouts)} ===")
        breakout_cols = ["ticker", "price", "ret_6m",
                         "qm_breakout_vendor_score", "qm_breakout_score",
                         "qm_breakout_entry_price", "qm_breakout_stop_price", "qm_breakout_stop_loss_pct",
                         "entry_score", "pct_from_hi", "pct_to_pivot"]
        print(qm_breakouts[breakout_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        qm_eps = df[
            (df["qm_episodic_pivot_qualifies"] == True)
            & (df["entry_score"] >= regime["min_entry"])
        ].copy()
        qm_eps = qm_eps.sort_values(
            ["qm_episodic_pivot_vendor_score", "qm_episodic_pivot_score", "leadership_score", "rs_pct_rank"],
            ascending=[False, False, False, False],
        )
        print(f"\n=== QULLAMAGGIE EPISODIC PIVOT CANDIDATES: {len(qm_eps)} ===")
        ep_cols = ["ticker", "price", "ret_6m", "recent_expansion", "post_expansion_tight",
                   "qm_episodic_pivot_vendor_score", "qm_episodic_pivot_score",
                   "qm_episodic_pivot_entry_price", "qm_episodic_pivot_stop_price",
                   "qm_episodic_pivot_stop_loss_pct", "entry_score", "pct_from_hi"]
        print(qm_eps[ep_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min · results at {args.output_dir}/results_universe_{date_tag}.csv ===")

if __name__ == "__main__":
    main()

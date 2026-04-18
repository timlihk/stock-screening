"""
Parameterized universe SEPA scanner.

Pipeline:
  1. Fetch symbol directory from Nasdaq Trader (Nasdaq + other-listed).
  2. Filter by exchange, drop warrants/units/preferreds/rights, optional ETFs.
  3. Pass 1: batch yfinance (1mo), compute 20-day avg dollar volume,
     filter by min_price and min_adv_usd.
  4. Pass 2: batch yfinance (15mo) on survivors, run full SEPA trend template,
     VCP-lite signals, Deepvue/Minervini extensions, and (optional) Qullamaggie
     momentum layer.
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

BENCH = ["SPY", "QQQ", "IWM"]
OUT_DIR_DEFAULT = "/tmp/sepa-scan"

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

def analyze(tkr, df, spy_ret_1y, use_qullamaggie=True):
    if df is None or df.empty:
        return None
    df = df.dropna(how="all")
    close = df["Close"].dropna(); high = df["High"].dropna()
    low = df["Low"].dropna();     vol  = df["Volume"].dropna()
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
    vcp_score = sum([contracting_pbs, contracting_range, vol_dry_up, pb_count >= 3])

    pivot = close.iloc[-16:-1].max() if len(close) >= 20 else np.nan
    pct_to_pivot = (price / pivot - 1) * 100 if not np.isnan(pivot) else np.nan
    above_pivot = pct_to_pivot > 0

    # Deepvue/Minervini extensions
    atr10 = atr(high, low, close, 10); atr50 = atr(high, low, close, 50)
    atr_ratio = (atr10 / atr50) if (not np.isnan(atr10) and atr50 > 0) else np.nan
    atr_compression = not np.isnan(atr_ratio) and atr_ratio <= 0.55
    if len(close) >= 5:
        r5_hi, r5_lo = high.iloc[-5:].max(), low.iloc[-5:].min()
        range5_pct = (r5_hi - r5_lo) / price * 100
    else:
        range5_pct = np.nan
    tight_range = not np.isnan(range5_pct) and range5_pct <= 8
    ret_6m = (price / close.iloc[-126] - 1) * 100 if len(close) >= 126 else np.nan
    ret_15d = (price / close.iloc[-16] - 1) * 100 if len(close) >= 16 else np.nan
    power_play = (not np.isnan(ret_6m) and not np.isnan(ret_15d)
                  and ret_6m > 85 and -15 <= ret_15d <= 5)
    ma20 = close.rolling(20).mean().iloc[-1]
    breakout_confirm = (price > ma20 and not np.isnan(vol_ratio)
                        and vol_ratio >= 1.5 and above_pivot)
    setup_score = vcp_score + sum([atr_compression, tight_range, power_play, breakout_confirm])

    out = dict(
        ticker=tkr, price=price,
        pct_from_hi=pct_from_hi, pct_above_lo=pct_above_lo,
        ret_1y=ret_1y, rs_vs_spy=rs_vs_spy,
        vol_ratio=vol_ratio, all_pass=all_pass, vcp_score=vcp_score,
        atr_ratio=atr_ratio, range5_pct=range5_pct,
        ret_6m=ret_6m, ret_15d=ret_15d,
        contracting_pbs=contracting_pbs, vol_dry_up=vol_dry_up,
        atr_compression=atr_compression, tight_range=tight_range,
        power_play=power_play, breakout_confirm=breakout_confirm,
        pct_to_pivot=pct_to_pivot, setup_score=setup_score,
    )

    if use_qullamaggie:
        # Qullamaggie-style momentum signals
        ret_1m = (price / close.iloc[-22] - 1) * 100 if len(close) >= 22 else np.nan
        ret_3m = (price / close.iloc[-63] - 1) * 100 if len(close) >= 63 else np.nan
        # top-1-2% momentum gate: up ≥25% in 1mo OR ≥50% in 3mo OR ≥100% in 6mo
        qm_momentum = any([(not np.isnan(ret_1m) and ret_1m >= 25),
                           (not np.isnan(ret_3m) and ret_3m >= 50),
                           (not np.isnan(ret_6m) and ret_6m >= 100)])
        # 10/20 EMA ride: price > 10 EMA > 20 EMA and within 7% of 10 EMA
        ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        qm_ema_ride = (price > ema10 > ema20
                       and abs(price - ema10) / price < 0.07)
        # ADR% over last 20 days — Qullamaggie prefers ≥5% for tradeable volatility
        adr_pct = ((high - low) / close).tail(20).mean() * 100
        qm_adr_pct = adr_pct >= 5
        # Tight consolidation proxy (reuse 5-day range tightness)
        qm_consolidation = tight_range
        qm_score = sum([qm_momentum, qm_ema_ride, qm_adr_pct, qm_consolidation])
        out.update(dict(
            ret_1m=ret_1m, ret_3m=ret_3m,
            qm_momentum=qm_momentum, qm_ema_ride=qm_ema_ride,
            adr_pct=adr_pct, qm_adr_pct=qm_adr_pct,
            qm_consolidation=qm_consolidation, qm_score=qm_score,
        ))
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

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exchanges", default="nasdaq,nyse",
                   help="Comma-separated: nasdaq, nyse, amex, arca, bats")
    p.add_argument("--universe-source", default="sec_edgar",
                   choices=["sec_edgar", "nasdaq_trader"])
    p.add_argument("--use-finviz-prefilter", action="store_true",
                   help="Pre-filter universe via Finviz (much faster)")
    p.add_argument("--finviz-above-sma", type=int, default=50, choices=[0, 20, 50, 200])
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-adv-usd", type=float, default=50e6,
                   help="Min 20-day avg dollar volume ($)")
    p.add_argument("--include-etf", action="store_true")
    p.add_argument("--min-setup-score", type=int, default=3)
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
        )
    else:
        universe = fetch_universe(exchanges, include_etf=args.include_etf)
    if args.max_tickers:
        universe = universe[:args.max_tickers]
    print(f"  universe size: {len(universe)}", file=sys.stderr)

    print(f"\n[3/4] Liquidity pass (price >= ${args.min_price:.2f}, ADV >= ${args.min_adv_usd:,.0f})...",
          file=sys.stderr)
    liq = liquidity_pass(universe, args.min_price, args.min_adv_usd)
    liq.to_csv(f"{args.output_dir}/universe_liquid_{date_tag}.csv", index=False)
    print(f"  liquid survivors: {len(liq)}", file=sys.stderr)

    print(f"\n[4/4] Full SEPA scan on {len(liq)} survivors (Qullamaggie={args.use_qullamaggie})...",
          file=sys.stderr)
    df = full_scan(liq["ticker"].tolist(), spy["ret_1y"],
                   use_qullamaggie=args.use_qullamaggie)
    df = df.merge(liq[["ticker", "adv_usd"]], on="ticker", how="left")
    df.to_csv(f"{args.output_dir}/results_universe_{date_tag}.csv", index=False)

    passers = df[df.all_pass].copy()
    print(f"\n=== TREND-TEMPLATE PASS RATE: {len(passers)}/{len(df)} ===")

    shortlist = passers[passers.setup_score >= args.min_setup_score].copy()
    if args.use_qullamaggie and "qm_score" in shortlist.columns:
        shortlist = shortlist.sort_values(
            ["setup_score", "qm_score", "rs_vs_spy"], ascending=[False, False, False])
    else:
        shortlist = shortlist.sort_values(["setup_score", "rs_vs_spy"], ascending=[False, False])

    cols = ["ticker", "price", "pct_from_hi", "pct_to_pivot", "vcp_score",
            "tight_range", "power_play", "breakout_confirm",
            "vol_ratio", "rs_vs_spy", "setup_score"]
    if args.use_qullamaggie:
        cols += ["ret_1m", "ret_3m", "adr_pct", "qm_score"]
    print(f"\n=== COMPOSITE SHORTLIST (setup_score>={args.min_setup_score}): {len(shortlist)} ===\n")
    print(shortlist[cols].head(50).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Qullamaggie-focused cut (independent of SEPA score)
    if args.use_qullamaggie and "qm_score" in df.columns:
        qm_only = df[(df.qm_score >= 3) & (df.price >= args.min_price)].copy()
        qm_only = qm_only.sort_values("qm_score", ascending=False)
        print(f"\n=== QULLAMAGGIE HIGH-SCORE (qm_score>=3): {len(qm_only)} ===")
        qm_cols = ["ticker", "price", "ret_1m", "ret_3m", "ret_6m",
                   "adr_pct", "qm_momentum", "qm_ema_ride", "qm_consolidation",
                   "qm_score", "setup_score", "pct_from_hi"]
        print(qm_only[qm_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min · results at {args.output_dir}/results_universe_{date_tag}.csv ===")

if __name__ == "__main__":
    main()

"""
Sector rotation dashboard.

Ranks sector, industry, and factor ETFs by 1/3/6/12-mo returns and
position vs 50/200 SMA. Output: JSON + HTML heatmap dashboard.

Usage:
    python3 scanner/sector_scan.py --output-dir public/results/latest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

ETFS = {
    # GICS sectors (SPDR)
    "XLE":  ("Energy", "sector"),
    "XLF":  ("Financials", "sector"),
    "XLK":  ("Technology", "sector"),
    "XLV":  ("Health Care", "sector"),
    "XLI":  ("Industrials", "sector"),
    "XLP":  ("Consumer Staples", "sector"),
    "XLY":  ("Consumer Discretionary", "sector"),
    "XLB":  ("Materials", "sector"),
    "XLRE": ("Real Estate", "sector"),
    "XLC":  ("Communication Services", "sector"),
    "XLU":  ("Utilities", "sector"),
    # Industry / theme
    "SMH":  ("Semiconductors", "industry"),
    "XBI":  ("Biotech (EW)", "industry"),
    "IBB":  ("Biotech (cap-wt)", "industry"),
    "GDX":  ("Gold Miners", "industry"),
    "SIL":  ("Silver Miners", "industry"),
    "KRE":  ("Regional Banks", "industry"),
    "KBE":  ("Banks", "industry"),
    "XHB":  ("Homebuilders", "industry"),
    "XRT":  ("Retail", "industry"),
    "XOP":  ("Oil & Gas Explorers", "industry"),
    "ITA":  ("Aerospace & Defense", "industry"),
    "IYT":  ("Transportation", "industry"),
    "XME":  ("Metals & Mining", "industry"),
    # Factor ETFs
    "MTUM": ("Momentum", "factor"),
    "QUAL": ("Quality", "factor"),
    "USMV": ("Low Volatility", "factor"),
    # Benchmarks
    "SPY":  ("S&P 500", "benchmark"),
    "QQQ":  ("Nasdaq-100", "benchmark"),
    "IWM":  ("Russell 2000", "benchmark"),
    "DIA":  ("Dow Jones", "benchmark"),
}


def fetch(symbols, period="15mo"):
    return yf.download(
        list(symbols), period=period, auto_adjust=True,
        progress=False, threads=True, group_by="ticker",
    )


def analyze_etf(tkr, df, spy_returns):
    if df is None or df.empty:
        return None
    close = df["Close"].dropna()
    if len(close) < 252:
        return None
    price = float(close.iloc[-1])
    ret = lambda n: float((price / close.iloc[-n] - 1) * 100) if len(close) >= n else np.nan
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    hi52 = float(close.iloc[-252:].max())
    lo52 = float(close.iloc[-252:].min())
    return {
        "price": price,
        "ret_1w": ret(5),
        "ret_1m": ret(22),
        "ret_3m": ret(63),
        "ret_6m": ret(126),
        "ret_ytd": ret_ytd(close),
        "ret_1y": ret(252),
        "vs_spy_3m": ret(63) - spy_returns.get("ret_3m", 0) if np.isfinite(ret(63)) else np.nan,
        "vs_spy_6m": ret(126) - spy_returns.get("ret_6m", 0) if np.isfinite(ret(126)) else np.nan,
        "vs_spy_1y": ret(252) - spy_returns.get("ret_1y", 0) if np.isfinite(ret(252)) else np.nan,
        "above_50ma": price > ma50,
        "above_200ma": price > ma200,
        "pct_from_52h": float((price / hi52 - 1) * 100),
        "pct_above_52l": float((price / lo52 - 1) * 100),
    }


def ret_ytd(close):
    jan1 = pd.Timestamp(f"{close.index[-1].year}-01-01", tz=close.index.tz)
    mask = close.index >= jan1
    if not mask.any():
        return np.nan
    base = close[mask].iloc[0]
    return float((close.iloc[-1] / base - 1) * 100)


def build_html(rows, date, output_path):
    rows_by_group = {"sector": [], "industry": [], "factor": [], "benchmark": []}
    for r in rows:
        rows_by_group.get(r["group"], []).append(r)

    def color_for(val, lo=-20, hi=20):
        if val is None or not np.isfinite(val):
            return "#1b2029"
        v = max(lo, min(hi, val))
        if v >= 0:
            t = v / hi
            r_, g_, b_ = 46, int(91 + t * 60), int(63 + t * 40)
        else:
            t = abs(v) / abs(lo)
            r_, g_, b_ = int(91 + t * 60), 46, int(61 - t * 30)
        return f"rgb({r_},{g_},{b_})"

    def fmt_cell(val, is_pct=True):
        if val is None or not np.isfinite(val):
            return '<td class="num na">—</td>'
        sign = "+" if val > 0 else ""
        bg = color_for(val)
        return f'<td class="num" style="background:{bg}">{sign}{val:.1f}{"%" if is_pct else ""}</td>'

    def table_for_group(group_name, rows_):
        if not rows_:
            return ""
        rows_sorted = sorted(rows_, key=lambda r: r.get("ret_3m", -999), reverse=True)
        body = []
        for r in rows_sorted:
            body.append(f"""
                <tr>
                    <td><b>{r['ticker']}</b></td>
                    <td class="name">{r['name']}</td>
                    {fmt_cell(r.get('ret_1w'))}
                    {fmt_cell(r.get('ret_1m'))}
                    {fmt_cell(r.get('ret_3m'))}
                    {fmt_cell(r.get('ret_6m'))}
                    {fmt_cell(r.get('ret_ytd'))}
                    {fmt_cell(r.get('ret_1y'))}
                    {fmt_cell(r.get('vs_spy_3m'))}
                    {fmt_cell(r.get('vs_spy_1y'))}
                    <td class="flag">{'✅' if r.get('above_50ma') else '❌'}</td>
                    <td class="flag">{'✅' if r.get('above_200ma') else '❌'}</td>
                    {fmt_cell(r.get('pct_from_52h'))}
                </tr>""")
        return f"""
            <h2>{group_name.title()}</h2>
            <table>
              <thead>
                <tr><th>Ticker</th><th>Name</th><th>1W</th><th>1M</th><th>3M</th>
                    <th>6M</th><th>YTD</th><th>1Y</th><th>vs SPY 3M</th>
                    <th>vs SPY 1Y</th><th>&gt;50MA</th><th>&gt;200MA</th><th>From 52H</th></tr>
              </thead>
              <tbody>{''.join(body)}</tbody>
            </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Sector Rotation — {date}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 22px; background: #0f1115; color: #e6e6e6;
         font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
         line-height: 1.45; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px;
       color: #9aa0a6; margin: 24px 0 10px; }}
  p.sub {{ color: #9aa0a6; font-size: 13px; margin: 0 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px;
           margin-bottom: 14px; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid #262b36; }}
  th {{ background: #1b2029; color: #9aa0a6; font-weight: 500;
        font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
        text-align: left; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.num.na {{ color: #5b5b5b; background: #1b2029; }}
  td.flag {{ text-align: center; }}
  td.name {{ color: #cfd4dc; }}
</style>
</head>
<body>
<h1>Sector Rotation — {date}</h1>
<p class="sub">Ranked by 3-month return within each group. Green = outperforming; red = underperforming. Heatmap column shading shows return magnitude.</p>

{table_for_group('Sectors', rows_by_group['sector'])}
{table_for_group('Industries / Themes', rows_by_group['industry'])}
{table_for_group('Factor ETFs', rows_by_group['factor'])}
{table_for_group('Benchmarks', rows_by_group['benchmark'])}
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="public/results/latest")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching {len(ETFS)} ETFs...", file=sys.stderr)
    data = fetch(list(ETFS.keys()))

    # Pre-compute SPY returns for relative-strength deltas
    spy_df = data["SPY"]
    spy_close = spy_df["Close"].dropna()
    spy_returns = {}
    for (label, n) in [("ret_3m", 63), ("ret_6m", 126), ("ret_1y", 252)]:
        spy_returns[label] = float((spy_close.iloc[-1] / spy_close.iloc[-n] - 1) * 100) if len(spy_close) >= n else 0

    rows = []
    for t, (name, group) in ETFS.items():
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            stats = analyze_etf(t, df, spy_returns)
            if stats is None:
                continue
            stats["ticker"] = t
            stats["name"] = name
            stats["group"] = group
            rows.append(stats)
        except Exception as e:
            print(f"  {t}: {e}", file=sys.stderr)

    # Write JSON (machine-readable)
    with open(os.path.join(args.output_dir, "sectors.json"), "w") as f:
        json.dump({
            "date": args.date,
            "spy_returns": spy_returns,
            "rows": rows,
        }, f, indent=2, default=str)

    # Write HTML dashboard
    build_html(rows, args.date, os.path.join(args.output_dir, "sectors.html"))
    print(f"Wrote sectors.html + sectors.json ({len(rows)} ETFs)", file=sys.stderr)


if __name__ == "__main__":
    main()

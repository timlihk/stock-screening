"""
Generate a one-page SEPA chart review dashboard.

Layout: 2-col grid, chart (TradingView, 1Y daily, SMA 10/20/50/200) on the left,
Yahoo Finance company description on the right. One row per ticker.

Usage:
    python3 build_chart_page.py \
        --date 2026-04-18 \
        --tickers ASC,ACA,PRIM,JOE,GOLF,MCHB,IMVT,BFH,LION,COGT \
        --notes-file /tmp/sepa-scan/notes_20260418.txt

`notes-file` format: one line per ticker, "TICKER|short SEPA note"
If omitted, notes are left blank.

See chart_page_guideline.md for the design spec.
"""
import argparse, json, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from descriptions import get_descriptions
    HAVE_CACHE = True
except ImportError:
    HAVE_CACHE = False

OUT_DIR = "/tmp/sepa-scan"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>SEPA Finalists — __DATE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px;
    background: #0f1115; color: #e6e6e6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    line-height: 1.45;
  }
  h1 { font-size: 18px; margin: 0 0 4px; letter-spacing: .3px; }
  .sub { font-size: 12px; color: #9aa0a6; margin: 0 0 20px; }
  .row {
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
    gap: 14px;
    margin-bottom: 14px;
  }
  .card {
    background: #161a22; border: 1px solid #262b36;
    border-radius: 8px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .card header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; background: #1b2029; border-bottom: 1px solid #262b36;
    font-size: 13px;
  }
  .tag { color: #9aa0a6; font-size: 11px; }
  .chart { height: 520px; width: 100%; }
  .desc { padding: 14px 16px; font-size: 13px; overflow-y: auto; }
  .desc .meta { font-size: 12px; color: #9aa0a6; margin-bottom: 10px; }
  .desc .stats {
    display: flex; flex-wrap: wrap; gap: 6px 10px; margin-bottom: 12px;
  }
  .desc .stats span {
    background: #1b2029; padding: 3px 8px; border-radius: 4px;
    font-size: 11px; color: #c6cbd3; border: 1px solid #262b36;
  }
  .desc .summary { color: #cfd4dc; font-size: 12.5px; white-space: pre-wrap; }
  .desc a { color: #7ab3ff; text-decoration: none; font-size: 11px; }
  .desc a:hover { text-decoration: underline; }
  @media (max-width: 1100px) {
    .row { grid-template-columns: 1fr; }
    .chart { height: 420px; }
  }
</style>
</head>
<body>
<h1>SEPA Finalists — __DATE__</h1>
<p class="sub">1-year daily view · SMA 10 / 20 / 50 / 200 · ranked by buy-zone proximity + VCP quality</p>

<div id="container"></div>

<script>
  const finalists = __FINALISTS__;
  const info = __INFO__;

  function fmtCap(c) {
    if (!c) return "";
    if (c >= 1e12) return "$" + (c / 1e12).toFixed(1) + "T";
    if (c >= 1e9)  return "$" + (c / 1e9).toFixed(2) + "B";
    if (c >= 1e6)  return "$" + (c / 1e6).toFixed(0) + "M";
    return "$" + c;
  }

  const container = document.getElementById("container");

  finalists.forEach(({ t, note }, idx) => {
    const meta = info[t] || {};
    const cap = fmtCap(meta.marketCap);
    const emp = meta.employees ? meta.employees.toLocaleString() + " empl." : "";
    const sector = [meta.sector, meta.industry].filter(Boolean).join(" · ");
    const country = meta.country || "";
    const summary = meta.summary || "(no description available from Yahoo Finance)";
    const website = meta.website
      ? `<a href="${meta.website}" target="_blank" rel="noreferrer">${meta.website.replace(/^https?:\\/\\//, "")}</a>`
      : "";

    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <div class="card">
        <header>
          <div><strong>${t}</strong> <span class="tag">${note ? "· " + note : ""}</span></div>
          <span class="tag">#${idx + 1}</span>
        </header>
        <div class="chart" id="chart-${t}"></div>
      </div>
      <div class="card">
        <header>
          <div><strong>${meta.name || t}</strong></div>
          <span class="tag">${cap}</span>
        </header>
        <div class="desc">
          <div class="meta">${sector}${country ? " · " + country : ""}</div>
          <div class="stats">
            ${cap ? `<span>Mkt Cap ${cap}</span>` : ""}
            ${emp ? `<span>${emp}</span>` : ""}
            ${website ? `<span>${website}</span>` : ""}
          </div>
          <div class="summary">${summary}</div>
        </div>
      </div>
    `;
    container.appendChild(row);

    const w = document.createElement("script");
    w.type = "text/javascript";
    w.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    w.async = true;
    w.innerHTML = JSON.stringify({
      autosize: true,
      symbol: t,
      interval: "D",
      timezone: "America/New_York",
      theme: "dark",
      style: "1",
      locale: "en",
      range: "12M",
      withdateranges: true,
      allow_symbol_change: true,
      hide_side_toolbar: false,
      save_image: false,
      studies: [
        { id: "MASimple@tv-basicstudies", inputs: { length: 10 } },
        { id: "MASimple@tv-basicstudies", inputs: { length: 20 } },
        { id: "MASimple@tv-basicstudies", inputs: { length: 50 } },
        { id: "MASimple@tv-basicstudies", inputs: { length: 200 } }
      ],
      support_host: "https://www.tradingview.com"
    });
    document.getElementById(`chart-${t}`).appendChild(w);
  });
</script>
</body>
</html>
"""

def fetch_info(tickers):
    # Prefer the persistent cache module if available (handles rate limits +
    # inter-run reuse). Fall back to direct yfinance for local-only use.
    if HAVE_CACHE:
        return get_descriptions(tickers)
    info = {}
    for t in tickers:
        try:
            i = yf.Ticker(t).info
            info[t] = {
                "name": i.get("longName") or i.get("shortName") or t,
                "sector": i.get("sector") or "",
                "industry": i.get("industry") or "",
                "country": i.get("country") or "",
                "website": i.get("website") or "",
                "marketCap": i.get("marketCap") or 0,
                "employees": i.get("fullTimeEmployees") or 0,
                "summary": i.get("longBusinessSummary") or "",
            }
        except Exception as e:
            info[t] = {"name": t, "summary": f"(fetch failed: {e})"}
    return info

def load_notes(path):
    if not path or not os.path.exists(path):
        return {}
    notes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            t, n = line.split("|", 1)
            notes[t.strip()] = n.strip()
    return notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD, used in title + output filename")
    ap.add_argument("--tickers", required=True, help="Comma-separated ticker list in display order")
    ap.add_argument("--notes-file", help="Optional path to TICKER|note lines")
    ap.add_argument("--no-open", action="store_true", help="Skip opening in browser")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    notes = load_notes(args.notes_file)

    print(f"Fetching descriptions for {len(tickers)} tickers...", file=sys.stderr)
    info = fetch_info(tickers)

    os.makedirs(OUT_DIR, exist_ok=True)
    date_tag = args.date.replace("-", "")
    desc_path = f"{OUT_DIR}/descriptions_{date_tag}.json"
    with open(desc_path, "w") as f:
        json.dump(info, f, indent=2)

    finalists = [{"t": t, "note": notes.get(t, "")} for t in tickers]
    html = (HTML_TEMPLATE
            .replace("__DATE__", args.date)
            .replace("__FINALISTS__", json.dumps(finalists))
            .replace("__INFO__", json.dumps(info)))

    out_path = f"{OUT_DIR}/charts_{date_tag}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")

    if not args.no_open:
        subprocess.run(["open", out_path])

if __name__ == "__main__":
    main()

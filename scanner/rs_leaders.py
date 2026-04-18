"""
Pure relative-strength leaders — IBD-style.

Reads the main SEPA scan's results CSV and surfaces the top RS names
regardless of VCP / Power Play / pattern criteria. Captures momentum
leaders that never form a clean VCP but still have the biggest 1y
returns.

Inputs:
  results_universe_YYYYMMDD.csv (must have rs_pct_rank or ret_1y + price)

Outputs:
  rs_leaders.txt — TICKER|note format for build_chart_page.py
  rs_leaders.json — machine-readable (rank + all returns)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--output-dir", default="public/results/latest")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--require-above-50ma", action="store_true", default=True,
                    help="Basic sanity: require price > 50MA (drops clearly broken names)")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("results_csv is empty; skipping RS leaders", file=sys.stderr)
        # Still write empty artifacts so the dashboard can render "no data".
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "rs_leaders.txt"), "w") as f:
            f.write("")
        with open(os.path.join(args.output_dir, "rs_leaders.json"), "w") as f:
            json.dump({"rows": []}, f)
        return

    # Sanity filters: strong momentum names should at least be above their 50MA
    # (proxy: price > ma50 in original scan, or use `c5` = price > ma50 column).
    if "c5" in df.columns and args.require_above_50ma:
        df = df[df["c5"] == True]

    # Need rs_pct_rank; fall back to ret_1y if upstream is an older CSV
    rank_col = "rs_pct_rank" if "rs_pct_rank" in df.columns else "ret_1y"
    df = df.dropna(subset=[rank_col])
    df = df.sort_values(rank_col, ascending=False).head(args.top_n)

    os.makedirs(args.output_dir, exist_ok=True)

    # Write TICKER|note shortlist for build_chart_page
    with open(os.path.join(args.output_dir, "rs_leaders.txt"), "w") as f:
        for _, r in df.iterrows():
            note_parts = [f"RSpct {int(r[rank_col])}" if rank_col == "rs_pct_rank" else f"1y {r[rank_col]:+.0f}%"]
            if "setup_score" in r and pd.notna(r["setup_score"]):
                note_parts.append(f"Setup {int(r['setup_score'])}")
            if "qm_score" in r and pd.notna(r["qm_score"]):
                note_parts.append(f"QM {int(r['qm_score'])}")
            if "ret_1y" in r and pd.notna(r["ret_1y"]):
                note_parts.append(f"1y {r['ret_1y']:+.0f}%")
            f.write(f"{r['ticker']}|{' · '.join(note_parts)}\n")

    # Write JSON with the full top-N for any downstream tooling
    keep_cols = [c for c in ["ticker", "price", "rs_pct_rank", "ret_1y", "ret_6m",
                              "ret_3m", "ret_1m", "setup_score", "qm_score",
                              "pct_from_hi", "adv_usd", "all_pass"] if c in df.columns]
    with open(os.path.join(args.output_dir, "rs_leaders.json"), "w") as f:
        json.dump({"rows": df[keep_cols].to_dict(orient="records"),
                   "count": len(df), "rank_col": rank_col}, f, indent=2, default=str)

    print(f"Wrote rs_leaders.txt + rs_leaders.json ({len(df)} names)", file=sys.stderr)


if __name__ == "__main__":
    main()

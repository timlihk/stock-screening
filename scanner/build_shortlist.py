from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from earnings import get_earnings_map
from fundamentals import get_fundamentals_map
from sector_align import (
    load_description_cache,
    load_sector_ranks,
    sector_bonus_for,
    sector_name_for,
)

ARCHETYPES = ["sepa_vcp", "power_play", "qm_breakout", "qm_episodic_pivot"]


def _write_empty_shortlists(results_dir: str) -> None:
    open(f"{results_dir}/shortlist.txt", "w").close()
    for arch in ARCHETYPES:
        open(f"{results_dir}/shortlist_{arch}.txt", "w").close()


def _format_note(row, rank_col: str, arch: str | None = None) -> str:
    headline_arch = arch or row.get("primary_setup", "setup")
    headline_score_col = f"{headline_arch}_score" if headline_arch and headline_arch != "setup" else "setup_score"
    headline_score = row.get(headline_score_col, row.get("setup_score", 0))
    parts = [f"{headline_arch} {headline_score:.0f}"]

    also = [a for a in ARCHETYPES if a != headline_arch and row.get(f"{a}_qualifies", False)]
    if also:
        parts.append("+" + "/".join(also))
    if "entry_score" in row and pd.notna(row["entry_score"]):
        parts.append(f"Entry {int(row['entry_score'])}")
    if "leadership_score" in row and pd.notna(row["leadership_score"]):
        parts.append(f"Lead {int(row['leadership_score'])}")
    if "tightness_score" in row and pd.notna(row["tightness_score"]):
        parts.append(f"Tight {int(row['tightness_score'])}/3")
    if "fundamental_score" in row and pd.notna(row["fundamental_score"]) and int(row["fundamental_score"]) > 0:
        parts.append(f"Fund {int(row['fundamental_score'])}/5")
    if headline_arch == "qm_breakout" and pd.notna(row.get("qm_breakout_vendor_score")) and row["qm_breakout_vendor_score"] > 0:
        parts.append(f"QMraw {row['qm_breakout_vendor_score']:.0f}")
    if headline_arch == "qm_episodic_pivot" and pd.notna(row.get("qm_episodic_pivot_vendor_score")) and row["qm_episodic_pivot_vendor_score"] > 0:
        parts.append(f"QMraw {row['qm_episodic_pivot_vendor_score']:.0f}")
    if rank_col in row and pd.notna(row[rank_col]):
        parts.append(f"{'RSpct' if rank_col == 'rs_pct_rank' else 'RS'} {int(row[rank_col]) if rank_col == 'rs_pct_rank' else f'{row[rank_col]:+.0f}'}")
    if row.get("sector_name") and str(row["sector_name"]) != "nan":
        parts.append(f"{row['sector_name']}")
    if pd.notna(row.get("days_to_earnings")) and row["days_to_earnings"] < 999:
        parts.append(f"Earn {int(row['days_to_earnings'])}d")
    if "rs_vs_spy" in row and pd.notna(row["rs_vs_spy"]) and rank_col != "rs_vs_spy":
        parts.append(f"RS {row.rs_vs_spy:+.0f}")
    return " · ".join(parts)


def _enough_survivors(rows: list[dict]) -> bool:
    if len(rows) < 20:
        return False
    survivors = pd.DataFrame(rows)
    active_archetypes = [arch for arch in ARCHETYPES if f"{arch}_qualifies" in survivors.columns]
    if not active_archetypes:
        return False
    for arch in active_archetypes:
        qual_col = f"{arch}_qualifies"
        if int(survivors[qual_col].sum()) < 10:
            return False
    return True


def build_shortlist(results_dir: str, date: str) -> None:
    date_tag = date.replace("-", "")
    csv_path = f"{results_dir}/results_universe_{date_tag}.csv"
    df = pd.read_csv(csv_path)
    if "regime_eligible" not in df.columns:
        raise RuntimeError("results CSV is missing regime_eligible; rerun the scanner with the updated logic")

    passers = df[df["regime_eligible"] == True].copy()
    if passers.empty:
        print("No regime-eligible passers; writing empty shortlists")
        _write_empty_shortlists(results_dir)
        return

    sector_ranks = load_sector_ranks()
    desc_cache = load_description_cache()
    passers["sector_name"] = passers["ticker"].apply(lambda t: sector_name_for(t, desc_cache))
    passers["sector_bonus"] = passers["ticker"].apply(lambda t: sector_bonus_for(t, sector_ranks, desc_cache))

    fund_map = get_fundamentals_map(passers["ticker"].tolist())

    def _fund(ticker: str, key: str, default=0):
        entry = fund_map.get(ticker) or {}
        value = entry.get(key)
        return value if value is not None else default

    passers["fundamental_score"] = passers["ticker"].apply(lambda t: _fund(t, "fundamental_score"))
    passers["revenue_yoy_growth"] = passers["ticker"].apply(lambda t: _fund(t, "revenue_yoy_growth", None))
    passers["net_income_yoy_growth"] = passers["ticker"].apply(lambda t: _fund(t, "net_income_yoy_growth", None))
    passers["earnings_accelerating"] = passers["ticker"].apply(lambda t: _fund(t, "earnings_accelerating", None))
    passers["revenue_accelerating"] = passers["ticker"].apply(lambda t: _fund(t, "revenue_accelerating", None))

    rank_col = "rs_pct_rank" if "rs_pct_rank" in passers.columns else "rs_vs_spy"
    sort_cols = ["best_family_excess", "fundamental_score", "entry_score",
                 "leadership_score", "sector_bonus", "qualified_count",
                 rank_col]
    passers = passers.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    selected_rows: list[dict] = []
    checked_candidates = 0
    dropped_for_earnings = 0
    batch_size = 40

    for start in range(0, len(passers), batch_size):
        chunk = passers.iloc[start:start + batch_size].copy()
        if chunk.empty:
            break
        chunk_earnings = get_earnings_map(chunk["ticker"].tolist())
        chunk["days_to_earnings"] = chunk["ticker"].map(chunk_earnings.get)
        chunk["near_earnings"] = chunk["days_to_earnings"] <= 10
        checked_candidates += len(chunk)
        dropped_for_earnings += int(chunk["near_earnings"].sum())
        selected_rows.extend(chunk[~chunk["near_earnings"]].to_dict(orient="records"))
        if _enough_survivors(selected_rows):
            break

    survivors = pd.DataFrame(selected_rows)
    if survivors.empty:
        print("All ranked passers failed earnings filter; writing empty shortlists")
        _write_empty_shortlists(results_dir)
        return

    tradable = survivors.head(20)
    with open(f"{results_dir}/shortlist.txt", "w") as handle:
        for _, row in tradable.iterrows():
            handle.write(f"{row['ticker']}|{_format_note(row, rank_col)}\n")

    for arch in ARCHETYPES:
        qual_col = f"{arch}_qualifies"
        score_col = f"{arch}_score"
        if qual_col not in survivors.columns:
            subset = survivors.iloc[0:0].copy()
        else:
            subset = survivors[survivors[qual_col] == True].copy()
        if not subset.empty and score_col in subset.columns:
            subset = subset.sort_values(
                [score_col, "entry_score", "leadership_score", "sector_bonus", rank_col],
                ascending=[False, False, False, False, False],
            )
        subset = subset.head(10)
        with open(f"{results_dir}/shortlist_{arch}.txt", "w") as handle:
            for _, row in subset.iterrows():
                handle.write(f"{row['ticker']}|{_format_note(row, rank_col, arch=arch)}\n")
        print(f"  {arch}: {len(subset)} names")

    print(f"Top 20 shortlist: {tradable.ticker.tolist()}")
    print(f"  checked {checked_candidates} candidates for earnings proximity")
    print(f"  dropped {dropped_for_earnings} candidates for earnings proximity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--results-dir", default="public/results/latest")
    return parser.parse_args()


def main():
    args = parse_args()
    build_shortlist(args.results_dir, args.date)


if __name__ == "__main__":
    main()

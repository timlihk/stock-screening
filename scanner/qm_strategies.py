from __future__ import annotations

from dataclasses import asdict

import pandas as pd

try:
    from vendor_qullamaggie.config import Settings
    from vendor_qullamaggie.data import NasdaqClient, download_daily_history, filter_common_stock_universe
    from vendor_qullamaggie.scanner import _build_metrics, _scan_breakouts, _scan_episodic_pivots
except ImportError:
    from .vendor_qullamaggie.config import Settings
    from .vendor_qullamaggie.data import NasdaqClient, download_daily_history, filter_common_stock_universe
    from .vendor_qullamaggie.scanner import _build_metrics, _scan_breakouts, _scan_episodic_pivots


def _score_to_family_scale(score: float) -> int:
    """Map Andy-Roger's 0-100 score into this scanner's 0-8 family frame."""
    if score <= 0:
        return 0
    return max(0, min(8, int(round(score / 12.5))))


def run_vendor_qm_scan(symbols: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Run Andy-Roger's breakout + episodic-pivot scanner on this repo's liquid
    survivor universe instead of the entire US market.

    This keeps his actual QM logic while avoiding a second full-universe scan.
    """
    if not symbols:
        return {}, {}

    settings = Settings(
        breakout_top_n=max(len(symbols), 50),
        ep_top_n=max(len(symbols), 50),
    )

    nasdaq = NasdaqClient()
    screener_rows = nasdaq.fetch_stock_screener()
    screener_map = {row.symbol: row for row in screener_rows}

    common_rows = filter_common_stock_universe(
        screener_rows,
        min_price=settings.min_price,
        min_market_cap=settings.min_market_cap,
    )
    common_rows = [row for row in common_rows if row.symbol in symbols]
    if not common_rows:
        return {}, {}

    histories = download_daily_history(
        [row.symbol for row in common_rows],
        period=settings.history_period,
        chunk_size=settings.history_chunk_size,
    )
    metrics = _build_metrics(histories)

    breakouts = _scan_breakouts(common_rows, histories, metrics, settings)
    ep_candidates = [screener_map[s] for s in symbols if s in screener_map]
    episodic_pivots = _scan_episodic_pivots(ep_candidates, histories, metrics, settings, nasdaq)

    breakout_map = {}
    for plan in breakouts:
        payload = asdict(plan)
        payload["family_score"] = _score_to_family_scale(plan.score)
        breakout_map[plan.symbol] = payload

    ep_map = {}
    for plan in episodic_pivots:
        payload = asdict(plan)
        payload["family_score"] = _score_to_family_scale(plan.score)
        ep_map[plan.symbol] = payload

    return breakout_map, ep_map


def apply_vendor_qm_scores(df: pd.DataFrame, symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Annotate the scanner dataframe with vendored QM breakout / EP picks.
    Non-picked names get zero scores.
    """
    if df.empty:
        return df

    tickers = symbols or df["ticker"].tolist()
    breakout_map, ep_map = run_vendor_qm_scan(tickers)
    df = df.copy()

    df["qm_breakout_vendor_score"] = df["ticker"].map(lambda t: breakout_map.get(t, {}).get("score", 0.0))
    df["qm_breakout_score"] = df["ticker"].map(lambda t: breakout_map.get(t, {}).get("family_score", 0))
    df["qm_breakout_entry_price"] = df["ticker"].map(lambda t: breakout_map.get(t, {}).get("entry_price"))
    df["qm_breakout_stop_price"] = df["ticker"].map(lambda t: breakout_map.get(t, {}).get("stop_price"))
    df["qm_breakout_stop_loss_pct"] = df["ticker"].map(lambda t: breakout_map.get(t, {}).get("stop_loss_pct"))

    df["qm_episodic_pivot_vendor_score"] = df["ticker"].map(lambda t: ep_map.get(t, {}).get("score", 0.0))
    df["qm_episodic_pivot_score"] = df["ticker"].map(lambda t: ep_map.get(t, {}).get("family_score", 0))
    df["qm_episodic_pivot_entry_price"] = df["ticker"].map(lambda t: ep_map.get(t, {}).get("entry_price"))
    df["qm_episodic_pivot_stop_price"] = df["ticker"].map(lambda t: ep_map.get(t, {}).get("stop_price"))
    df["qm_episodic_pivot_stop_loss_pct"] = df["ticker"].map(lambda t: ep_map.get(t, {}).get("stop_loss_pct"))

    return df

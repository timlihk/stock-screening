from __future__ import annotations

import pandas as pd

from .config import Settings
from .data import NasdaqClient, download_daily_history, filter_common_stock_universe
from .indicators import adr_pct, avg_dollar_volume, percentile_rank, return_pct, sma
from .models import ScreenerRow, TradePlan
from .risk import position_plan


def build_scan(settings: Settings) -> tuple[list[TradePlan], list[TradePlan]]:
    nasdaq = NasdaqClient()
    screener_rows = nasdaq.fetch_stock_screener()
    universe = filter_common_stock_universe(
        screener_rows,
        min_price=settings.min_price,
        min_market_cap=settings.min_market_cap,
    )
    histories = download_daily_history(
        [row.symbol for row in universe],
        period=settings.history_period,
        chunk_size=settings.history_chunk_size,
    )
    metrics = _build_metrics(histories)
    breakout_picks = _scan_breakouts(universe, histories, metrics, settings)
    episodic_pivot_picks = _scan_episodic_pivots(screener_rows, histories, metrics, settings, nasdaq)
    return breakout_picks[: settings.breakout_top_n], episodic_pivot_picks[: settings.ep_top_n]


def _build_metrics(histories: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    rows = []
    for symbol, df in histories.items():
        if len(df) < 130:
            continue
        close = df["close"]
        rows.append(
            {
                "symbol": symbol,
                "ret_21": return_pct(close, 21),
                "ret_63": return_pct(close, 63),
                "ret_126": return_pct(close, 126),
            }
        )
    if not rows:
        return {}
    metric_frame = pd.DataFrame(rows).set_index("symbol")
    if metric_frame.empty:
        return {}
    metric_frame["rank_21"] = percentile_rank(metric_frame["ret_21"])
    metric_frame["rank_63"] = percentile_rank(metric_frame["ret_63"])
    metric_frame["rank_126"] = percentile_rank(metric_frame["ret_126"])
    return metric_frame.to_dict(orient="index")


def _scan_breakouts(
    universe: list[ScreenerRow],
    histories: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, float]],
    settings: Settings,
) -> list[TradePlan]:
    row_map = {row.symbol: row for row in universe}
    plans: list[TradePlan] = []
    for symbol, stat in metrics.items():
        if max(stat["rank_21"], stat["rank_63"], stat["rank_126"]) < settings.breakout_leader_percentile_min:
            continue
        df = histories.get(symbol)
        row = row_map.get(symbol)
        if df is None or row is None or len(df) < 130:
            continue
        plan = _build_breakout_plan(symbol, row, df, stat, settings)
        if plan is not None:
            plans.append(plan)
    return sorted(plans, key=lambda item: item.score, reverse=True)


def _build_breakout_plan(
    symbol: str,
    row: ScreenerRow,
    df: pd.DataFrame,
    stat: dict[str, float],
    settings: Settings,
) -> TradePlan | None:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    adr20 = adr_pct(df)
    avg_dollar_vol20 = avg_dollar_volume(df)
    if avg_dollar_vol20 < settings.min_avg_dollar_volume:
        return None

    sma10 = sma(close, 10)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    if not (close.iloc[-1] >= sma10 >= sma20 >= sma50):
        return None

    recent = df.iloc[-63:]
    impulse_high_offset = int(recent["high"].values.argmax())
    impulse_high_price = float(recent["high"].iloc[impulse_high_offset])
    impulse_low_price = float(recent["low"].iloc[: impulse_high_offset + 1].min())
    impulse_gain_pct = ((impulse_high_price / impulse_low_price) - 1.0) * 100.0
    bars_since_impulse = len(recent) - 1 - impulse_high_offset
    if impulse_gain_pct < settings.breakout_impulse_gain_min:
        return None
    if not (settings.breakout_consolidation_days_min <= bars_since_impulse <= settings.breakout_consolidation_days_max):
        return None

    consolidation = recent.iloc[impulse_high_offset:]
    pivot_price = float(consolidation["high"].max())
    base_low = float(consolidation["low"].min())
    current_close = float(close.iloc[-1])
    distance_to_pivot_pct = ((pivot_price - current_close) / pivot_price) * 100.0
    if distance_to_pivot_pct > settings.breakout_max_distance_to_pivot_pct:
        return None

    consolidation_range_pct = ((float(consolidation["high"].max()) - base_low) / current_close) * 100.0
    max_allowed_range = max(adr20 * 3.0, 12.0)
    if consolidation_range_pct > max_allowed_range:
        return None

    split = max(len(consolidation) // 2, 1)
    early_low = float(consolidation["low"].iloc[:split].min())
    late_low = float(consolidation["low"].iloc[split:].min())
    if late_low < (early_low * 0.97):
        return None

    vol_recent = float(volume.iloc[-10:].mean())
    vol_prior = float(volume.iloc[-30:-10].mean()) if len(volume) >= 30 else vol_recent
    volume_contraction = vol_recent / vol_prior if vol_prior else 1.0

    entry_price = round(pivot_price * 1.001, 2)
    stop_loss_pct = ((entry_price - base_low) / entry_price) * 100.0
    if stop_loss_pct <= 0 or stop_loss_pct > adr20:
        return None
    stop_price = round(entry_price * (1.0 - (stop_loss_pct / 100.0)), 2)

    sizing = position_plan(settings, entry_price, stop_loss_pct)
    if sizing.share_count <= 0:
        return None

    rs_score = (stat["rank_21"] + stat["rank_63"] + stat["rank_126"]) / 3.0
    pivot_score = max(0.0, 100.0 - (distance_to_pivot_pct * 15.0))
    tightness_score = max(0.0, 100.0 - (consolidation_range_pct * 5.0))
    volume_score = max(0.0, 100.0 - (max(volume_contraction, 0.5) * 25.0))
    score = round((rs_score * 0.45) + (pivot_score * 0.25) + (tightness_score * 0.2) + (volume_score * 0.1), 2)

    return TradePlan(
        symbol=symbol,
        name=row.name,
        setup="Breakout",
        score=score,
        entry_price=entry_price,
        stop_price=stop_price,
        stop_loss_pct=round(stop_loss_pct, 2),
        buy_cash_pct=sizing.buy_cash_pct,
        buy_cash_amount=sizing.buy_cash_amount,
        share_count=sizing.share_count,
        first_scale_out_gain_pct=sizing.first_scale_out_gain_pct,
        trailing_exit_rule="Sell 40% at 2R, then trail the rest on the first close below the 10-day moving average.",
        notes=[
            "Daily-chart breakout candidate only. Confirm opening-range strength after the bell.",
            "Wide-stop names are rejected if the planned stop exceeds 1x ADR20.",
        ],
        metrics={
            "adr20_pct": round(adr20, 2),
            "avg_dollar_vol20": round(avg_dollar_vol20, 2),
            "impulse_gain_pct": round(impulse_gain_pct, 2),
            "distance_to_pivot_pct": round(distance_to_pivot_pct, 2),
            "rank_21": round(stat["rank_21"], 2),
            "rank_63": round(stat["rank_63"], 2),
            "rank_126": round(stat["rank_126"], 2),
        },
    )


def _scan_episodic_pivots(
    screener_rows: list[ScreenerRow],
    histories: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, float]],
    settings: Settings,
    nasdaq: NasdaqClient,
) -> list[TradePlan]:
    candidates = [
        row
        for row in screener_rows
        if row.pct_change >= settings.ep_gap_pct_min
        and row.last_sale >= settings.min_price
        and row.market_cap >= settings.min_market_cap
    ]
    candidates = sorted(candidates, key=lambda row: row.pct_change, reverse=True)[: max(settings.ep_top_n * 5, 25)]
    plans: list[TradePlan] = []
    for row in candidates:
        df = histories.get(row.symbol)
        stat = metrics.get(row.symbol)
        if df is None or stat is None or len(df) < 130:
            continue
        plan = _build_ep_plan(row, df, stat, settings, nasdaq)
        if plan is not None:
            plans.append(plan)
    return sorted(plans, key=lambda item: item.score, reverse=True)


def _build_ep_plan(
    row: ScreenerRow,
    df: pd.DataFrame,
    stat: dict[str, float],
    settings: Settings,
    nasdaq: NasdaqClient,
) -> TradePlan | None:
    del stat
    close = df["close"]
    adr20 = adr_pct(df)
    avg_dollar_vol20 = avg_dollar_volume(df)
    if avg_dollar_vol20 < settings.min_avg_dollar_volume:
        return None

    prior_3m_run_pct = return_pct(close, 63)
    if prior_3m_run_pct > settings.ep_max_prior_3m_run_pct:
        return None

    snapshot = nasdaq.fetch_extended_trading(row.symbol)
    avg_volume20 = float(df["volume"].rolling(20).mean().iloc[-1])
    if avg_volume20 <= 0:
        return None
    premarket_volume_ratio = snapshot.volume / avg_volume20
    if premarket_volume_ratio < settings.ep_premarket_volume_ratio_min:
        return None

    entry_price = round(max(snapshot.high_price, snapshot.consolidated_price) * 1.001, 2)
    if snapshot.low_price <= 0 or snapshot.low_price >= entry_price:
        return None
    stop_loss_pct = ((entry_price - snapshot.low_price) / entry_price) * 100.0
    if stop_loss_pct <= 0 or stop_loss_pct > (adr20 * 1.5):
        return None
    stop_price = round(snapshot.low_price, 2)
    sizing = position_plan(settings, entry_price, stop_loss_pct)
    if sizing.share_count <= 0:
        return None

    gap_score = min(row.pct_change * 4.0, 100.0)
    volume_score = min(premarket_volume_ratio * 200.0, 100.0)
    freshness_score = max(0.0, 100.0 - max(prior_3m_run_pct, 0.0) * 2.5)
    score = round((gap_score * 0.45) + (volume_score * 0.35) + (freshness_score * 0.2), 2)

    return TradePlan(
        symbol=row.symbol,
        name=row.name,
        setup="Episodic Pivot",
        score=score,
        entry_price=entry_price,
        stop_price=stop_price,
        stop_loss_pct=round(stop_loss_pct, 2),
        buy_cash_pct=sizing.buy_cash_pct,
        buy_cash_amount=sizing.buy_cash_amount,
        share_count=sizing.share_count,
        first_scale_out_gain_pct=sizing.first_scale_out_gain_pct,
        trailing_exit_rule="Sell 40% at 2R, then trail the rest on the first close below the 20-day moving average.",
        notes=[
            "Premarket EP candidate only. Validate the catalyst manually before entry.",
            "The published setup still needs an opening-range-high confirmation after the open.",
        ],
        metrics={
            "gap_pct": round(row.pct_change, 2),
            "adr20_pct": round(adr20, 2),
            "avg_dollar_vol20": round(avg_dollar_vol20, 2),
            "premarket_volume_ratio": round(premarket_volume_ratio, 2),
            "prior_3m_run_pct": round(prior_3m_run_pct, 2),
        },
    )

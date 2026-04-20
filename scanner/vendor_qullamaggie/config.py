from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    available_cash: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 15.0
    hard_position_cap_pct: float = 30.0
    min_price: float = 10.0
    min_market_cap: float = 300_000_000.0
    min_avg_dollar_volume: float = 15_000_000.0
    top_n: int = 10
    breakout_top_n: int = 6
    ep_top_n: int = 4
    breakout_leader_percentile_min: float = 98.0
    breakout_impulse_gain_min: float = 30.0
    breakout_consolidation_days_min: int = 7
    breakout_consolidation_days_max: int = 40
    breakout_max_distance_to_pivot_pct: float = 5.0
    ep_gap_pct_min: float = 10.0
    ep_premarket_volume_ratio_min: float = 0.2
    ep_max_prior_3m_run_pct: float = 25.0
    history_period: str = "9mo"
    history_chunk_size: int = 100

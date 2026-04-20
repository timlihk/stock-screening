from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScreenerRow:
    symbol: str
    name: str
    last_sale: float
    pct_change: float
    volume: int
    market_cap: float
    sector: str
    industry: str


@dataclass(frozen=True)
class ExtendedTradingSnapshot:
    symbol: str
    consolidated_price: float
    delta_pct: float
    volume: int
    high_price: float
    low_price: float


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    name: str
    setup: str
    score: float
    entry_price: float
    stop_price: float
    stop_loss_pct: float
    buy_cash_pct: float
    buy_cash_amount: float
    share_count: int
    first_scale_out_gain_pct: float
    trailing_exit_rule: str
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

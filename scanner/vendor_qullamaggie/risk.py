from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .utils import safe_floor_shares


@dataclass(frozen=True)
class PositionPlan:
    buy_cash_pct: float
    buy_cash_amount: float
    share_count: int
    first_scale_out_gain_pct: float


def position_plan(settings: Settings, entry_price: float, stop_loss_pct: float) -> PositionPlan:
    raw_position_pct = 0.0
    if stop_loss_pct > 0:
        raw_position_pct = (settings.risk_per_trade_pct / stop_loss_pct) * 100.0
    buy_cash_pct = min(settings.max_position_pct, settings.hard_position_cap_pct, raw_position_pct)
    buy_cash_amount = settings.available_cash * (buy_cash_pct / 100.0)
    share_count = safe_floor_shares(buy_cash_amount, entry_price)
    first_scale_out_gain_pct = stop_loss_pct * 2.0
    return PositionPlan(
        buy_cash_pct=round(buy_cash_pct, 2),
        buy_cash_amount=round(buy_cash_amount, 2),
        share_count=share_count,
        first_scale_out_gain_pct=round(first_scale_out_gain_pct, 2),
    )

from __future__ import annotations

import pandas as pd


def adr_pct(df: pd.DataFrame, length: int = 20) -> float:
    series = ((df["high"] / df["low"]) - 1.0) * 100.0
    return float(series.rolling(length).mean().iloc[-1])


def avg_dollar_volume(df: pd.DataFrame, length: int = 20) -> float:
    series = df["close"] * df["volume"]
    return float(series.rolling(length).mean().iloc[-1])


def return_pct(series: pd.Series, bars: int) -> float:
    if len(series) <= bars:
        return 0.0
    return float(((series.iloc[-1] / series.iloc[-(bars + 1)]) - 1.0) * 100.0)


def sma(series: pd.Series, length: int) -> float:
    return float(series.rolling(length).mean().iloc[-1])


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0

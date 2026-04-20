from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

from .models import ExtendedTradingSnapshot, ScreenerRow
from .utils import extract_parenthetical_percent, parse_int, parse_money, parse_percent, yahoo_symbol


class NasdaqClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "user-agent": "Mozilla/5.0",
                "accept": "application/json, text/plain, */*",
                "origin": "https://www.nasdaq.com",
                "referer": "https://www.nasdaq.com/market-activity/stocks/screener",
            }
        )

    def fetch_stock_screener(self) -> list[ScreenerRow]:
        response = self.session.get(
            "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true",
            timeout=30,
        )
        response.raise_for_status()
        rows = response.json()["data"]["rows"]
        result = []
        for row in rows:
            result.append(
                ScreenerRow(
                    symbol=row["symbol"],
                    name=row["name"],
                    last_sale=parse_money(row["lastsale"]),
                    pct_change=parse_percent(row["pctchange"]),
                    volume=parse_int(row["volume"]),
                    market_cap=parse_money(row["marketCap"]),
                    sector=row.get("sector", "") or "",
                    industry=row.get("industry", "") or "",
                )
            )
        return result

    def fetch_extended_trading(self, symbol: str, assetclass: str = "stocks") -> ExtendedTradingSnapshot:
        response = self.session.get(
            f"https://api.nasdaq.com/api/quote/{symbol}/extended-trading?assetclass={assetclass}&markettype=pre",
            timeout=30,
            headers={"referer": f"https://www.nasdaq.com/market-activity/{assetclass}/{symbol.lower()}"},
        )
        response.raise_for_status()
        payload = response.json()["data"]
        info_row = payload["infoTable"]["rows"][0]
        return ExtendedTradingSnapshot(
            symbol=symbol,
            consolidated_price=parse_money(info_row["consolidated"]),
            delta_pct=extract_parenthetical_percent(info_row["consolidated"]),
            volume=parse_int(info_row["volume"]),
            high_price=parse_money(info_row["highPrice"]),
            low_price=parse_money(info_row["lowPrice"]),
        )


def filter_common_stock_universe(rows: Iterable[ScreenerRow], min_price: float, min_market_cap: float) -> list[ScreenerRow]:
    banned_name_tokens = (
        "warrant",
        "rights",
        "unit",
        "depositary",
        "acquisition",
        "beneficial interest",
        "partnership",
        "trust",
        "etf",
        "fund",
    )
    allowed: list[ScreenerRow] = []
    for row in rows:
        lower_name = row.name.lower()
        if row.last_sale < min_price:
            continue
        if row.market_cap < min_market_cap:
            continue
        if any(token in lower_name for token in banned_name_tokens):
            continue
        if "^" in row.symbol:
            continue
        allowed.append(row)
    return allowed


def download_daily_history(symbols: Iterable[str], period: str, chunk_size: int) -> dict[str, pd.DataFrame]:
    original_symbols = list(symbols)
    yahoo_map = {yahoo_symbol(symbol): symbol for symbol in original_symbols}
    histories: dict[str, pd.DataFrame] = {}
    yahoo_symbols = list(yahoo_map.keys())

    for start in range(0, len(yahoo_symbols), chunk_size):
        chunk = yahoo_symbols[start : start + chunk_size]
        raw = yf.download(
            chunk,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            for yahoo_name in chunk:
                if yahoo_name not in raw.columns.get_level_values(0):
                    continue
                symbol = yahoo_map[yahoo_name]
                frame = raw[yahoo_name].dropna(how="all").copy()
                if frame.empty:
                    continue
                histories[symbol] = frame.rename(columns=str.lower)
        else:
            symbol = yahoo_map[chunk[0]]
            histories[symbol] = raw.dropna(how="all").rename(columns=str.lower)

    return histories

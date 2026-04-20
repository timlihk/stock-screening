from __future__ import annotations

import math
import re


def parse_money(value: str | None) -> float:
    if not value:
        return 0.0
    cleaned = value.replace("$", "").replace(",", "").strip()
    cleaned = cleaned.split(" ")[0]
    cleaned = cleaned.replace("+", "")
    if cleaned in {"", "N/A"}:
        return 0.0
    return float(cleaned)


def parse_percent(value: str | None) -> float:
    if not value:
        return 0.0
    cleaned = value.replace("%", "").replace(",", "").strip()
    if cleaned in {"", "N/A"}:
        return 0.0
    return float(cleaned)


def parse_int(value: str | None) -> int:
    if not value:
        return 0
    cleaned = value.replace(",", "").strip()
    if cleaned in {"", "N/A"}:
        return 0
    return int(float(cleaned))


def extract_parenthetical_percent(text: str | None) -> float:
    if not text:
        return 0.0
    match = re.search(r"\(([-+0-9.]+)%\)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def safe_floor_shares(position_dollars: float, entry_price: float) -> int:
    if entry_price <= 0:
        return 0
    return max(math.floor(position_dollars / entry_price), 0)


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-").replace("/", "-")

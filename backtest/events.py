from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

import pandas as pd


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass
class MarketEvent:
    symbol: str
    bar: pd.Series
    prev_close: float = 0.0


@dataclass
class SignalEvent:
    symbol: str
    direction: int  # 1=BUY, -1=SELL, 0=HOLD
    strength: float
    price: float


@dataclass
class OrderEvent:
    symbol: str
    direction: int
    quantity: int
    order_type: str = "MARKET"
    limit_price: float = 0.0


@dataclass
class FillEvent:
    symbol: str
    direction: int
    quantity: int
    fill_price: float
    commission: float
    timestamp: datetime = None

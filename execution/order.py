from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    symbol: str
    direction: int  # 1=BUY, -1=SELL
    quantity: int  # shares (multiple of 100)
    order_type: str = "MARKET"
    limit_price: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OrderResult:
    success: bool
    fill_price: float = 0.0
    fill_quantity: int = 0
    commission: float = 0.0
    message: str = ""

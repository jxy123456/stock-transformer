from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Set, Tuple

from backtest.events import FillEvent


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    buy_date: date = None


class Portfolio:
    def __init__(self, initial_cash: float = 1_000_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.todays_buys: Set[str] = set()
        self.equity_curve: List[Tuple[date, float]] = []
        self.trade_history: List[dict] = []

    @property
    def total_value(self) -> float:
        return self.cash + sum(
            pos.quantity * pos.avg_cost for pos in self.positions.values()
        )

    def get_position_value(self, symbol: str, current_price: float) -> float:
        if symbol in self.positions:
            return self.positions[symbol].quantity * current_price
        return 0.0

    def on_fill(self, fill: FillEvent, trade_date: date = None):
        trade_value = fill.fill_price * fill.quantity

        if fill.direction == 1:  # BUY
            total_cost = trade_value + fill.commission
            if total_cost > self.cash:
                # Adjust quantity to fit available cash
                max_qty = int(self.cash / (fill.fill_price * (1 + 0.001))) // 100 * 100
                if max_qty < 100:
                    return
                fill = FillEvent(
                    symbol=fill.symbol,
                    direction=fill.direction,
                    quantity=max_qty,
                    fill_price=fill.fill_price,
                    commission=max(max_qty * fill.fill_price * 0.0003, 5.0),
                )
                trade_value = fill.fill_price * fill.quantity
                total_cost = trade_value + fill.commission

            self.cash -= total_cost
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                avg_cost=fill.fill_price,
                buy_date=trade_date,
            )
            self.todays_buys.add(fill.symbol)

        elif fill.direction == -1:  # SELL
            self.cash += trade_value - fill.commission
            if fill.symbol in self.positions:
                del self.positions[fill.symbol]

        self.trade_history.append(
            {
                "date": trade_date,
                "symbol": fill.symbol,
                "direction": "BUY" if fill.direction == 1 else "SELL",
                "quantity": fill.quantity,
                "price": fill.fill_price,
                "commission": fill.commission,
            }
        )

    def on_market_close(self, trade_date: date, prices: Dict[str, float]):
        self.todays_buys.clear()
        total = self.cash + sum(
            self.positions[s].quantity * prices.get(s, 0)
            for s in self.positions
        )
        self.equity_curve.append((trade_date, total))

    def get_holding_pnl(self, symbol: str, current_price: float) -> float:
        if symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        return (current_price - pos.avg_cost) / pos.avg_cost

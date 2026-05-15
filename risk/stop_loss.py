from typing import Dict, List

from backtest.events import SignalEvent
from backtest.portfolio import Portfolio
from strategy.signal_generator import Signal


class StopLossTakeProfit:
    def __init__(self, config: dict = None):
        cfg = config or {}
        risk = cfg.get("risk", {})
        self.stop_loss_pct = risk.get("stop_loss_pct", 0.08)
        self.take_profit_pct = risk.get("take_profit_pct", 0.15)
        self.trailing_stop_pct = risk.get("trailing_stop_pct", 0.05)
        self.highest_price: Dict[str, float] = {}

    def check(
        self, portfolio: Portfolio, current_prices: Dict[str, float]
    ) -> List[SignalEvent]:
        forced_sells = []

        for symbol, position in portfolio.positions.items():
            current_price = current_prices.get(symbol, 0)
            if current_price <= 0:
                continue

            pnl_pct = (current_price - position.avg_cost) / position.avg_cost

            # Fixed stop-loss
            if pnl_pct <= -self.stop_loss_pct:
                forced_sells.append(
                    SignalEvent(symbol=symbol, direction=-1, strength=1.0, price=current_price)
                )
                continue

            # Take profit
            if pnl_pct >= self.take_profit_pct:
                forced_sells.append(
                    SignalEvent(symbol=symbol, direction=-1, strength=1.0, price=current_price)
                )
                continue

            # Trailing stop
            self.highest_price[symbol] = max(
                self.highest_price.get(symbol, current_price), current_price
            )
            if current_price < self.highest_price[symbol] * (1 - self.trailing_stop_pct):
                forced_sells.append(
                    SignalEvent(symbol=symbol, direction=-1, strength=1.0, price=current_price)
                )

        return forced_sells

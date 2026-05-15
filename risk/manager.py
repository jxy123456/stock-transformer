from typing import Dict, List

from backtest.events import SignalEvent
from backtest.portfolio import Portfolio
from risk.drawdown_control import DrawdownControl
from risk.position_limit import PositionLimit
from risk.stop_loss import StopLossTakeProfit
from strategy.signal_generator import Signal


class RiskManager:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.position_limit = PositionLimit(self.config)
        self.drawdown_control = DrawdownControl(self.config)
        self.stop_loss = StopLossTakeProfit(self.config)

    def filter(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        current_prices: Dict[str, float],
    ) -> List[Signal]:
        approved = []

        # Check stop-loss on existing positions first
        forced_sells = self.stop_loss.check(portfolio, current_prices)
        for fs in forced_sells:
            approved.append(Signal(fs.symbol, fs.direction, fs.strength, fs.price))

        # Filter new signals
        for signal in signals:
            # Skip new buys if circuit breaker active
            if signal.direction == 1 and self.drawdown_control.is_circuit_break_active():
                continue

            # Check position limit
            if not self.position_limit.allow(signal, portfolio):
                continue

            # Apply drawdown position reduction
            if signal.direction == 1:
                reduction = self.drawdown_control.get_position_reduction_factor()
                if reduction < 0.1:
                    continue
                signal = Signal(
                    signal.symbol,
                    signal.direction,
                    signal.strength * reduction,
                    signal.price,
                )

            approved.append(signal)

        return approved

    def on_market_close(
        self, portfolio: Portfolio, current_prices: Dict[str, float]
    ):
        self.drawdown_control.update(portfolio)

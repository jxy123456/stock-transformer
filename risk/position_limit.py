from backtest.portfolio import Portfolio
from strategy.signal_generator import Signal


class PositionLimit:
    def __init__(self, config: dict = None):
        cfg = config or {}
        risk = cfg.get("risk", {})
        self.max_positions = risk.get("max_positions", 5)
        self.max_position_pct = risk.get("max_position_pct", 0.25)

    def allow(self, signal: Signal, portfolio: Portfolio) -> bool:
        if signal.direction == 1:  # BUY
            if len(portfolio.positions) >= self.max_positions:
                return False
            position_value = signal.price * 100  # minimum 100 shares
            if portfolio.total_value > 0:
                pct = position_value / portfolio.total_value
                if pct > self.max_position_pct:
                    return False
        return True

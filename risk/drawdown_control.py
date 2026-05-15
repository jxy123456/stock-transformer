from backtest.portfolio import Portfolio


class DrawdownControl:
    def __init__(self, config: dict = None):
        cfg = config or {}
        risk = cfg.get("risk", {})
        self.max_drawdown_pct = risk.get("max_drawdown_pct", 0.15)
        self.reduce_at_pct = risk.get("reduce_at_pct", 0.10)
        self.peak_value = 0.0
        self.current_drawdown = 0.0

    def update(self, portfolio: Portfolio):
        current_value = portfolio.total_value
        self.peak_value = max(self.peak_value, current_value)
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

    def is_circuit_break_active(self) -> bool:
        return self.current_drawdown >= self.max_drawdown_pct

    def get_position_reduction_factor(self) -> float:
        if self.current_drawdown >= self.reduce_at_pct:
            factor = 1.0 - (
                (self.current_drawdown - self.reduce_at_pct)
                / (self.max_drawdown_pct - self.reduce_at_pct)
            )
            return max(factor, 0.0)
        return 1.0

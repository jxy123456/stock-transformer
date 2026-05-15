from strategy.signal_generator import Signal


class PositionSizer:
    def __init__(self, config: dict = None):
        cfg = config or {}
        strategy = cfg.get("strategy", {})
        risk = cfg.get("risk", {})
        self.method = strategy.get("sizing_method", "vol_adjusted")
        self.max_position_pct = risk.get("max_position_pct", 0.25)

    def size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        volatility: float = 0.02,
    ) -> int:
        if self.method == "fixed_fraction":
            target_value = portfolio_value * self.max_position_pct * signal.strength

        elif self.method == "vol_adjusted":
            vol = max(volatility, 0.005)
            vol_target = 0.02  # target 2% volatility contribution
            fraction = min(vol_target / vol, self.max_position_pct)
            target_value = portfolio_value * fraction * signal.strength

        elif self.method == "kelly":
            fraction = min(self.max_position_pct * signal.strength, self.max_position_pct)
            target_value = portfolio_value * fraction
        else:
            target_value = portfolio_value * self.max_position_pct * signal.strength

        shares = int(target_value / current_price)
        shares = (shares // 100) * 100  # round to lot of 100
        return max(shares, 100)

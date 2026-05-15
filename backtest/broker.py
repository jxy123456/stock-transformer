from backtest.events import FillEvent, OrderEvent
from backtest.portfolio import Portfolio


class ASHareBroker:
    def __init__(self, config: dict = None):
        cfg = config or {}
        bt = cfg.get("backtest", {})
        self.commission_rate = bt.get("commission_rate", 0.0003)
        self.stamp_tax_rate = bt.get("stamp_tax_rate", 0.001)
        self.min_commission = bt.get("min_commission", 5.0)
        self.slippage_bps = bt.get("slippage_bps", 5)

    def execute_order(
        self,
        order: OrderEvent,
        open_price: float,
        prev_close: float,
        portfolio: Portfolio,
    ) -> FillEvent:
        # T+1 check
        if order.direction == -1 and order.symbol in portfolio.todays_buys:
            return None

        # Price limit check
        limit_factor = self._get_limit_factor(order.symbol)
        limit_up = prev_close * (1 + limit_factor)
        limit_down = prev_close * (1 - limit_factor)

        # Apply slippage
        if order.direction == 1:
            fill_price = open_price * (1 + self.slippage_bps / 10000)
            if fill_price > limit_up:
                return None  # cannot buy at limit up
        else:
            fill_price = open_price * (1 - self.slippage_bps / 10000)
            if fill_price < limit_down:
                return None  # cannot sell at limit down

        # Round quantity to lot of 100
        quantity = (order.quantity // 100) * 100
        if quantity < 100:
            return None

        # Calculate commission
        trade_value = fill_price * quantity
        commission = max(trade_value * self.commission_rate, self.min_commission)
        if order.direction == -1:
            commission += trade_value * self.stamp_tax_rate  # stamp tax on sell only

        return FillEvent(
            symbol=order.symbol,
            direction=order.direction,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
        )

    @staticmethod
    def _get_limit_factor(symbol: str) -> float:
        # ChiNext (300xxx) or STAR Market (688xxx): 20% limit
        if symbol.startswith("300") or symbol.startswith("688"):
            return 0.20
        # ST stocks: 5% limit (simplified check)
        # Main board: 10% limit
        return 0.10

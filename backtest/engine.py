from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from backtest.broker import ASHareBroker
from backtest.events import FillEvent, OrderEvent, SignalEvent
from backtest.portfolio import Portfolio
from backtest.statistics import BacktestStatistics
from risk.manager import RiskManager
from strategy.position_sizer import PositionSizer
from strategy.signal_generator import SignalGenerator


class BacktestEngine:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        signal_generator: SignalGenerator,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        broker: ASHareBroker,
        portfolio: Portfolio,
        model=None,
        feature_engine=None,
        preprocessor=None,
        config: dict = None,
    ):
        self.data = data
        self.signal_generator = signal_generator
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.broker = broker
        self.portfolio = portfolio
        self.model = model
        self.feature_engine = feature_engine
        self.preprocessor = preprocessor
        self.config = config or {}

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> dict:
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()

        # Get all trading dates
        all_dates = set()
        for symbol in symbols:
            if symbol in self.data and not self.data[symbol].empty:
                df = self.data[symbol]
                if "datetime" in df.columns:
                    dates = pd.to_datetime(df["datetime"]).dt.date
                    all_dates.update(dates[(dates >= start) & (dates <= end)].tolist())

        trading_days = sorted(all_dates)
        logger.info(f"Backtest: {len(trading_days)} trading days, {len(symbols)} symbols")

        # Buffered signals from previous day
        pending_orders: List[OrderEvent] = []

        for day_idx, trade_date in enumerate(trading_days):
            # Step 1: Execute pending orders from yesterday's signals
            for order in pending_orders:
                symbol = order.symbol
                if symbol not in self.data:
                    continue
                df = self.data[symbol]
                day_data = self._get_day_data(df, trade_date)
                if day_data is None:
                    continue

                fill = self.broker.execute_order(
                    order,
                    open_price=day_data["open"],
                    prev_close=day_data.get("prev_close", day_data["open"]),
                    portfolio=self.portfolio,
                )
                if fill is not None:
                    self.portfolio.on_fill(fill, trade_date)
            pending_orders.clear()

            # Step 2: Generate new signals at today's close
            current_prices = {}
            predictions = {}
            for symbol in symbols:
                day_data = self._get_day_data(self.data.get(symbol, pd.DataFrame()), trade_date)
                if day_data is not None:
                    current_prices[symbol] = day_data["close"]

            # Use model predictions if available, otherwise simple momentum
            if self.model is not None:
                predictions = self._get_predictions(symbols, trade_date)
            else:
                # Fallback: simple momentum signal for testing
                for symbol in symbols:
                    df = self.data.get(symbol, pd.DataFrame())
                    if df.empty:
                        continue
                    day_data = self._get_day_data(df, trade_date)
                    if day_data is not None:
                        ret = day_data.get("return_1d", 0)
                        predictions[symbol] = ret if isinstance(ret, (int, float)) else 0

            signals = self.signal_generator.generate_signals(
                predictions, self.portfolio.positions, current_prices
            )

            # Step 3: Risk management filter
            approved_signals = self.risk_manager.filter(
                signals, self.portfolio, current_prices
            )

            # Step 4: Convert signals to orders (execute tomorrow)
            for signal in approved_signals:
                vol = self._estimate_volatility(signal.symbol, trade_date)
                shares = self.position_sizer.size(
                    signal, self.portfolio.total_value, signal.price, vol
                )
                order = OrderEvent(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    quantity=shares,
                )
                pending_orders.append(order)

            # Step 5: End of day
            self.portfolio.on_market_close(trade_date, current_prices)
            self.risk_manager.on_market_close(self.portfolio, current_prices)

        # Compute statistics
        metrics = BacktestStatistics.compute(
            self.portfolio.equity_curve,
            self.portfolio.trade_history,
            self.config.get("backtest", {}).get("risk_free_rate", 0.02),
        )
        return metrics

    def _get_day_data(self, df: pd.DataFrame, trade_date: date) -> dict:
        if df.empty or "datetime" not in df.columns:
            return None
        df_dt = pd.to_datetime(df["datetime"]).dt.date
        mask = df_dt == trade_date
        if not mask.any():
            return None
        row = df[mask].iloc[0]

        # Get previous close
        idx = mask.idxmax()
        prev_close = df["close"].iloc[idx - 1] if idx > 0 else row["close"]

        result = {"open": row["open"], "close": row["close"], "prev_close": prev_close}
        if "return_1d" in df.columns:
            result["return_1d"] = row["return_1d"]
        return result

    def _get_predictions(self, symbols: List[str], trade_date: date) -> Dict[str, float]:
        predictions = {}
        # This would use the trained model to predict
        # For now, return empty (model inference happens in the training script)
        return predictions

    def _estimate_volatility(self, symbol: str, trade_date: date) -> float:
        df = self.data.get(symbol, pd.DataFrame())
        if df.empty or "return_1d" not in df.columns:
            return 0.02
        recent = df[df["datetime"].dt.date <= trade_date].tail(20)
        if len(recent) < 5:
            return 0.02
        return float(recent["return_1d"].std())

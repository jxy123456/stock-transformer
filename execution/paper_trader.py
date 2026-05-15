from datetime import date
from typing import Dict

import numpy as np
import torch
from loguru import logger

from backtest.broker import ASHareBroker
from backtest.portfolio import Portfolio
from data.dataset import TimeSeriesDataset
from data.fetcher import AShareFetcher
from data.features import FeatureEngine
from risk.manager import RiskManager
from strategy.position_sizer import PositionSizer
from strategy.signal_generator import SignalGenerator
from torch.utils.data import DataLoader


class PaperTrader:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        feature_engine: FeatureEngine,
    ):
        self.model = model
        self.config = config
        self.feature_engine = feature_engine

        self.fetcher = AShareFetcher(
            source=config.get("data", {}).get("source", "akshare"),
            tushare_token=config.get("data", {}).get("tushare_token", ""),
        )
        self.signal_generator = SignalGenerator(config)
        self.position_sizer = PositionSizer(config)
        self.risk_manager = RiskManager(config)
        self.broker = ASHareBroker(config)
        initial_cash = config.get("backtest", {}).get("initial_cash", 1_000_000)
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_daily(self, symbols: list, trade_date: date):
        logger.info(f"Paper trade: {trade_date}")

        # Fetch latest data
        current_prices = {}
        predictions = {}

        for symbol in symbols:
            end_date = trade_date.strftime("%Y%m%d")
            start_date = (trade_date - __import__("datetime").timedelta(days=120)).strftime("%Y%m%d")
            df = self.fetcher.fetch_daily(symbol, start_date, end_date)

            if df.empty:
                continue

            df = self.feature_engine.compute(df)
            df = df.dropna()

            if df.empty:
                continue

            current_prices[symbol] = df["close"].iloc[-1]

            # Model prediction
            feature_cols = self.feature_engine.feature_columns
            available_cols = [c for c in feature_cols if c in df.columns]
            data = df[available_cols].values.astype(np.float32)

            seq_len = self.config.get("model", {}).get("seq_len", 60)
            if len(data) < seq_len:
                continue

            x = torch.FloatTensor(data[-seq_len:]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x).cpu().numpy()[0]
            predictions[symbol] = pred[0]  # first day predicted return

        # Generate signals
        signals = self.signal_generator.generate_signals(
            predictions, self.portfolio.positions, current_prices
        )

        # Risk filter
        approved = self.risk_manager.filter(signals, self.portfolio, current_prices)

        # Execute
        for signal in approved:
            vol = 0.02  # simplified
            shares = self.position_sizer.size(
                signal, self.portfolio.total_value, signal.price, vol
            )
            from backtest.events import OrderEvent
            order = OrderEvent(
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=shares,
            )
            fill = self.broker.execute_order(
                order, signal.price, signal.price, self.portfolio
            )
            if fill is not None:
                self.portfolio.on_fill(fill, trade_date)

        self.portfolio.on_market_close(trade_date, current_prices)
        self.risk_manager.on_market_close(self.portfolio, current_prices)

        logger.info(
            f"Portfolio value: {self.portfolio.total_value:,.2f}, "
            f"Positions: {len(self.portfolio.positions)}"
        )

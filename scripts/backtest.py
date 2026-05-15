import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from data.fetcher import AShareFetcher
from data.cache import DataCache
from data.features import FeatureEngine
from backtest.broker import ASHareBroker
from backtest.engine import BacktestEngine
from backtest.portfolio import Portfolio
from backtest.statistics import BacktestStatistics
from risk.manager import RiskManager
from strategy.signal_generator import SignalGenerator
from strategy.position_sizer import PositionSizer


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbols", nargs="+", default=["000001"], help="Stock symbols")
    parser.add_argument("--start", type=str, default="20220101")
    parser.add_argument("--end", type=str, default="20241231")
    parser.add_argument("--cash", type=float, default=1_000_000)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("backtest")

    # Fetch data
    fetcher = AShareFetcher(source=config.get("data.source", "akshare"))
    cache = DataCache(cache_dir=config.get("data.cache_dir", "outputs/data_cache"))
    feature_engine = FeatureEngine(config.raw)

    data = {}
    for symbol in args.symbols:
        df = cache.get(symbol, args.start, args.end)
        if df.empty:
            df = fetcher.fetch_daily(symbol, args.start, args.end)
            if not df.empty:
                cache.put(symbol, args.start, args.end, df)
        if not df.empty:
            df = feature_engine.compute(df)
            df["datetime"] = df["datetime"] if "datetime" in df.columns else df.index
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} rows")

    if not data:
        logger.error("No data available for backtest")
        return

    # Setup components
    portfolio = Portfolio(initial_cash=args.cash)
    broker = ASHareBroker(config.raw)
    signal_gen = SignalGenerator(config.raw)
    position_sizer = PositionSizer(config.raw)
    risk_mgr = RiskManager(config.raw)

    engine = BacktestEngine(
        data=data,
        signal_generator=signal_gen,
        position_sizer=position_sizer,
        risk_manager=risk_mgr,
        broker=broker,
        portfolio=portfolio,
        config=config.raw,
    )

    # Run
    metrics = engine.run(args.symbols, args.start, args.end)
    report = BacktestStatistics.format_report(metrics)
    print(report)

    # Save results
    output_dir = Path("outputs/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "backtest_report.txt", "w") as f:
        f.write(report)

    # Save equity curve
    if portfolio.equity_curve:
        import pandas as pd
        eq_df = pd.DataFrame(portfolio.equity_curve, columns=["date", "value"])
        eq_df.to_csv(output_dir / "equity_curve.csv", index=False)

    # Save trade history
    if portfolio.trade_history:
        import pandas as pd
        trades_df = pd.DataFrame(portfolio.trade_history)
        trades_df.to_csv(output_dir / "trades.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

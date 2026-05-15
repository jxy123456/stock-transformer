import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.visualization import (
    plot_equity_curve,
    plot_predictions,
    plot_training_history,
    plot_trade_signals,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--results-dir", type=str, default="outputs/backtest_results")
    parser.add_argument("--symbol", type=str, default="000001")
    args = parser.parse_args()

    logger = setup_logger("analyze")
    results_dir = Path(args.results_dir)

    import pandas as pd

    # Plot equity curve
    eq_path = results_dir / "equity_curve.csv"
    if eq_path.exists():
        df = pd.read_csv(eq_path)
        equity_curve = list(zip(pd.to_datetime(df["date"]).dt.date, df["value"]))
        plot_equity_curve(
            equity_curve,
            title=f"Equity Curve - {args.symbol}",
            save_path=str(results_dir / "equity_curve.png"),
        )
        logger.info(f"Equity curve plot saved")

    # Print trade summary
    trades_path = results_dir / "trades.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        logger.info(f"Total trades: {len(trades)}")
        if not trades.empty and "direction" in trades.columns:
            buys = len(trades[trades["direction"] == "BUY"])
            sells = len(trades[trades["direction"] == "SELL"])
            logger.info(f"Buys: {buys}, Sells: {sells}")
            if "commission" in trades.columns:
                logger.info(f"Total commission: {trades['commission'].sum():,.2f}")

    # Print report
    report_path = results_dir / "backtest_report.txt"
    if report_path.exists():
        print(report_path.read_text())


if __name__ == "__main__":
    main()

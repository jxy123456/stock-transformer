import numpy as np
import pandas as pd


class BacktestStatistics:
    @staticmethod
    def compute(equity_curve: list, trade_history: list, risk_free_rate: float = 0.02) -> dict:
        if not equity_curve:
            return {}

        dates = [e[0] for e in equity_curve]
        values = np.array([e[1] for e in equity_curve])

        # Returns
        daily_returns = np.diff(values) / values[:-1]
        total_return = (values[-1] - values[0]) / values[0]
        n_days = len(values) - 1
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Sharpe ratio
        if np.std(daily_returns) > 1e-8:
            sharpe = (np.mean(daily_returns) - risk_free_rate / 252) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 1e-8:
            sortino = (np.mean(daily_returns) - risk_free_rate / 252) / np.std(downside) * np.sqrt(252)
        else:
            sortino = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = float(np.max(drawdown))

        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 1e-8 else 0.0

        # Trade statistics
        trades = trade_history
        n_trades = len(trades)
        if n_trades > 0:
            buy_trades = [t for t in trades if t["direction"] == "BUY"]
            sell_trades = [t for t in trades if t["direction"] == "SELL"]
            n_wins = 0
            total_profit = 0.0
            total_loss = 0.0
            for sell in sell_trades:
                matching_buys = [b for b in buy_trades if b["symbol"] == sell["symbol"] and b.get("_matched", False) is False]
                if matching_buys:
                    buy = matching_buys[0]
                    buy["_matched"] = True
                    pnl = (sell["price"] - buy["price"]) * sell["quantity"]
                    if pnl > 0:
                        n_wins += 1
                        total_profit += pnl
                    else:
                        total_loss += abs(pnl)

            win_rate = n_wins / max(len(sell_trades), 1)
            profit_factor = total_profit / max(total_loss, 1e-8)
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Total commission
        total_commission = sum(t.get("commission", 0) for t in trades)

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "n_trades": n_trades,
            "total_commission": float(total_commission),
            "final_value": float(values[-1]),
            "n_days": n_days,
        }

    @staticmethod
    def format_report(metrics: dict) -> str:
        if not metrics:
            return "No backtest data"

        lines = [
            "=" * 50,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 50,
            f"Total Return:      {metrics['total_return']:.2%}",
            f"Annualized Return: {metrics['annualized_return']:.2%}",
            f"Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}",
            f"Sortino Ratio:     {metrics['sortino_ratio']:.4f}",
            f"Max Drawdown:      {metrics['max_drawdown']:.2%}",
            f"Calmar Ratio:      {metrics['calmar_ratio']:.4f}",
            "-" * 50,
            f"Win Rate:          {metrics['win_rate']:.2%}",
            f"Profit Factor:     {metrics['profit_factor']:.4f}",
            f"Total Trades:      {metrics['n_trades']}",
            f"Total Commission:  {metrics['total_commission']:.2f}",
            "-" * 50,
            f"Final Value:       {metrics['final_value']:,.2f}",
            f"Trading Days:      {metrics['n_days']}",
            "=" * 50,
        ]
        return "\n".join(lines)

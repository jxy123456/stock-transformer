"""事件驱动回测引擎。"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.special import softmax

from backtest.base import BaseBacktest
from data.features.v1_45 import CENTERS_1D, CENTERS_5D, CENTERS_20D, V1_45FeatureEngine

BUCKET_CENTERS = {"1d": CENTERS_1D, "5d": CENTERS_5D, "20d": CENTERS_20D}


class EventDrivenBacktest(BaseBacktest):
    def __init__(self):
        self.portfolio = None

    def run(self, model, data: dict, config: dict) -> dict:
        bc = config.get("backtest", {})
        initial_cash = bc.get("initial_cash", 1_000_000)
        top_k = bc.get("top_k", 50)
        weights = bc.get("scoring_weights", {"1d": 0.2, "5d": 0.5, "20d": 0.3})
        risk = bc.get("risk", {})
        max_positions = risk.get("max_positions", 30)
        single_pct = risk.get("single_pct", 0.05)
        stop_loss = risk.get("stop_loss", 0.08)

        seq_len = config.get("features", {}).get("seq_len", 120)
        device = next(model.parameters()).device
        model.eval()

        # collect all trading dates
        all_dates = set()
        for df in data.values():
            if "datetime" in df.columns:
                all_dates.update(pd.to_datetime(df["datetime"]).dt.date)
        trading_days = sorted(all_dates)
        start_idx = 0
        while start_idx < len(trading_days) and trading_days[start_idx] < date(2015, 1, 1):
            start_idx += 1

        cash = initial_cash
        holdings = {}  # {symbol: {"shares": int, "cost": float}}
        equity = []
        trades = []
        pending_buys = []
        pending_sells = {}

        feature_cols = [c for c in V1_45FeatureEngine(config, None).feature_columns
                        if any(c in df.columns for df in data.values())]

        logger.info(f"Backtest: {len(trading_days)} trading days, {len(data)} stocks")

        for di in range(start_idx, len(trading_days)):
            td = trading_days[di]

            prices, opens = {}, {}
            for s in data:
                df = data[s]
                df_dt = pd.to_datetime(df["datetime"]).dt.date
                mask = df_dt == td
                if mask.any():
                    row = df[mask].iloc[0]
                    prices[s] = float(row["close"])
                    opens[s] = float(row["open"]) if "open" in row else prices[s]

            for s, reason in list(pending_sells.items()):
                if s not in holdings:
                    pending_sells.pop(s, None)
                    continue
                px = opens.get(s)
                if px is None:
                    continue
                cash += holdings[s]["shares"] * px * 0.998  # ~stamp tax + commission
                trades.append({"date": td, "symbol": s, "action": "SELL",
                               "price": px, "shares": holdings[s]["shares"],
                               "cost": holdings[s]["cost"], "reason": reason})
                del holdings[s]
                pending_sells.pop(s, None)

            for s in pending_buys:
                if s in holdings or len(holdings) >= max_positions:
                    continue
                px = opens.get(s)
                if px is None or px <= 0:
                    continue
                shares = int(cash * single_pct / px / 100) * 100
                if shares >= 100 and shares * px <= cash:
                    cash -= shares * px * 1.0003  # commission
                    holdings[s] = {"shares": shares, "cost": px}
                    trades.append({"date": td, "symbol": s, "action": "BUY",
                                   "price": px, "shares": shares})
            pending_buys = []

            # generate signals after today's close; orders execute next open
            scores = {}
            for s in data:
                df = data[s]
                df_dt = pd.to_datetime(df["datetime"]).dt.date
                mask = df_dt <= td
                if mask.sum() < seq_len:
                    continue
                window = df.reindex(columns=feature_cols, fill_value=0.0).iloc[mask.values][-seq_len:]
                if len(window) < seq_len:
                    continue
                x = torch.FloatTensor(window.values).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                er_1d = softmax(out["logits_1d"].cpu().numpy(), axis=-1) @ CENTERS_1D
                er_5d = softmax(out["logits_5d"].cpu().numpy(), axis=-1) @ CENTERS_5D
                er_20d = softmax(out["logits_20d"].cpu().numpy(), axis=-1) @ CENTERS_20D
                scores[s] = (weights["1d"] * er_1d[0] + weights["5d"] * er_5d[0] +
                             weights["20d"] * er_20d[0])

            for s, h in list(holdings.items()):
                px = prices.get(s, h["cost"])
                pnl = (px - h["cost"]) / h["cost"]
                if pnl <= -stop_loss:
                    pending_sells[s] = "stop_loss"
                elif scores.get(s, 0) <= 0:
                    pending_sells[s] = "signal"

            ranked = sorted(
                ((s, score) for s, score in scores.items()
                 if score > 0 and s not in holdings and s not in pending_sells),
                key=lambda x: x[1],
                reverse=True,
            )
            slots = min(top_k, max_positions - len(holdings) + len(pending_sells))
            pending_buys = [s for s, _ in ranked[:max(slots, 0)]]

            # mark equity
            total_mv = cash + sum(h["shares"] * prices.get(s, h["cost"])
                                  for s, h in holdings.items())
            equity.append((td, total_mv))

        # compute metrics
        return self._compute_metrics(equity, trades, initial_cash)

    def _compute_metrics(self, equity, trades, initial_cash):
        if not equity:
            return {}
        values = np.array([e[1] for e in equity])
        rets = np.diff(values) / values[:-1]
        total_ret = (values[-1] - values[0]) / values[0]
        ann_ret = (1 + total_ret) ** (252 / max(len(rets), 1)) - 1
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))
        peak = np.maximum.accumulate(values)
        dd = (peak - values) / peak
        max_dd = float(np.max(dd))

        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        n_wins = sum(1 for s in sells
                     if s["price"] > next((b["price"] for b in buys
                       if b["symbol"] == s["symbol"]), s["price"]))
        win_rate = n_wins / max(len(sells), 1)

        return {
            "total_return": float(total_ret),
            "annualized_return": float(ann_ret),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": float(win_rate),
            "n_trades": len(trades),
            "final_value": float(values[-1]),
            "equity_curve": [(e[0].isoformat(), e[1]) for e in equity],
        }

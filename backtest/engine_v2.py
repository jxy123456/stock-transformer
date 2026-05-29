"""回测引擎 v2：numpy 向量化、T+1、A 股规则。"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.special import softmax

from data.features.v1_45 import CENTERS_1D, CENTERS_5D, CENTERS_20D


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        bc = config.get("backtest", {})
        risk = bc.get("risk", {})
        self.initial_cash = bc.get("initial_cash", 20_000)
        self.max_positions = bc.get("max_positions", risk.get("max_positions", 5))
        self.single_pct = bc.get("single_pct", risk.get("single_pct", 0.20))
        self.commission = bc.get("commission", 0.0003)
        self.stamp_tax = bc.get("stamp_tax", 0.001)
        self.slippage = bc.get("slippage", 0.0005)
        self.min_commission = bc.get("min_commission", 5.0)
        self.stop_loss = bc.get("stop_loss", risk.get("stop_loss", 0.08))
        self.take_profit = bc.get("take_profit", risk.get("take_profit", 0.15))
        self.scoring_weights = bc.get("scoring_weights", {"1d": 0.2, "5d": 0.5, "20d": 0.3})
        self.buy_threshold = bc.get("buy_threshold", 0.0)
        self.seq_len = config.get("features", {}).get("seq_len", 120)

    def run(self, model, feature_dfs: dict, norm_stats: dict = None):
        """feature_dfs: {symbol: DataFrame with datetime + features + close}"""
        model.eval()
        device = next(model.parameters()).device
        mean = np.array(norm_stats.get("mean", [])) if norm_stats else None
        std = np.array(norm_stats.get("std", [])) if norm_stats else None
        feature_cols = norm_stats.get("feature_columns", []) if norm_stats else []

        # ---- build common date index, filter to test period ----
        ds_cfg = self.config.get("data_split", {})
        test_start = pd.Timestamp(ds_cfg.get("val_end", "2023-12-31"))
        all_dates = sorted(set().union(*[
            set(pd.to_datetime(df["datetime"]).values) for df in feature_dfs.values()
        ]))
        all_dates = [d for d in all_dates if pd.Timestamp(d) > test_start]
        logger.info(f"Backtest: {len(all_dates)} test days ({test_start.date()}~), {len(feature_dfs)} stocks")

        # prepare pre-loaded numpy arrays per stock for fast slicing
        stock_data = {}
        for s, df in feature_dfs.items():
            dt = pd.to_datetime(df["datetime"]).values
            available = [c for c in feature_cols if c in df.columns]
            feat = df[available].values.astype(np.float32)
            close = df["close"].values.astype(np.float32)
            open_ = df["open"].values.astype(np.float32) if "open" in df.columns else close
            # pre-compute date→index mapping
            date_idx = {pd.Timestamp(d).date(): i for i, d in enumerate(dt)}
            stock_data[s] = {"feat": feat, "close": close, "open": open_, "date_idx": date_idx}

        # fill missing cols with 0 in feature array
        if mean is not None and std is not None:
            n_feat = len(mean)
            for s in stock_data:
                f = stock_data[s]["feat"]
                if f.shape[1] < n_feat:
                    pad = np.zeros((f.shape[0], n_feat - f.shape[1]), dtype=np.float32)
                    stock_data[s]["feat"] = np.concatenate([f, pad], axis=1)

        # ---- state ----
        cash = self.initial_cash
        holdings = {}  # {symbol: {shares, cost, highest}}
        equity = []
        trades = []
        pending_buys = set()  # T+1 tracking

        n_dates = len(all_dates)
        prev_prices = {}

        for di, td in enumerate(all_dates):
            td_date = pd.Timestamp(td).date()
            if (di + 1) % 250 == 1:
                logger.info(f"  [{di+1}/{n_dates}] {td_date}  cash={cash:.0f}  pos={len(holdings)}")

            # get today's close + open prices
            prices, opens = {}, {}
            for s, sd in stock_data.items():
                idx = sd["date_idx"].get(td_date)
                if idx is not None:
                    prices[s] = float(sd["close"][idx])
                    opens[s] = float(sd["open"][idx])

            if not prices:
                continue

            # ---- execute pending T+1 orders from yesterday ----
            to_remove = []
            for s, order in list(holdings.items()):
                if order.get("_pending", False):
                    px = opens.get(s, prices.get(s, order["cost"]))
                    limit_check_px = prev_prices.get(s, px)
                    limit = self._get_limit(s)
                    if px > limit_check_px * (1 + limit) or px < limit_check_px * (1 - limit):
                        to_remove.append(s)  # hit limit, cancel order
                        continue
                    order.pop("_pending")
                    holdings[s] = order
                    pending_buys.discard(s)

            for s in to_remove:
                cash += holdings[s]["shares"] * prices.get(s, holdings[s]["cost"]) * 0.998
                trades.append({"date": str(td_date), "symbol": s, "action": "SELL",
                               "price": prices.get(s), "shares": holdings[s]["shares"],
                               "cost": holdings[s]["cost"], "reason": "limit_hit"})
                del holdings[s]

            pending_buys.clear()

            # ---- check stop-loss / take-profit on current holdings ----
            forced_sells = []
            for s, h in holdings.items():
                px = prices.get(s, h["cost"])
                pnl = (px - h["cost"]) / h["cost"]
                if pnl <= -self.stop_loss:
                    forced_sells.append((s, "stop_loss"))
                elif pnl >= self.take_profit:
                    forced_sells.append((s, "take_profit"))
                else:
                    h["highest"] = max(h.get("highest", px), px)

            for s, reason in forced_sells:
                px = prices.get(s, holdings[s]["cost"])
                trade_val = px * holdings[s]["shares"]
                cost = max(trade_val * self.commission, self.min_commission) + trade_val * self.stamp_tax
                cash += trade_val - cost
                trades.append({"date": str(td_date), "symbol": s, "action": "SELL",
                               "price": px, "shares": holdings[s]["shares"],
                               "cost": holdings[s]["cost"], "reason": reason})
                del holdings[s]

            # ---- batch inference ----
            scores = {}
            batch_x, batch_syms = [], []
            for s, sd in stock_data.items():
                idx = sd["date_idx"].get(td_date)
                if idx is None or idx < self.seq_len:
                    continue
                x = sd["feat"][idx - self.seq_len : idx]
                if mean is not None:
                    x = (x - mean) / (std + 1e-8)
                batch_x.append(x)
                batch_syms.append(s)
                if len(batch_x) >= 256:  # process in chunks to avoid OOM
                    self._infer_chunk(model, device, batch_x, batch_syms, scores)
                    batch_x, batch_syms = [], []
            if batch_x:
                self._infer_chunk(model, device, batch_x, batch_syms, scores)

            # ---- sell holdings with negative score ----
            to_sell = [s for s in holdings if scores.get(s, 0) < self.buy_threshold]
            for s in to_sell:
                px = prices.get(s, holdings[s]["cost"])
                trade_val = px * holdings[s]["shares"]
                cost = max(trade_val * self.commission, self.min_commission) + trade_val * self.stamp_tax
                cash += trade_val - cost
                trades.append({"date": str(td_date), "symbol": s, "action": "SELL",
                               "price": px, "shares": holdings[s]["shares"],
                               "cost": holdings[s]["cost"], "reason": "signal"})
                del holdings[s]

            # ---- buy top-K stocks ----
            slots = self.max_positions - len(holdings)
            if slots > 0:
                candidates = [(s, v) for s, v in scores.items()
                              if v > self.buy_threshold and s not in holdings]
                candidates.sort(key=lambda x: x[1], reverse=True)

                for s, _ in candidates[:slots]:
                    px = prices.get(s)
                    if px is None or px <= 0:
                        continue
                    target_val = cash * self.single_pct
                    shares = max(int(target_val / px / 100) * 100, 100)
                    if shares * px > cash * 0.3:  # don't spend >30% of cash on one lot
                        continue
                    cost = max(shares * px * self.commission, self.min_commission)
                    if shares * px + cost > cash:
                        continue
                    cash -= shares * px + cost
                    holdings[s] = {"shares": shares, "cost": px, "highest": px}
                    pending_buys.add(s)
                    trades.append({"date": str(td_date), "symbol": s, "action": "BUY",
                                   "price": px, "shares": shares})

            # ---- mark equity ----
            total_mv = cash + sum(h["shares"] * prices.get(s, h["cost"])
                                  for s, h in holdings.items())
            equity.append((str(td_date), total_mv))
            prev_prices = prices

        return self._compute_metrics(equity, trades), equity, trades

    def _infer_chunk(self, model, device, batch_x, batch_syms, scores):
        x_t = torch.FloatTensor(np.stack(batch_x)).to(device)
        with torch.no_grad():
            out = model(x_t)
        er1 = softmax(out["logits_1d"].cpu().numpy(), axis=-1) @ CENTERS_1D
        er5 = softmax(out["logits_5d"].cpu().numpy(), axis=-1) @ CENTERS_5D
        er20 = softmax(out["logits_20d"].cpu().numpy(), axis=-1) @ CENTERS_20D
        for i, s in enumerate(batch_syms):
            scores[s] = (self.scoring_weights["1d"] * er1[i].item() +
                         self.scoring_weights["5d"] * er5[i].item() +
                         self.scoring_weights["20d"] * er20[i].item())

    def _get_limit(self, symbol: str) -> float:
        if symbol.startswith("300") or symbol.startswith("688"):
            return 0.20
        if "ST" in symbol.upper():
            return 0.05
        return 0.10

    def _compute_metrics(self, equity, trades):
        if len(equity) < 2:
            return {}
        values = np.array([e[1] for e in equity])
        rets = np.diff(values) / values[:-1]
        days = len(rets)

        total_ret = (values[-1] - values[0]) / values[0]
        ann_ret = (1 + total_ret) ** (252 / max(days, 1)) - 1

        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))
        downside = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(downside) + 1e-8) * np.sqrt(252)) if len(downside) > 0 else 0.0

        peak = np.maximum.accumulate(values)
        dd = (peak - values) / peak
        max_dd = float(np.max(dd))

        sells = [t for t in trades if t["action"] == "SELL"]
        wins = sum(1 for t in sells if t["price"] > t.get("cost", 0))
        win_rate = wins / max(len(sells), 1) if sells else 0.0

        total_commission = sum(max(t["price"] * t["shares"] * (self.commission + self.stamp_tax)
                                   if t["action"] == "SELL"
                                   else t["price"] * t["shares"] * self.commission,
                                   self.min_commission) for t in trades)

        return {
            "total_return": float(total_ret),
            "annualized_return": float(ann_ret),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate": float(win_rate),
            "n_trades": len(trades),
            "n_buys": sum(1 for t in trades if t["action"] == "BUY"),
            "n_sells": len(sells),
            "total_commission": float(total_commission),
            "final_value": float(values[-1]),
            "n_days": days,
        }

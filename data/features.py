import numpy as np
import pandas as pd


class FeatureEngine:
    """32 ratio-based features from OHLCV. No absolute prices."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()
        c = result["close"]
        h = result["high"]
        l = result["low"]
        o = result["open"]
        v = result["volume"]
        a = result["amount"]

        self._compute_log_returns(result, c)
        self._compute_kline_structure(result, o, h, l, c)
        self._compute_trend_position(result, c)
        self._compute_price_position(result, c, h, l)
        self._compute_volume_features(result, v, a)
        self._compute_volatility(result, c, h, l)
        self._compute_drawdown(result, c)

        return result

    # ---- A. Log Returns (5) ----

    @staticmethod
    def _compute_log_returns(result: pd.DataFrame, c: pd.Series):
        for n in [1, 5, 20, 60, 120]:
            result[f"log_ret_{n}d"] = np.log(c / c.shift(n))

    # ---- A. K-line Structure (4) ----

    @staticmethod
    def _compute_kline_structure(result: pd.DataFrame, o, h, l, c):
        prev_c = c.shift(1)
        result["intraday_ret"] = (c - o) / o
        result["overnight_ret"] = (o - prev_c) / prev_c
        result["high_low_range"] = (h - l) / o
        result["close_position_in_bar"] = (c - l) / (h - l + 1e-8)

    # ---- A. Trend Position (5) ----

    @staticmethod
    def _compute_trend_position(result: pd.DataFrame, c: pd.Series):
        for n in [20, 60, 120]:
            ma = c.rolling(n).mean()
            result[f"close_to_ma{n}"] = c / ma - 1

        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        ma120 = c.rolling(120).mean()

        result["ma20_to_ma60"] = ma20 / ma60 - 1
        result["ma60_to_ma120"] = ma60 / ma120 - 1

    # ---- B. Price Position (6) + Drawdown (2) ----

    @staticmethod
    def _compute_price_position(result: pd.DataFrame, c, h, l):
        for n in [60, 250]:
            result[f"price_rank_{n}d"] = c.rolling(n).rank(pct=True)
            result[f"distance_to_high_{n}d"] = c / h.rolling(n).max() - 1
            result[f"distance_to_low_{n}d"] = c / l.rolling(n).min() - 1

    # ---- C. Volume (4) ----

    @staticmethod
    def _compute_volume_features(result: pd.DataFrame, v, a):
        result["log_amount"] = np.log1p(a)

        v_ma5 = v.rolling(5).mean()
        v_ma20 = v.rolling(20).mean()

        result["volume_ratio_1_20"] = v / (v_ma20 + 1e-8)
        result["volume_ratio_5_20"] = v_ma5 / (v_ma20 + 1e-8)

        v_std20 = v.rolling(20).std()
        result["volume_zscore_20d"] = (v - v_ma20) / (v_std20 + 1e-8)

    # ---- D. Volatility (6) ----

    @staticmethod
    def _compute_volatility(result: pd.DataFrame, c, h, l):
        log_ret = np.log(c / c.shift(1))

        for n in [20, 60]:
            result[f"realized_vol_{n}d"] = np.sqrt((log_ret ** 2).rolling(n).sum())

        neg_ret = log_ret.where(log_ret < 0, 0.0)
        for n in [20, 60]:
            result[f"downside_vol_{n}d"] = np.sqrt((neg_ret ** 2).rolling(n).sum())

        result["downside_vol_ratio_20d"] = result["downside_vol_20d"] / (result["realized_vol_20d"] + 1e-8)

        hl_ratio = np.log(h / l)
        result["high_low_vol_20d"] = np.sqrt((hl_ratio ** 2).rolling(20).sum())

    # ---- B. Drawdown (2) ----

    @staticmethod
    def _compute_drawdown(result: pd.DataFrame, c: pd.Series):
        ret = c.pct_change()
        cum = (1 + ret).cumprod()
        result["max_drawdown_250d"] = cum.rolling(250).apply(
            lambda x: _max_dd(x), raw=True
        )

        rolling_min_60 = c.rolling(60).min()
        result["rebound_from_low_60d"] = c / rolling_min_60 - 1

    @property
    def feature_columns(self) -> list:
        return [
            # A. 收益与趋势 (14)
            "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d", "log_ret_120d",
            "intraday_ret", "overnight_ret", "high_low_range", "close_position_in_bar",
            "close_to_ma20", "close_to_ma60", "close_to_ma120",
            "ma20_to_ma60", "ma60_to_ma120",
            # B. 价格位置与回撤 (8)
            "price_rank_60d", "price_rank_250d",
            "distance_to_high_60d", "distance_to_high_250d",
            "distance_to_low_60d", "distance_to_low_250d",
            "max_drawdown_250d", "rebound_from_low_60d",
            # C. 成交量 (4)
            "log_amount", "volume_ratio_1_20", "volume_ratio_5_20", "volume_zscore_20d",
            # D. 波动率 (6)
            "realized_vol_20d", "realized_vol_60d",
            "downside_vol_20d", "downside_vol_60d", "downside_vol_ratio_20d", "high_low_vol_20d",
        ]


def _max_dd(cum_returns: np.ndarray) -> float:
    if len(cum_returns) < 2:
        return 0.0
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / (peak + 1e-8)
    return dd.min()

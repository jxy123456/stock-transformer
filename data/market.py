import numpy as np
import pandas as pd


class MarketFeatureEngine:
    """9 market-level features: beta, correlation, relative returns, market state."""

    def __init__(self, index_data: pd.DataFrame):
        self.index_data = index_data

    def compute(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        result = stock_df.copy()

        if self.index_data.empty:
            for col in self.feature_columns:
                result[col] = np.nan
            return result

        idx = self.index_data.copy()
        idx["datetime"] = pd.to_datetime(idx["datetime"])
        idx = idx.set_index("datetime").sort_index()
        idx = idx[~idx.index.duplicated(keep="last")]

        dates = pd.to_datetime(result["datetime"])
        idx_close = idx["close"].reindex(dates, method="ffill").values
        idx_s = pd.Series(idx_close, index=result.index)

        idx_ret = np.log(idx_s / idx_s.shift(1))
        stock_ret = np.log(result["close"] / result["close"].shift(1))

        # Beta and correlation
        cov = stock_ret.rolling(60).cov(idx_ret)
        var = idx_ret.rolling(60).var()
        result["market_beta_60d"] = cov / (var + 1e-8)
        result["market_corr_60d"] = stock_ret.rolling(60).corr(idx_ret)

        # Relative returns vs market
        stock_cum_20 = result["close"] / result["close"].shift(20)
        idx_cum_20 = idx_s / idx_s.shift(20)
        result["relative_ret_20d_vs_market"] = stock_cum_20 / (idx_cum_20 + 1e-8) - 1

        stock_cum_60 = result["close"] / result["close"].shift(60)
        idx_cum_60 = idx_s / idx_s.shift(60)
        result["relative_ret_60d_vs_market"] = stock_cum_60 / (idx_cum_60 + 1e-8) - 1

        # Market state
        result["market_ret_1d"] = idx_ret
        result["market_ret_20d"] = np.log(idx_s / idx_s.shift(20))
        result["market_vol_20d"] = idx_ret.rolling(20).std()
        result["market_drawdown_60d"] = idx_s / idx_s.rolling(60).max() - 1

        # market_up_stock_ratio_1d: written by CrossSectionalEngine if available
        if "market_up_stock_ratio_1d" not in result.columns:
            result["market_up_stock_ratio_1d"] = np.nan

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "market_beta_60d", "market_corr_60d",
            "relative_ret_20d_vs_market", "relative_ret_60d_vs_market",
            "market_ret_1d", "market_ret_20d", "market_vol_20d", "market_drawdown_60d",
            "market_up_stock_ratio_1d",
        ]

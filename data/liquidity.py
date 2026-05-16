import numpy as np
import pandas as pd


class LiquidityFeatureEngine:
    """4 liquidity features: turnover metrics + Amihud illiquidity."""

    def compute(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        result = stock_df.copy()

        # Turnover features (require "turnover" column from daily_basic)
        if "turnover" in result.columns:
            t = result["turnover"]
            result["turnover_ma20"] = t.rolling(20).mean()
            result["turnover_ratio_5_20"] = t.rolling(5).mean() / (t.rolling(20).mean() + 1e-8)
            t_ma20 = t.rolling(20).mean()
            t_std20 = t.rolling(20).std()
            result["turnover_zscore_20d"] = (t - t_ma20) / (t_std20 + 1e-8)
        else:
            result["turnover_ma20"] = np.nan
            result["turnover_ratio_5_20"] = np.nan
            result["turnover_zscore_20d"] = np.nan

        # Amihud illiquidity (needs log_ret_1d and log_amount from FeatureEngine)
        if "log_ret_1d" in result.columns and "log_amount" in result.columns:
            abs_ret_over_amount = result["log_ret_1d"].abs() / (np.exp(result["log_amount"]) + 1e-8)
            result["amihud_illiquidity_20d"] = abs_ret_over_amount.rolling(20).mean()
        else:
            result["amihud_illiquidity_20d"] = np.nan

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "turnover_ma20", "turnover_ratio_5_20", "turnover_zscore_20d",
            "amihud_illiquidity_20d",
        ]

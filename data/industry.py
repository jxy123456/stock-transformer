import numpy as np
import pandas as pd
from scipy.stats import rankdata


class IndustryFeatureEngine:
    """5 industry-relative features requiring industry classification."""

    def __init__(self, industry_map: dict, industry_returns: dict):
        self.industry_map = industry_map
        self.industry_returns = industry_returns
        self._industry_rank_cache = None

    def _compute_industry_ranks(self, result: pd.DataFrame) -> pd.Series:
        """Rank each industry's 20d return among all industries, per date."""
        if self._industry_rank_cache is not None:
            return self._industry_rank_cache

        dates = result["datetime"].values
        industry_20d_rets = {}

        for ind_name, ind_df in self.industry_returns.items():
            if ind_df.empty:
                continue
            ind_close = ind_df.set_index("datetime")["close"].sort_index()
            for date in dates:
                dt = pd.to_datetime(date)
                if dt in ind_close.index:
                    loc = ind_close.index.get_loc(dt)
                    if loc >= 20:
                        ret = ind_close.iloc[loc] / ind_close.iloc[loc - 20] - 1
                        industry_20d_rets.setdefault(dt, {})[ind_name] = ret

        rank_cache = {}
        for dt, rets in industry_20d_rets.items():
            names = list(rets.keys())
            values = np.array([rets[n] for n in names])
            ranks = rankdata(values, method="average") / len(values)
            for name, rank in zip(names, ranks):
                rank_cache[(dt, name)] = rank

        self._industry_rank_cache = rank_cache
        return rank_cache

    def compute(self, stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        result = stock_df.copy()
        industry = self.industry_map.get(symbol, "unknown")

        if industry not in self.industry_returns:
            for col in self.feature_columns:
                result[col] = np.nan
            return result

        ind_df = self.industry_returns[industry]
        ind_close = ind_df.set_index("datetime")["close"]
        ind_close = ind_close.reindex(result["datetime"], method="ffill").values
        ind_s = pd.Series(ind_close, index=result.index)

        ind_ret = np.log(ind_s / ind_s.shift(1))

        stock_cum_20 = result["close"] / result["close"].shift(20)
        ind_cum_20 = ind_s / ind_s.shift(20)
        result["relative_ret_20d_vs_industry"] = stock_cum_20 / (ind_cum_20 + 1e-8) - 1

        stock_cum_60 = result["close"] / result["close"].shift(60)
        ind_cum_60 = ind_s / ind_s.shift(60)
        result["relative_ret_60d_vs_industry"] = stock_cum_60 / (ind_cum_60 + 1e-8) - 1

        result["industry_ret_20d"] = np.log(ind_s / ind_s.shift(20))
        result["industry_vol_20d"] = ind_ret.rolling(20).std()

        # Industry rank: where this industry's 20d return ranks among all industries
        rank_cache = self._compute_industry_ranks(result)
        rank_vals = []
        for dt in result["datetime"]:
            key = (pd.to_datetime(dt), industry)
            rank_vals.append(rank_cache.get(key, np.nan))
        result["industry_rank_ret_20d"] = rank_vals

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "relative_ret_20d_vs_industry", "relative_ret_60d_vs_industry",
            "industry_ret_20d", "industry_vol_20d",
            "industry_rank_ret_20d",
        ]

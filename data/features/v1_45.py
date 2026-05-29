"""39 特征方案。"""

import numpy as np
import pandas as pd

from data.features.base import BaseFeatureEngine

# 5 日收益率桶
BINS_5D = [-np.inf, -0.10, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.10, np.inf]
CENTERS_5D = np.array([-0.12, -0.085, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.085, 0.12])

# 20 日收益率桶
BINS_20D = [-np.inf, -0.25, -0.18, -0.12, -0.08, -0.03, 0.03, 0.08, 0.12, 0.18, 0.25, np.inf]
CENTERS_20D = np.array([-0.30, -0.215, -0.15, -0.10, -0.055, 0.0, 0.055, 0.10, 0.15, 0.215, 0.30])

LOSS_WEIGHTS = {"5d": 0.6, "20d": 0.4}


class V1_45FeatureEngine(BaseFeatureEngine):
    """39-feature engine per the multi-horizon model spec."""

    def __init__(self, config: dict, cache, index_data=None, **kwargs):
        super().__init__(config, cache)
        self.index_data = index_data or {}

    @property
    def feature_columns(self) -> list:
        return [
            # 价格类 (7)
            "open_gap", "intraday_ret", "close_ret_1d", "amplitude",
            "high_to_preclose", "low_to_preclose", "close_to_high",
            # 收益类 (3)
            "ret_5d", "ret_20d", "ret_60d",
            # 波动+回撤 (3)
            "volatility_20d", "volatility_60d", "max_drawdown_20d",
            # 均线偏离 (2)
            "close_to_ma20", "close_to_ma60",
            # 价格位置 (2)
            "price_position_20d", "price_position_60d",
            # 成交量类 (4)
            "volume_chg_5d", "amount_chg_5d", "volume_ratio_20d", "amount_ratio_20d",
            # 换手率 (2)
            "turnover_rate", "turnover_to_20d",
            # 市值类 (3)
            "log_total_market_cap_max_norm", "log_float_market_cap_max_norm",
            "total_market_cap_rank_market",
            # 估值类 (2)
            "earnings_yield", "book_to_price",
            # 基本面 (7)
            "revenue_yoy", "net_profit_yoy", "gross_margin", "net_margin",
            "roe", "debt_to_asset", "ocf_to_net_profit",
            # 大盘 (3)
            "market_ret_20d", "market_volatility_20d",
            "excess_ret_market_20d",
            # 横截面排名 (1)
            "ret_20d_rank_market",
        ]

    # ---- individual stock computation ----

    def compute(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        daily = self.cache.load_daily(symbol)
        if daily.empty:
            return pd.DataFrame()
        # filter date range
        daily["datetime"] = pd.to_datetime(daily["datetime"])
        daily = daily[(daily["datetime"] >= start_date) & (daily["datetime"] <= end_date)]
        if daily.empty:
            return pd.DataFrame()

        result = daily[["datetime", "pre_close", "close"]].copy()
        c, h, l, o, v, a = (daily["close"], daily["high"], daily["low"],
                             daily["open"], daily["volume"], daily["amount"])
        pc = daily["pre_close"]

        self._add_price_features(result, c, h, l, o, pc)
        self._add_return_features(result, c)
        self._add_vol_dd_features(result, c, h, l)
        self._add_ma_features(result, c)
        self._add_position_features(result, c, h, l)
        self._add_volume_features(result, v, a)

        # valuation-based features
        val = self.cache.load_valuation(symbol)
        if not val.empty:
            val["datetime"] = pd.to_datetime(val["datetime"])
            self._add_valuation_features(result, val)

        # fundamental features
        fin = self.cache.load_financial(symbol)
        if not fin.empty:
            self._add_fundamental_features(result, daily, fin)

        # market features
        self._add_market_features(result, daily)

        result = result.drop(columns=["pre_close"])  # keep datetime + close
        return result

    # ---- price features (1-7) ----

    def _add_price_features(self, df, c, h, l, o, pc):
        df["open_gap"] = o / pc - 1
        df["intraday_ret"] = c / o - 1
        df["close_ret_1d"] = c / pc - 1
        df["amplitude"] = h / l - 1
        df["high_to_preclose"] = h / pc - 1
        df["low_to_preclose"] = l / pc - 1
        df["close_to_high"] = c / h - 1

    # ---- return features (8-10) ----

    def _add_return_features(self, df, c):
        df["ret_5d"] = c / c.shift(5) - 1
        df["ret_20d"] = c / c.shift(20) - 1
        df["ret_60d"] = c / c.shift(60) - 1

    # ---- volatility & drawdown (11-13) ----

    def _add_vol_dd_features(self, df, c, h, l):
        ret_1d = c / c.shift(1) - 1
        df["volatility_20d"] = ret_1d.rolling(20).std()
        df["volatility_60d"] = ret_1d.rolling(60).std()
        peak_20 = h.rolling(20).max()
        dd_20 = c / peak_20 - 1
        df["max_drawdown_20d"] = dd_20.rolling(20).min()

    # ---- MA deviation (14-15) ----

    def _add_ma_features(self, df, c):
        df["close_to_ma20"] = c / c.rolling(20).mean() - 1
        df["close_to_ma60"] = c / c.rolling(60).mean() - 1

    # ---- price position (16-17) ----

    def _add_position_features(self, df, c, h, l):
        min20, max20 = l.rolling(20).min(), h.rolling(20).max()
        min60, max60 = l.rolling(60).min(), h.rolling(60).max()
        df["price_position_20d"] = (c - min20) / (max20 - min20 + 1e-8)
        df["price_position_60d"] = (c - min60) / (max60 - min60 + 1e-8)

    # ---- volume features (18-21) ----

    def _add_volume_features(self, df, v, a):
        df["volume_chg_5d"] = v / v.shift(5) - 1
        df["amount_chg_5d"] = a / a.shift(5) - 1
        vma20 = v.rolling(20).mean()
        ama20 = a.rolling(20).mean()
        df["volume_ratio_20d"] = v / (vma20 + 1e-8)
        df["amount_ratio_20d"] = a / (ama20 + 1e-8)

    # ---- valuation features (22-30) ----

    def _add_valuation_features(self, df, val: pd.DataFrame):
        try:
            val["datetime"] = pd.to_datetime(val["datetime"], errors="coerce")
            val = val.dropna(subset=["datetime"])
            val = val.set_index("datetime").sort_index()
            val = val[~val.index.duplicated(keep="last")]
            dates = pd.to_datetime(df["datetime"].values)
            combined_idx = val.index.union(dates)
            aligned = val.reindex(combined_idx).ffill().reindex(dates)
        except Exception:
            for col in ["turnover_rate", "turnover_to_20d",
                         "log_total_market_cap_max_norm", "log_float_market_cap_max_norm",
                         "total_market_cap_rank_market",
                         "earnings_yield", "book_to_price"]:
                df[col] = np.nan
            return

        # turnover (22-23)
        if "turnover_rate" in aligned.columns:
            t = aligned["turnover_rate"]
            df["turnover_rate"] = t.values
            t_ma20 = t.rolling(20).mean()
            df["turnover_to_20d"] = (t / (t_ma20 + 1e-8)).values
        else:
            df["turnover_rate"] = np.nan
            df["turnover_to_20d"] = np.nan

        # market cap (24-27) — raw values, cross-sectional normalization deferred
        if "market_cap" in aligned.columns:
            mc = aligned["market_cap"]
            log_mc = np.log1p(mc)
            daily_max = log_mc.groupby(dates).transform("max")
            df["log_total_market_cap_max_norm"] = (log_mc / (daily_max + 1e-8)).values
            df["total_market_cap_rank_market"] = np.nan
        else:
            df["log_total_market_cap_max_norm"] = np.nan
            df["total_market_cap_rank_market"] = np.nan

        if "circ_mv" in aligned.columns:
            cmv = aligned["circ_mv"]
            log_cmv = np.log1p(cmv)
            daily_max_c = log_cmv.groupby(dates).transform("max")
            df["log_float_market_cap_max_norm"] = (log_cmv / (daily_max_c + 1e-8)).values
        else:
            df["log_float_market_cap_max_norm"] = np.nan

        # valuation
        pe = aligned.get("pe", pd.Series(np.nan, index=aligned.index))
        pb = aligned.get("pb", pd.Series(np.nan, index=aligned.index))
        df["earnings_yield"] = (1.0 / pe.replace(0, np.nan)).values
        df["book_to_price"] = (1.0 / pb.replace(0, np.nan)).values

    # ---- fundamental features (31-37) ----

    def _add_fundamental_features(self, df, daily, fin: pd.DataFrame):
        fin = fin.copy()
        date_col = "report_date" if "report_date" in fin.columns else "ann_date" if "ann_date" in fin.columns else None
        if date_col not in fin.columns:
            for col in ["revenue_yoy", "net_profit_yoy", "gross_margin",
                         "net_margin", "roe", "debt_to_asset", "ocf_to_net_profit"]:
                df[col] = np.nan
            return

        try:
            raw_dates = fin[date_col]
            if pd.api.types.is_numeric_dtype(raw_dates) or raw_dates.astype(str).str.fullmatch(r"\d{8}").all():
                fin[date_col] = pd.to_datetime(raw_dates.astype(str), format="%Y%m%d", errors="coerce")
            else:
                fin[date_col] = pd.to_datetime(raw_dates, errors="coerce")
            fin = fin.dropna(subset=[date_col])
            fin = fin.sort_values(date_col)
            dates = pd.DatetimeIndex(pd.to_datetime(daily["datetime"].values))
            effective_pos = dates.searchsorted(fin[date_col].values, side="right")
            valid = effective_pos < len(dates)
            fin = fin.loc[valid].copy()
            if fin.empty:
                raise ValueError("No financial reports are effective within the daily date range")
            fin["_effective_date"] = dates[effective_pos[valid]].values
            fin = fin.set_index("_effective_date").sort_index()
            fin = fin[~fin.index.duplicated(keep="last")]
            combined_idx = fin.index.union(dates)
            fin = fin.reindex(combined_idx).ffill().reindex(dates)
            aligned = fin
        except Exception:
            for col in ["revenue_yoy", "net_profit_yoy", "gross_margin",
                         "net_margin", "roe", "debt_to_asset", "ocf_to_net_profit"]:
                df[col] = np.nan
            return

        df["revenue_yoy"] = self._col(aligned, "revenue_yoy", df)
        df["net_profit_yoy"] = self._col(aligned, "profit_yoy", df)
        df["roe"] = self._col(aligned, "roe", df, fallback="roe_dt")
        df["debt_to_asset"] = self._col(aligned, "debt_to_assets", df)
        df["net_margin"] = self._col(aligned, "netprofit_margin", df)
        df["gross_margin"] = self._col(aligned, "grossprofit_margin", df)
        df["ocf_to_net_profit"] = self._col(aligned, "ocf_to_netprofit", df)

    # ---- market features (38-39, 42) ----

    def _add_market_features(self, df, daily):
        idx_key = self.config.get("index", "000300")
        idx = self.index_data.get(str(idx_key), pd.DataFrame())
        if idx.empty or "datetime" not in idx.columns:
            for col in ["market_ret_20d", "market_volatility_20d", "excess_ret_market_20d"]:
                df[col] = np.nan
            return
        idx = idx.copy()
        idx["datetime"] = pd.to_datetime(idx["datetime"])
        idx = idx.set_index("datetime").sort_index()
        dates = pd.to_datetime(daily["datetime"])
        idx_close = idx["close"].reindex(dates, method="ffill")
        idx_ret = idx_close / idx_close.shift(1) - 1
        df["market_ret_20d"] = (idx_close / idx_close.shift(20) - 1).values
        df["market_volatility_20d"] = idx_ret.rolling(20).std().values
        df["excess_ret_market_20d"] = df["ret_20d"].values - df["market_ret_20d"].values

    # ---- helpers ----

    @staticmethod
    def _col(aligned: pd.DataFrame, col: str, df: pd.DataFrame,
             fallback: str = None) -> np.ndarray:
        if col in aligned.columns:
            return aligned[col].values
        if fallback and fallback in aligned.columns:
            return aligned[fallback].values
        return np.full(len(df), np.nan)

    # ---- batch cross-sectional features ----

    def compute_batch(self, symbols: list, date: str) -> dict:
        """Compute rank features for a single date across all symbols."""
        result = {}
        # metrics needed for ranking: base_value -> target_rank_column
        rank_pairs = [
            ("log_total_market_cap_max_norm", "total_market_cap_rank_market"),
            ("ret_20d", "ret_20d_rank_market"),
        ]

        # collect values for this date
        vals = {}
        for s in symbols:
            df = self.load(s)
            if df.empty:
                continue
            df["datetime"] = pd.to_datetime(df["datetime"])
            row = df[df["datetime"] == date]
            if row.empty:
                continue
            for base_col, _ in rank_pairs:
                if base_col in row.columns:
                    v = row[base_col].iloc[0]
                    if not pd.isna(v):
                        vals.setdefault(base_col, {})[s] = v

        # full-market ranks
        for base_col, rank_col in rank_pairs:
            if base_col in vals:
                ranked = self._rank_pct(vals[base_col])
                for s, r in ranked.items():
                    result.setdefault(s, {})[rank_col] = r

        return result

    @staticmethod
    def _rank_pct(vals: dict) -> dict:
        from scipy.stats import rankdata
        syms = list(vals.keys())
        arr = np.array([vals[s] for s in syms])
        ranks = rankdata(arr, method="average") / len(arr)
        return dict(zip(syms, ranks))


def bucketize(ret: float, bins: list) -> int:
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= ret < hi:
            return i
    return len(bins) - 2

import numpy as np
import pandas as pd
from loguru import logger

from data.fetcher import AShareFetcher
from data.cache import DataCache


class FundamentalEngine:
    """将财报数据转化为逐日特征，可拼接到技术特征中供模型使用。"""

    def __init__(self, fetcher: AShareFetcher = None, cache: DataCache = None):
        self.fetcher = fetcher or AShareFetcher()
        self.cache = cache

    def fetch_and_build(self, symbol: str) -> pd.DataFrame:
        """获取估值指标历史序列，返回按交易日对齐的 DataFrame。"""
        df_val = self._fetch_valuation(symbol)
        df_fin = self._fetch_financial_summary(symbol)

        # 估值指标是日频的，直接可用
        result = df_val.copy()

        # 财报是季频的，forward-fill 到日频
        if not df_fin.empty:
            df_fin = self._align_financial_to_daily(df_fin, result)
            for col in df_fin.columns:
                if col not in result.columns:
                    result[col] = df_fin[col]

        return result

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从原始基本面数据中计算衍生特征。"""
        if df.empty:
            return df

        result = df.copy()

        # PE 分位数（滚动 250 日）
        if "pe" in result.columns:
            result["pe_quantile"] = result["pe"].rolling(250, min_periods=60).rank(pct=True)

        # PB 分位数
        if "pb" in result.columns:
            result["pb_quantile"] = result["pb"].rolling(250, min_periods=60).rank(pct=True)

        # ROE 变化
        if "roe" in result.columns:
            result["roe_change"] = result["roe"].diff()

        # 营收增速变化
        if "revenue_yoy" in result.columns:
            result["revenue_accel"] = result["revenue_yoy"].diff()

        # 净利润增速变化
        if "profit_yoy" in result.columns:
            result["profit_accel"] = result["profit_yoy"].diff()

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "pe", "pe_quantile",
            "pb", "pb_quantile",
            "roe", "roe_change",
            "revenue_yoy", "revenue_accel",
            "profit_yoy", "profit_accel",
            "dv_ratio",  # 股息率
        ]

    def _fetch_valuation(self, symbol: str) -> pd.DataFrame:
        if self.cache:
            cached = self.cache.get(f"{symbol}_val", "19900101", "20991231")
            if not cached.empty:
                return cached

        df = self.fetcher.fetch_valuation_metrics(symbol)
        if df.empty:
            return pd.DataFrame()

        # 标准化列名
        col_map = {
            "trade_date": "datetime",
            "date": "datetime",
            "pe_ttm": "pe",
            "pe": "pe",
            "pb": "pb",
            "ps_ttm": "ps",
            "ps": "ps",
            "dv_ratio": "dv_ratio",
            "dv_ttm": "dv_ratio",
            "total_mv": "market_cap",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        if self.cache:
            self.cache.put(f"{symbol}_val", "19900101", "20991231", df)

        return df

    def _fetch_financial_summary(self, symbol: str) -> pd.DataFrame:
        df = self.fetcher.fetch_financial_summary(symbol)
        if df.empty:
            return pd.DataFrame()

        # 提取关键字段
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "roe" in col_lower:
                col_map[col] = "roe"
            elif "营收" in col and "同比" in col:
                col_map[col] = "revenue_yoy"
            elif "净利" in col and "同比" in col:
                col_map[col] = "profit_yoy"
            elif "日期" in col or "report" in col_lower:
                col_map[col] = "report_date"

        df = df.rename(columns=col_map)

        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])

        return df

    @staticmethod
    def _align_financial_to_daily(
        financial: pd.DataFrame, daily: pd.DataFrame
    ) -> pd.DataFrame:
        """将季频财报 forward-fill 到日频。"""
        if financial.empty or daily.empty or "datetime" not in daily.columns:
            return pd.DataFrame()

        # 用日频日期对财报做 forward fill
        fin_cols = [c for c in financial.columns if c != "report_date"]
        if not fin_cols or "report_date" not in financial.columns:
            return pd.DataFrame()

        fin = financial.set_index("report_date").sort_index()
        daily_dates = daily["datetime"]
        aligned = fin.reindex(daily_dates, method="ffill")
        return aligned.reset_index(drop=True)

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger


class AShareFetcher:
    def __init__(self, source: str = "akshare", tushare_token: str = ""):
        self.source = source
        self.tushare_token = tushare_token

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        if self.source == "akshare":
            df = self._fetch_akshare(symbol, start_date, end_date, adjust)
        else:
            df = self._fetch_tushare(symbol, start_date, end_date, adjust)

        if df is None or df.empty:
            logger.warning(f"Primary source failed for {symbol}, trying fallback")
            if self.source == "akshare":
                df = self._fetch_tushare(symbol, start_date, end_date, adjust)
            else:
                df = self._fetch_akshare(symbol, start_date, end_date, adjust)

        return self._standardize(df)

    def fetch_index(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            df = df.rename(columns={"date": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
            return self._standardize(df)
        except Exception as e:
            logger.error(f"Failed to fetch index {index_code}: {e}")
            return pd.DataFrame()

    def fetch_stock_list(self) -> pd.DataFrame:
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            return df[["代码", "名称"]].rename(columns={"代码": "symbol", "名称": "name"})
        except Exception as e:
            logger.error(f"Failed to fetch stock list: {e}")
            return pd.DataFrame()

    # ---------- Financial Report Data ----------

    def fetch_income_statement(self, symbol: str) -> pd.DataFrame:
        """获取利润表（按报告期）"""
        try:
            import akshare as ak
            df = ak.stock_financial_report_sina(
                stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
                symbol="利润表",
            )
            return df
        except Exception as e:
            logger.warning(f"akshare income statement failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """获取资产负债表"""
        try:
            import akshare as ak
            df = ak.stock_financial_report_sina(
                stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
                symbol="资产负债表",
            )
            return df
        except Exception as e:
            logger.warning(f"akshare balance sheet failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_cashflow_statement(self, symbol: str) -> pd.DataFrame:
        """获取现金流量表"""
        try:
            import akshare as ak
            df = ak.stock_financial_report_sina(
                stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
                symbol="现金流量表",
            )
            return df
        except Exception as e:
            logger.warning(f"akshare cashflow failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_financial_summary(self, symbol: str) -> pd.DataFrame:
        """获取主要财务指标摘要（东方财富数据源，更简洁）"""
        try:
            import akshare as ak
            df = ak.stock_financial_abstract_ths(symbol=symbol)
            return df
        except Exception:
            try:
                import akshare as ak
                df = ak.stock_financial_analysis_indicator(symbol=symbol)
                return df
            except Exception as e2:
                logger.warning(f"Financial summary failed for {symbol}: {e2}")
                return pd.DataFrame()

    def fetch_valuation_metrics(self, symbol: str) -> pd.DataFrame:
        """获取估值指标：PE/PB/PS 等历史序列"""
        try:
            import akshare as ak
            df = ak.stock_a_indicator_lg(symbol=symbol)
            return df
        except Exception as e:
            logger.warning(f"Valuation metrics failed for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_akshare(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            return df
        except Exception as e:
            logger.warning(f"akshare failed for {symbol}: {e}")
            return None

    def _fetch_tushare(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> Optional[pd.DataFrame]:
        try:
            import tushare as ts
            if not self.tushare_token:
                return None
            ts.set_token(self.tushare_token)
            pro = ts.pro_api()
            ts_code = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is None or df.empty:
                return None
            adj = "qfq" if adjust == "qfq" else None
            if adj:
                df_adj = pro.daily_basic(ts_code=ts_code, fields="trade_date,close,pre_close")
            return df
        except Exception as e:
            logger.warning(f"tushare failed for {symbol}: {e}")
            return None

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        column_map = {
            "日期": "datetime",
            "日期时间": "datetime",
            "date": "datetime",
            "开盘": "open",
            "open": "open",
            "收盘": "close",
            "close": "close",
            "最高": "high",
            "high": "high",
            "最低": "low",
            "low": "low",
            "成交量": "volume",
            "volume": "volume",
            "成交额": "amount",
            "amount": "amount",
            "换手率": "turnover",
            "turnover": "turnover",
        }

        df = df.rename(columns=column_map)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

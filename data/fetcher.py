import time
from datetime import datetime
from functools import wraps
from typing import Optional

import pandas as pd
from loguru import logger


def _retry(max_retries=3, base_delay=2):
    """重试装饰器：指数退避，对付限流/断连。"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}, retry in {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator


class AShareFetcher:
    def __init__(self, source: str = "akshare", tushare_token: str = ""):
        self.source = source
        self.tushare_token = tushare_token
        self._ts_pro = None

    @property
    def ts_pro(self):
        if self._ts_pro is None:
            import tushare as ts
            ts.set_token(self.tushare_token)
            self._ts_pro = ts.pro_api()
        return self._ts_pro

    @staticmethod
    def _ts_code(symbol: str) -> str:
        symbol = str(symbol).zfill(6)
        return f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"

    # ==================== 行情数据 ====================

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        if self.source == "tushare":
            df = self._fetch_daily_tushare(symbol, start_date, end_date, adjust)
            if df is None or df.empty:
                logger.warning(f"tushare failed for {symbol}, trying akshare fallback")
                df = self._fetch_daily_akshare(symbol, start_date, end_date, adjust)
        else:
            df = self._fetch_daily_akshare(symbol, start_date, end_date, adjust)
            if df is None or df.empty:
                logger.warning(f"akshare failed for {symbol}, trying tushare fallback")
                df = self._fetch_daily_tushare(symbol, start_date, end_date, adjust)

        return self._standardize(df)

    @_retry(max_retries=3, base_delay=3)
    def _fetch_daily_akshare(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> Optional[pd.DataFrame]:
        import akshare as ak
        return ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust=adjust,
        )

    @_retry(max_retries=3, base_delay=3)
    def _fetch_daily_tushare(
        self, symbol: str, start_date: str, end_date: str, adjust: str
    ) -> Optional[pd.DataFrame]:
        ts_code = self._ts_code(symbol)
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")

        if adjust == "qfq":
            df = self.ts_pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)
            if df is None or df.empty:
                return None
            # tushare 的 daily 是未复权, 需要 adj_factor 做前复权
            df_adj = self.ts_pro.adj_factor(ts_code=ts_code, start_date=sd, end_date=ed)
            if df_adj is not None and not df_adj.empty:
                factor = df_adj.set_index("trade_date")["adj_factor"]
                df = df.set_index("trade_date")
                latest_factor = factor.iloc[0]
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * factor / latest_factor
                df = df.reset_index()
        else:
            df = self.ts_pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)

        if df is None or df.empty:
            return None

        # tushare 列名映射
        df = df.rename(columns={
            "trade_date": "datetime",
            "vol": "volume",
            "pre_close": "pre_close",
        })
        return df

    # ==================== 股票列表 ====================

    @_retry(max_retries=5, base_delay=5)
    def fetch_stock_list(self) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_stock_list_tushare()
        return self._fetch_stock_list_akshare()

    def _fetch_stock_list_tushare(self) -> pd.DataFrame:
        df = self.ts_pro.stock_basic(
            exchange="", list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        if df.empty:
            raise RuntimeError("tushare stock_basic returned empty")
        df = df.rename(columns={"symbol": "code"})
        df["symbol"] = df["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
        return df[["symbol", "name"]]

    def _fetch_stock_list_akshare(self) -> pd.DataFrame:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        return df[["代码", "名称"]].rename(columns={"代码": "symbol", "名称": "name"})

    # ==================== 指数 ====================

    @_retry(max_retries=3, base_delay=3)
    def fetch_index(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_index_tushare(index_code, start_date, end_date)
        return self._fetch_index_akshare(index_code, start_date, end_date)

    def _fetch_index_tushare(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        df = self.ts_pro.index_daily(
            ts_code=f"{index_code}.SH" if index_code.startswith("0") else f"{index_code}.SZ",
            start_date=sd, end_date=ed,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "datetime", "vol": "volume"})
        return self._standardize(df)

    def _fetch_index_akshare(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
        return self._standardize(df)

    # ==================== 估值指标 ====================

    @_retry(max_retries=3, base_delay=2)
    def fetch_valuation_metrics(self, symbol: str) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_valuation_tushare(symbol)
        return self._fetch_valuation_akshare(symbol)

    def _fetch_valuation_tushare(self, symbol: str) -> pd.DataFrame:
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.daily_basic(
            ts_code=ts_code,
            fields="trade_date,close,pe_ttm,pb,ps_ttm,dv_ratio,dv_ttm,turnover_rate,total_mv,circ_mv",
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "trade_date": "datetime",
            "pe_ttm": "pe",
            "ps_ttm": "ps",
            "dv_ttm": "dv_ratio",
            "total_mv": "market_cap",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def _fetch_valuation_akshare(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_a_indicator_lg(symbol=symbol)

    # ==================== 财报数据 ====================

    @_retry(max_retries=3, base_delay=2)
    def fetch_financial_summary(self, symbol: str) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_financial_summary_tushare(symbol)
        return self._fetch_financial_summary_akshare(symbol)

    def _fetch_financial_summary_tushare(self, symbol: str) -> pd.DataFrame:
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.fina_indicator(
            ts_code=ts_code,
            fields="ts_code,ann_date,end_date,roe,roe_dt,netprofit_margin,grossprofit_margin,or_yoy,netprofit_yoy",
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "ann_date": "report_date",
            "end_date": "end_period",
            "or_yoy": "revenue_yoy",
            "netprofit_yoy": "profit_yoy",
        })
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        df["end_period"] = pd.to_datetime(df["end_period"], format="%Y%m%d", errors="coerce")
        return df

    def _fetch_financial_summary_akshare(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        try:
            return ak.stock_financial_abstract_ths(symbol=symbol)
        except Exception:
            return ak.stock_financial_analysis_indicator(symbol=symbol)

    @_retry(max_retries=3, base_delay=2)
    def fetch_income_statement(self, symbol: str) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_income_tushare(symbol)
        return self._fetch_income_akshare(symbol)

    def _fetch_income_tushare(self, symbol: str) -> pd.DataFrame:
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.income(ts_code=ts_code, fields="ts_code,ann_date,end_date,total_revenue,revenue,oper_cost,total_cogs,netprofit")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _fetch_income_akshare(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
            symbol="利润表",
        )

    @_retry(max_retries=3, base_delay=2)
    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_balance_tushare(symbol)
        return self._fetch_balance_akshare(symbol)

    def _fetch_balance_tushare(self, symbol: str) -> pd.DataFrame:
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.balancesheet(ts_code=ts_code, fields="ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _fetch_balance_akshare(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
            symbol="资产负债表",
        )

    @_retry(max_retries=3, base_delay=2)
    def fetch_cashflow_statement(self, symbol: str) -> pd.DataFrame:
        if self.source == "tushare":
            return self._fetch_cashflow_tushare(symbol)
        return self._fetch_cashflow_akshare(symbol)

    def _fetch_cashflow_tushare(self, symbol: str) -> pd.DataFrame:
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.cashflow(ts_code=ts_code, fields="ts_code,ann_date,end_date,n_cashflow_act,n_cashflow_inv_act,n_cashflow_fnc_act")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _fetch_cashflow_akshare(self, symbol: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}",
            symbol="现金流量表",
        )

    # ==================== 列名标准化 ====================

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        column_map = {
            "日期": "datetime",
            "日期时间": "datetime",
            "date": "datetime",
            "trade_date": "datetime",
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
            "vol": "volume",
            "成交额": "amount",
            "amount": "amount",
            "换手率": "turnover",
            "turnover": "turnover",
            "turnover_rate": "turnover",
        }

        df = df.rename(columns=column_map)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

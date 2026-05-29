#!/usr/bin/env python3
"""
A 股数据管理：全量下载 + 增量更新。

用法:
  python scripts/manage_data.py download --all --start 20150101
  python scripts/manage_data.py download --symbols 000001 600519 --start 20150101
  python scripts/manage_data.py update --all
  python scripts/manage_data.py update --symbols 000001 --force
"""
import argparse
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import yaml
from loguru import logger

# ================================================================
#  config
# ================================================================

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_config() -> dict:
    """加载 default.yaml，做 ${ENV_VAR} 插值。"""
    path = CONFIG_DIR / "default.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    def _resolve(v):
        if isinstance(v, str):
            def _sub(m):
                var = m.group(1)
                val = os.environ.get(var)
                if val is None:
                    logger.warning(f"环境变量 '{var}' 未设置，使用空字符串。如用 tushare 需设置。")
                    return ""
                return val
            return re.sub(r"\$\{(\w+)\}", _sub, v)
        if isinstance(v, dict):
            return {k: _resolve(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_resolve(vv) for vv in v]
        return v
    return _resolve(cfg)


# ================================================================
#  logger
# ================================================================

def setup_logger(name: str):
    log_dir = Path(__file__).parent.parent / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_dir / f"{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )
    return logger


# ================================================================
#  trading calendar
# ================================================================

class AShareCalendar:
    def __init__(self):
        self._trading_days: set = set()
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            self._trading_days = set(pd.to_datetime(df["trade_date"]).dt.date)
        except Exception:
            raise RuntimeError("加载 A 股交易日历失败，无法继续。")
        self._loaded = True

    def is_trading_day(self, d: date) -> bool:
        self._ensure_loaded()
        if d.weekday() >= 5:
            return False
        return d in self._trading_days

    def next_trading_day(self, d: date) -> date:
        self._ensure_loaded()
        cur = d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur


# ================================================================
#  retry helper
# ================================================================

def _retry(max_retries=3, base_delay=2):
    def deco(func):
        @wraps(func)
        def wrapper(*a, **kw):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*a, **kw)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"{func.__name__} 第{attempt}次失败: {e}, {delay}s 后重试")
                    time.sleep(delay)
        return wrapper
    return deco


# ================================================================
#  fetcher
# ================================================================

class AShareFetcher:
    def __init__(self, source="akshare", tushare_token=""):
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
    def _ts_code(symbol):
        s = str(symbol).zfill(6)
        return f"{s}.SH" if s.startswith("6") else f"{s}.SZ"

    @staticmethod
    def _standardize(df):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "datetime", "日期时间": "datetime", "date": "datetime", "trade_date": "datetime",
            "开盘": "open", "open": "open",
            "收盘": "close", "close": "close",
            "最高": "high", "high": "high",
            "最低": "low", "low": "low",
            "成交量": "volume", "volume": "volume", "vol": "volume",
            "成交额": "amount", "amount": "amount",
            "换手率": "turnover", "turnover": "turnover", "turnover_rate": "turnover",
        })
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ---- daily ----

    def fetch_daily(self, symbol, start_date, end_date, adjust="qfq"):
        if self.source == "tushare":
            return self._standardize(self._daily_ts(symbol, start_date, end_date, adjust))
        else:
            df = self._daily_ak(symbol, start_date, end_date, adjust)
            if df is None or df.empty:
                logger.warning(f"akshare 失败，尝试 tushare: {symbol}")
                df = self._daily_ts(symbol, start_date, end_date, adjust)
            return self._standardize(df)

    @_retry(3, 3)
    def _daily_ak(self, symbol, start_date, end_date, adjust):
        import akshare as ak
        return ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)

    @_retry(3, 3)
    def _daily_ts(self, symbol, start_date, end_date, adjust):
        ts_code = self._ts_code(symbol)
        sd, ed = start_date.replace("-", ""), end_date.replace("-", "")
        df = self.ts_pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)
        if df is None or df.empty:
            return None
        if adjust == "qfq":
            df_adj = self.ts_pro.adj_factor(ts_code=ts_code, start_date=sd, end_date=ed)
            if df_adj is not None and not df_adj.empty:
                factor = df_adj.set_index("trade_date")["adj_factor"].sort_index()
                df = df.set_index("trade_date").sort_index()
                latest = factor.iloc[-1]
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * factor / latest
                df = df.reset_index()
            else:
                raise RuntimeError(f"adj_factor 获取失败 {symbol}，无法做前复权。")
        df = df.rename(columns={"trade_date": "datetime", "vol": "volume", "pre_close": "pre_close"})
        return df

    # ---- batch daily ----

    @_retry(3, 5)
    def fetch_daily_batch(self, symbols, start_date, end_date, adjust="qfq"):
        """批量拉取多只股票日线。返回 {symbol: DataFrame}。"""
        if self.source != "tushare":
            return {}
        sd, ed = start_date.replace("-", ""), end_date.replace("-", "")
        ts_codes = [self._ts_code(s) for s in symbols]
        result = {}

        for i in range(0, len(ts_codes), 100):
            chunk = ",".join(ts_codes[i:i+100])
            df = self.ts_pro.daily(ts_code=chunk, start_date=sd, end_date=ed)
            if df is None or df.empty:
                continue
            if adjust == "qfq":
                df_adj = self.ts_pro.adj_factor(ts_code=chunk, start_date=sd, end_date=ed)
                if df_adj is not None and not df_adj.empty:
                    factor = df_adj.set_index("trade_date")["adj_factor"].sort_index()
                    df = df.set_index("trade_date").sort_index()
                    latest = factor.iloc[-1]
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * factor / latest
                    df = df.reset_index()
                else:
                    raise RuntimeError(f"adj_factor 获取失败，无法做前复权。")
            for ts, g in df.groupby("ts_code"):
                sym = ts.split(".")[0]
                g = g.drop(columns=["ts_code"]).rename(
                    columns={"trade_date": "datetime", "vol": "volume", "pre_close": "pre_close"})
                result[sym] = self._standardize(g)
        return result

    # ---- batch valuation ----

    @_retry(3, 5)
    def fetch_valuation_batch(self, symbols):
        if self.source != "tushare":
            return {}
        ts_codes = [self._ts_code(s) for s in symbols]
        fields = "trade_date,close,pe_ttm,pb,ps_ttm,dv_ttm,turnover_rate,total_mv,circ_mv"
        result = {}
        for i in range(0, len(ts_codes), 100):
            chunk = ",".join(ts_codes[i:i+100])
            df = self.ts_pro.daily_basic(ts_code=chunk, fields=fields)
            if df is None or df.empty:
                continue
            df = df.rename(columns={"trade_date": "datetime", "pe_ttm": "pe", "ps_ttm": "ps",
                "dv_ttm": "dv_ratio", "total_mv": "market_cap"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            for ts, g in df.groupby("ts_code"):
                sym = ts.split(".")[0]
                result[sym] = g.drop(columns=["ts_code"])
        return result

    # ---- batch financial ----

    @_retry(3, 5)
    def fetch_financial_batch(self, symbols):
        if self.source != "tushare":
            return {}
        ts_codes = [self._ts_code(s) for s in symbols]
        fields = "ts_code,ann_date,end_date,roe,roe_dt,netprofit_margin,grossprofit_margin,or_yoy,netprofit_yoy,ocf_to_netprofit,debt_to_assets,current_ratio"
        result = {}
        for i in range(0, len(ts_codes), 100):
            chunk = ",".join(ts_codes[i:i+100])
            df = self.ts_pro.fina_indicator(ts_code=chunk, fields=fields)
            if df is None or df.empty:
                continue
            df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period",
                "or_yoy": "revenue_yoy", "netprofit_yoy": "profit_yoy"})
            for ts, g in df.groupby("ts_code"):
                sym = ts.split(".")[0]
                result[sym] = g.drop(columns=["ts_code"])
        return result

    # ---- stock list ----

    @_retry(5, 5)
    def fetch_stock_list(self):
        if self.source == "tushare":
            return self._stock_list_ts()
        return self._stock_list_ak()

    def _stock_list_ts(self):
        df = self.ts_pro.stock_basic(exchange="", list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date")
        if df.empty:
            raise RuntimeError("tushare stock_basic 返回空")
        df["symbol"] = df["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
        return df[["symbol", "name", "industry"]]

    def _stock_list_ak(self):
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        return df[["代码", "名称"]].rename(columns={"代码": "symbol", "名称": "name"})

    # ---- index ----

    @_retry(3, 3)
    def fetch_index(self, index_code, start_date, end_date):
        if self.source == "tushare":
            return self._index_ts(index_code, start_date, end_date)
        return self._index_ak(index_code, start_date, end_date)

    def _index_ts(self, index_code, start_date, end_date):
        sd, ed = start_date.replace("-", ""), end_date.replace("-", "")
        prefix = ".SH" if index_code.startswith("0") else ".SZ"
        df = self.ts_pro.index_daily(ts_code=f"{index_code}{prefix}", start_date=sd, end_date=ed)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "datetime", "vol": "volume"})
        return self._standardize(df)

    def _index_ak(self, index_code, start_date, end_date):
        import akshare as ak
        prefix = "sz" if index_code.startswith("3") or index_code.startswith("2") else "sh"
        df = ak.stock_zh_index_daily(symbol=f"{prefix}{index_code}")
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
        return self._standardize(df)

    # ---- valuation ----

    @_retry(3, 2)
    def fetch_valuation(self, symbol):
        if self.source == "tushare":
            return self._val_ts(symbol)
        return self._val_ak(symbol)

    def _val_ts(self, symbol):
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.daily_basic(ts_code=ts_code,
            fields="trade_date,close,pe_ttm,pb,ps_ttm,dv_ttm,turnover_rate,total_mv,circ_mv")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "datetime", "pe_ttm": "pe", "ps_ttm": "ps",
            "dv_ttm": "dv_ratio", "total_mv": "market_cap"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)

    def _val_ak(self, symbol):
        import akshare as ak
        df = ak.stock_a_indicator_lg(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        rename = {}
        for src, dst in [
            ("trade_date", "datetime"), ("date", "datetime"),
            ("pe_ttm", "pe"), ("pe", "pe"), ("pb", "pb"),
            ("ps_ttm", "ps"), ("ps", "ps"),
            ("dv_ttm", "dv_ratio"), ("total_mv", "market_cap"),
            ("turnover_rate", "turnover_rate"),
        ]:
            if src in df.columns:
                rename[src] = dst
        if rename:
            df = df.rename(columns=rename)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
        return df

    # ---- financial ----

    @_retry(3, 2)
    def fetch_financial(self, symbol):
        if self.source == "tushare":
            return self._fin_ts(symbol)
        return self._fin_ak(symbol)

    def _fin_ts(self, symbol):
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.fina_indicator(ts_code=ts_code,
            fields="ts_code,ann_date,end_date,roe,roe_dt,netprofit_margin,grossprofit_margin,or_yoy,netprofit_yoy,ocf_to_netprofit,debt_to_assets,current_ratio")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period",
            "or_yoy": "revenue_yoy", "netprofit_yoy": "profit_yoy"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        df["end_period"] = pd.to_datetime(df["end_period"], format="%Y%m%d", errors="coerce")
        return df

    def _fin_ak(self, symbol):
        import akshare as ak
        try:
            return ak.stock_financial_abstract_ths(symbol=symbol)
        except Exception:
            logger.warning(f"stock_financial_abstract_ths 失败 {symbol}，尝试 fallback")
            return ak.stock_financial_analysis_indicator(symbol=symbol)

    # ---- statements ----

    @_retry(3, 2)
    def fetch_income(self, symbol):
        if self.source == "tushare":
            return self._income_ts(symbol)
        return self._income_ak(symbol)

    def _income_ts(self, symbol):
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.income(ts_code=ts_code,
            fields="ts_code,ann_date,end_date,total_revenue,revenue,oper_cost,total_cogs,operate_profit,n_income_attr_p")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period", "n_income_attr_p": "netprofit"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _income_ak(self, symbol):
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}", symbol="利润表")

    @_retry(3, 2)
    def fetch_balance(self, symbol):
        if self.source == "tushare":
            return self._balance_ts(symbol)
        return self._balance_ak(symbol)

    def _balance_ts(self, symbol):
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.balancesheet(ts_code=ts_code,
            fields="ts_code,ann_date,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,total_cur_assets,total_cur_liab,inventories,accounts_rec")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _balance_ak(self, symbol):
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}", symbol="资产负债表")

    @_retry(3, 2)
    def fetch_cashflow(self, symbol):
        if self.source == "tushare":
            return self._cf_ts(symbol)
        return self._cf_ak(symbol)

    def _cf_ts(self, symbol):
        ts_code = self._ts_code(symbol)
        df = self.ts_pro.cashflow(ts_code=ts_code,
            fields="ts_code,ann_date,end_date,n_cashflow_act,n_cashflow_inv_act,n_cashflow_fnc_act")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"ann_date": "report_date", "end_date": "end_period"})
        df["report_date"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        return df

    def _cf_ak(self, symbol):
        import akshare as ak
        return ak.stock_financial_report_sina(
            stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}", symbol="现金流量表")


# ================================================================
#  cache
# ================================================================

class DataCache:
    def __init__(self, cache_dir="outputs/data_cache"):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def load_daily(self, symbol):
        p = self.dir / f"daily_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_daily(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"daily_{symbol}.parquet", index=False)

    def append_daily(self, symbol, new_df):
        existing = self.load_daily(symbol)
        if existing.empty:
            merged = new_df
        else:
            merged = pd.concat([existing, new_df], ignore_index=True)
            if "datetime" in merged.columns:
                merged = merged.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
        self.save_daily(symbol, merged)
        return merged

    def get_last_date(self, symbol) -> Optional[str]:
        df = self.load_daily(symbol)
        if df.empty or "datetime" not in df.columns:
            return None
        dts = pd.to_datetime(df["datetime"]).dropna()
        if dts.empty:
            return None
        return dts.max().strftime("%Y%m%d")

    def load_valuation(self, symbol):
        p = self.dir / f"valuation_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_valuation(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"valuation_{symbol}.parquet", index=False)

    def load_financial(self, symbol):
        p = self.dir / f"financial_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_financial(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"financial_{symbol}.parquet", index=False)

    def load_income(self, symbol):
        p = self.dir / f"income_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_income(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"income_{symbol}.parquet", index=False)

    def load_balance(self, symbol):
        p = self.dir / f"balance_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_balance(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"balance_{symbol}.parquet", index=False)

    def load_cashflow(self, symbol):
        p = self.dir / f"cashflow_{symbol}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_cashflow(self, symbol, df):
        if not df.empty:
            df.to_parquet(self.dir / f"cashflow_{symbol}.parquet", index=False)

    def load_index(self, index_code):
        p = self.dir / f"index_{index_code}.parquet"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    def save_index(self, index_code, df):
        if not df.empty:
            df.to_parquet(self.dir / f"index_{index_code}.parquet", index=False)

    def list_cached(self):
        rows = []
        for path in sorted(self.dir.glob("*.parquet")):
            parts = path.stem.split("_", 1)
            dtype = parts[0] if parts else ""
            sym = parts[1] if len(parts) > 1 else ""
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = path.stat().st_size / 1024
            try:
                meta = pq.read_metadata(path)
                n_rows = meta.num_rows
                last_date = ""
                if "datetime" in meta.schema.names:
                    df_head = pd.read_parquet(path, columns=["datetime"])
                    if not df_head.empty:
                        last_date = pd.to_datetime(df_head["datetime"]).max().strftime("%Y-%m-%d")
            except Exception:
                n_rows = 0
                last_date = ""
            rows.append({"type": dtype, "symbol": sym, "rows": n_rows,
                "last_date": last_date, "size_kb": f"{size_kb:.1f}", "updated": mtime})
        return pd.DataFrame(rows)

    def print_summary(self):
        d = self.dir
        for label, pattern in [("daily", "daily_*"), ("valuation", "valuation_*"),
            ("financial", "financial_*"), ("income", "income_*"),
            ("balance", "balance_*"), ("cashflow", "cashflow_*"), ("index", "index_*")]:
            logger.info(f"  {label}: {len(list(d.glob(pattern+'.parquet')))}")


# ================================================================
#  stock list
# ================================================================

def get_all_symbols(fetcher, cache):
    csv_path = cache.dir / "_stock_list.csv"
    if csv_path.exists():
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 7:
            df = pd.read_csv(csv_path)
            return [str(s).zfill(6) for s in df["symbol"].tolist()]

    df = fetcher.fetch_stock_list()
    if df.empty:
        raise RuntimeError("获取股票列表失败")
    codes = df["symbol"].tolist()
    valid = [str(c).zfill(6) for c in codes if str(c).zfill(6)[:2] in ("60", "00", "30", "68")]
    valid = sorted(set(valid))

    save = pd.DataFrame({"symbol": valid})
    if "industry" in df.columns:
        df["_s"] = df["symbol"].apply(lambda s: str(s).zfill(6))
        ind = df.set_index("_s")["industry"].to_dict()
        save["industry"] = save["symbol"].map(lambda s: ind.get(s, ""))
        pd.DataFrame(list(ind.items()), columns=["symbol", "industry"]).to_csv(
            cache.dir / "_industry_map.csv", index=False)
    save.to_csv(csv_path, index=False)
    return valid


def _report(name, failed, total):
    if failed:
        p = Path("outputs/logs") / f"failed_{name.lower()}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(failed))
        logger.warning(f"{name} 失败: {len(failed)}/{total}")
    logger.info(f"{name} 完成: {total - len(failed)}/{total}")


# ================================================================
#  download
# ================================================================

def cmd_download(args, cfg):
    fetcher = AShareFetcher(cfg.get("source", "akshare"), cfg.get("tushare_token", ""))
    cache = DataCache(cfg.get("cache_dir", "outputs/data_cache"))
    rate = cfg.get("rate_limit", 0.35)
    cool = cfg.get("error_cooldown", 1.0)
    adj = cfg.get("adjust", "qfq")

    start = args.start or cfg.get("start_date", "20150101")
    end = args.end or datetime.now().strftime("%Y%m%d")

    if args.all:
        symbols = get_all_symbols(fetcher, cache)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = cfg.get("symbols", ["000001"])
    logger.info(f"下载: {len(symbols)} 只股票, {start} ~ {end}")
    cache.print_summary()

    def _iter(label, fetch_fn, load_fn, save_fn):
        failed = []
        for i, s in enumerate(symbols):
            if args.skip_cached and not load_fn(s).empty:
                continue
            if (i + 1) % 50 == 1:
                logger.info(f"[{i+1}/{len(symbols)}] {label}: ...")
            try:
                df = fetch_fn(s)
                if df.empty:
                    failed.append(s); continue
                save_fn(s, df)
                logger.info(f"  {s}: {len(df)} rows")
                time.sleep(rate)
            except Exception as e:
                logger.error(f"  {s}: {e}")
                failed.append(s)
                time.sleep(cool)
        _report(label, failed, len(symbols))

    if not args.no_market:
        failed = []
        for i, s in enumerate(symbols):
            if args.skip_cached and cache.get_last_date(s):
                continue
            logger.info(f"[{i+1}/{len(symbols)}] Market: {s}")
            try:
                df = fetcher.fetch_daily(s, start, end, adjust=adj)
                if df.empty:
                    failed.append(s); continue
                cache.save_daily(s, df)
                logger.info(f"  {s}: {len(df)} rows")
                time.sleep(rate)
            except Exception as e:
                logger.error(f"  {s}: {e}")
                failed.append(s)
                time.sleep(cool)
        _report("Market", failed, len(symbols))

    if not args.no_valuation:
        _iter("Valuation", fetcher.fetch_valuation, cache.load_valuation, cache.save_valuation)
    if not args.no_fundamental:
        _iter("Financial", fetcher.fetch_financial, cache.load_financial, cache.save_financial)
    if not args.no_statements:
        for label, ffn, lfn, sfn in [
            ("Income", fetcher.fetch_income, cache.load_income, cache.save_income),
            ("Balance", fetcher.fetch_balance, cache.load_balance, cache.save_balance),
            ("Cashflow", fetcher.fetch_cashflow, cache.load_cashflow, cache.save_cashflow),
        ]:
            _iter(label, ffn, lfn, sfn)

    if not args.no_index:
        for idx in cfg.get("indexes", ["000300", "000905"]):
            idx = str(idx)
            e = cache.load_index(idx)
            if not e.empty:
                logger.info(f"Index {idx} 已有缓存 ({len(e)} rows)，跳过"); continue
            logger.info(f"下载指数 {idx}")
            try:
                df = fetcher.fetch_index(idx, start, end)
                if df.empty:
                    logger.warning(f"Index {idx}: 无数据"); continue
                cache.save_index(idx, df)
                logger.info(f"Index {idx}: {len(df)} rows")
            except Exception as ex:
                logger.error(f"Index {idx}: {ex}")

    logger.info("下载完成。")


# ================================================================
#  update
# ================================================================

def cmd_update(args, cfg):
    cal = AShareCalendar()
    fetcher = AShareFetcher(cfg.get("source", "akshare"), cfg.get("tushare_token", ""))
    cache = DataCache(cfg.get("cache_dir", "outputs/data_cache"))
    rate = cfg.get("rate_limit", 0.35)
    adj = cfg.get("adjust", "qfq")
    today = datetime.now().strftime("%Y%m%d")

    if args.all:
        symbols = get_all_symbols(fetcher, cache)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = cfg.get("symbols", ["000001"])

    default_start = args.default_start or cfg.get("start_date", "20150101")
    logger.info(f"更新: {len(symbols)} 只股票, force={args.force}")

    # ---- market (batch by date range) ----
    if not args.no_market:
        # group symbols by (start_date, end_date)
        groups = {}
        skipped = 0
        logger.info("  正在检查缓存日期...")
        for i, s in enumerate(symbols):
            if (i + 1) % 1000 == 1:
                logger.info(f"  [{i+1}/{len(symbols)}] ...")
            last = cache.get_last_date(s) if not args.force else None
            if last:
                start = cal.next_trading_day(datetime.strptime(last, "%Y%m%d").date()).strftime("%Y%m%d")
                if start > today:
                    skipped += 1; continue
            else:
                start = default_start
            groups.setdefault((start, today), []).append(s)

        logger.info(f"  行情: {skipped} 只已是最新，{sum(len(v) for v in groups.values())} 只需要更新")
        for (start, end), batch in groups.items():
            logger.info(f"  批次 {start}~{end}: {len(batch)} 只")
            results = fetcher.fetch_daily_batch(batch, start, end, adjust=adj)
            for s in batch:
                df = results.get(s, pd.DataFrame())
                if df.empty:
                    logger.warning(f"  {s}: 无新数据"); continue
                merged = cache.append_daily(s, df)
            logger.info(f"  批次 {start}~{end}: {len(results)} 只有数据")
            time.sleep(rate)

    # ---- valuation (batch) ----
    if not args.no_valuation and symbols:
        logger.info(f"  估值: {len(symbols)} 只，批量拉取...")
        results = fetcher.fetch_valuation_batch(symbols)
        for s in symbols:
            df = results.get(s, pd.DataFrame())
            if not df.empty:
                cache.save_valuation(s, df)
        logger.info(f"  估值: {len(results)} 只有数据")
        time.sleep(rate)

    # ---- financial (batch) ----
    if not args.no_fundamental and symbols:
        logger.info(f"  财报: {len(symbols)} 只，批量拉取...")
        results = fetcher.fetch_financial_batch(symbols)
        for s in symbols:
            df = results.get(s, pd.DataFrame())
            if not df.empty:
                cache.save_financial(s, df)
        logger.info(f"  财报: {len(results)} 只有数据")

    logger.info("更新完成。")


# ================================================================
#  main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="A 股数据管理")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("download", help="全量下载")
    p.add_argument("--all", action="store_true")
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-market", action="store_true")
    p.add_argument("--no-valuation", action="store_true")
    p.add_argument("--no-fundamental", action="store_true")
    p.add_argument("--no-statements", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--skip-cached", action="store_true")

    p = sub.add_parser("update", help="增量更新")
    p.add_argument("--all", action="store_true")
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--default-start", default=None)
    p.add_argument("--no-market", action="store_true")
    p.add_argument("--no-valuation", action="store_true")
    p.add_argument("--no-fundamental", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("status", help="查看缓存状态")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    setup_logger("manage_data")
    cfg = load_config()
    data_cfg = cfg.get("data", {})

    if args.cmd == "status":
        cache = DataCache(data_cfg.get("cache_dir", "outputs/data_cache"))
        summary = cache.list_cached()
        if summary.empty:
            print("没有缓存数据。")
        else:
            print(summary.to_string(index=False))
    elif args.cmd == "download":
        cmd_download(args, data_cfg)
    elif args.cmd == "update":
        cmd_update(args, data_cfg)


if __name__ == "__main__":
    main()

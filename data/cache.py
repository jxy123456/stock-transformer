from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class DataCache:
    """本地 Parquet 缓存，按 symbol 存单文件，支持增量追加。"""

    def __init__(self, cache_dir: str = "outputs/data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 行情缓存 (单文件 per symbol) ----

    def _daily_path(self, symbol: str) -> Path:
        return self.cache_dir / f"daily_{symbol}.parquet"

    def _valuation_path(self, symbol: str) -> Path:
        return self.cache_dir / f"valuation_{symbol}.parquet"

    def _financial_path(self, symbol: str) -> Path:
        return self.cache_dir / f"financial_{symbol}.parquet"

    def load_daily(self, symbol: str) -> pd.DataFrame:
        """加载该 symbol 的全部本地行情数据。"""
        path = self._daily_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except Exception as e:
            logger.warning(f"Cache read failed for {symbol}: {e}")
            return pd.DataFrame()

    def save_daily(self, symbol: str, df: pd.DataFrame):
        """覆盖写入行情数据。"""
        if df.empty:
            return
        path = self._daily_path(symbol)
        try:
            df.to_parquet(path, index=False)
            logger.debug(f"Saved {symbol}: {len(df)} rows -> {path}")
        except Exception as e:
            logger.warning(f"Cache write failed for {symbol}: {e}")

    def append_daily(self, symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """将新数据追加到本地缓存，按 datetime 去重，返回合并后的完整数据。"""
        existing = self.load_daily(symbol)
        if existing.empty:
            merged = new_df
        else:
            merged = pd.concat([existing, new_df], ignore_index=True)
            if "datetime" in merged.columns:
                merged = merged.drop_duplicates(subset=["datetime"], keep="last")
                merged = merged.sort_values("datetime").reset_index(drop=True)

        self.save_daily(symbol, merged)
        return merged

    def get_last_date(self, symbol: str) -> Optional[str]:
        """返回本地缓存中最后一条数据的日期 (YYYYMMDD)，无数据返回 None。"""
        df = self.load_daily(symbol)
        if df.empty or "datetime" not in df.columns:
            return None
        last = pd.to_datetime(df["datetime"]).max()
        return last.strftime("%Y%m%d")

    # ---- 估值指标缓存 ----

    def load_valuation(self, symbol: str) -> pd.DataFrame:
        path = self._valuation_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    def save_valuation(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        try:
            df.to_parquet(self._valuation_path(symbol), index=False)
        except Exception as e:
            logger.warning(f"Valuation cache write failed for {symbol}: {e}")

    # ---- 财报缓存 ----

    def load_financial(self, symbol: str) -> pd.DataFrame:
        path = self._financial_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    def save_financial(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        try:
            df.to_parquet(self._financial_path(symbol), index=False)
        except Exception as e:
            logger.warning(f"Financial cache write failed for {symbol}: {e}")

    # ---- 管理操作 ----

    def invalidate(self, symbol: str):
        for path in self.cache_dir.glob(f"*_{symbol}.parquet"):
            path.unlink()
            logger.debug(f"Invalidated: {path}")

    def list_cached(self) -> pd.DataFrame:
        """列出所有本地缓存文件及信息。"""
        rows = []
        for path in sorted(self.cache_dir.glob("*.parquet")):
            name = path.stem
            parts = name.split("_", 1)
            dtype = parts[0] if parts else ""
            sym = parts[1] if len(parts) > 1 else ""
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = path.stat().st_size / 1024
            try:
                df = pd.read_parquet(path)
                n_rows = len(df)
                last_date = ""
                if "datetime" in df.columns:
                    last_date = pd.to_datetime(df["datetime"]).max().strftime("%Y-%m-%d")
            except Exception:
                n_rows = 0
                last_date = ""
            rows.append({
                "type": dtype, "symbol": sym, "rows": n_rows,
                "last_date": last_date, "size_kb": f"{size_kb:.1f}", "updated": mtime,
            })
        return pd.DataFrame(rows)

    def clear(self):
        for path in self.cache_dir.glob("*.parquet"):
            path.unlink()

"""选股：流动性 + 上市年限筛选。"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

STOCK_LISTS_DIR = Path(__file__).parent.parent / "config" / "stock_lists"


def _load_daily(cache_dir: str, symbol: str) -> pd.DataFrame:
    p = Path(cache_dir) / f"daily_{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_list_dates(token: str) -> dict:
    """从 tushare 获取所有股票的上市日期和名称。返回 {symbol: {name, list_date}}。"""
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,symbol,name,list_date")
    result = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).zfill(6)
        result[sym] = {
            "name": row["name"],
            "list_date": row["list_date"],
        }
    return result


def select_liquid_top100(token: str = None, n: int = 100, min_years: int = 5,
                         max_price: float = 50, cache_dir: str = "outputs/data_cache"):
    """流动性前 N 只股票，上市满 min_years 年，排除 ST，股价 <= max_price。

    流动性 = 近 60 日日均成交额。
    max_price 用于过滤高价股（2 万本金，50 元/股 = 5000 一手，占 25%）。
    结果保存到 config/stock_lists/liquid{N}.yaml。
    """
    if token is None:
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "default.yaml"))
        token = cfg.get("data", {}).get("tushare_token", "")

    # 1. 获取上市日期和名称
    logger.info("获取股票列表及上市日期...")
    info = fetch_list_dates(token)
    logger.info(f"  共 {len(info)} 只股票")

    # 2. 筛选：排除上市不足 min_years 年，排除 ST
    cutoff = (datetime.now() - timedelta(days=min_years * 365)).strftime("%Y%m%d")
    candidates = {}
    for sym, v in info.items():
        if v["list_date"] > cutoff:
            continue
        if "ST" in v["name"]:
            continue
        candidates[sym] = v["name"]

    logger.info(f"  上市>{min_years}年 + 非ST: {len(candidates)} 只")

    # 3. 计算近 60 日日均成交额 + 最新收盘价
    logger.info("计算近 60 日日均成交额及最新价格...")
    scores = {}
    for i, sym in enumerate(candidates):
        if (i + 1) % 500 == 1:
            logger.info(f"  [{i+1}/{len(candidates)}] ...")
        df = _load_daily(cache_dir, sym)
        if df.empty or "amount" not in df.columns or "close" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        recent = df.sort_values("datetime").tail(60)
        if len(recent) < 30:
            continue
        avg_amount = float(recent["amount"].mean())
        last_close = float(recent["close"].iloc[-1])
        if avg_amount > 0 and last_close <= max_price:
            scores[sym] = (avg_amount, last_close)

    # 4. 排名取前 N
    ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    top = ranked[:n]

    logger.info(f"  有效数据（股价<={max_price}）: {len(scores)} 只, 取前 {n}")
    if not top:
        raise RuntimeError(f"没有符合条件的股票，请调高 max_price（当前 {max_price}）")

    # 5. 保存
    STOCK_LISTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "name": f"liquid{n}",
        "description": f"流动性前{n}（近60日日均成交额），上市满{min_years}年，排除ST，股价<={max_price}",
        "generated_at": datetime.now().isoformat(),
        "symbols": [s for s, _ in top],
        "details": [{"symbol": s, "name": candidates.get(s, ""),
                      "avg_amount_60d": round(v[0], 2),
                      "price": round(v[1], 2)} for s, v in top],
    }
    path = STOCK_LISTS_DIR / f"liquid{n}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False)

    # 打印摘要
    print(f"\n=== 前 10 只（股价 <= {max_price}）===")
    for s, (amt, price) in top[:10]:
        print(f"  {s}  {candidates.get(s, ''):10s}  日均成交额: {amt:,.0f}  股价: {price:.2f}")
    print(f"\n已保存: {path}")
    return [s for s, _ in top]


def load_symbols(name: str) -> list:
    """从 config/stock_lists/{name}.yaml 加载股票列表。"""
    path = STOCK_LISTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Stock list not found: {path}")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return cfg["symbols"]


if __name__ == "__main__":
    from loguru import logger
    select_liquid_top100()

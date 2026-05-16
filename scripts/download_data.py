"""
全量下载历史数据脚本。

用法:
  # 下载全部数据（行情+估值+财报+财报明细+指数+行业）
  python scripts/download_data.py --all --start 20150101

  # 只补下载缺失的数据
  python scripts/download_data.py --all --start 20150101 --skip-cached

  # 只下载行情，跳过其他
  python scripts/download_data.py --all --start 20150101 --no-fundamental --no-valuation --no-statements --no-index

  # 查看本地缓存状态
  python scripts/download_data.py --list-cache
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from data.fetcher import AShareFetcher
from data.cache import DataCache


def get_all_symbols(fetcher: AShareFetcher, cache: DataCache, logger, max_age_days: int = 7) -> list:
    """获取全 A 股股票代码列表，本地缓存 7 天。"""
    list_path = Path(cache.cache_dir) / "_stock_list.csv"

    if list_path.exists():
        mtime = datetime.fromtimestamp(list_path.stat().st_mtime)
        if (datetime.now() - mtime).days < max_age_days:
            df_local = pd.read_csv(list_path)
            symbols = [str(s).zfill(6) for s in df_local["symbol"].tolist()]
            logger.info(f"Stock list from cache ({len(symbols)} stocks, cached {mtime.date()})")
            return symbols

    df = fetcher.fetch_stock_list()
    if df.empty:
        raise RuntimeError("Failed to fetch stock list")

    codes = df["symbol"].tolist()

    valid = [str(c).zfill(6) for c in codes if str(c).zfill(6)[:2] in ("60", "00", "30", "68") or str(c).zfill(6)[:3] in ("688")]
    valid = sorted(set(valid))

    # Save with industry column if available
    save_df = pd.DataFrame({"symbol": valid})
    if "industry" in df.columns:
        industry_map = df.set_index("symbol")["industry"].to_dict()
        save_df["industry"] = save_df["symbol"].map(lambda s: industry_map.get(s, ""))
    save_df.to_csv(list_path, index=False)

    # Also save industry map separately
    if "industry" in df.columns:
        ind_path = Path(cache.cache_dir) / "_industry_map.csv"
        ind_df = df[["symbol", "industry"]].copy()
        ind_df["symbol"] = ind_df["symbol"].apply(lambda s: str(s).zfill(6))
        ind_df.to_csv(ind_path, index=False)

    logger.info(f"Stock list fetched: {len(codes)} total, {len(valid)} after filter")
    return valid


def download_market_data(symbols, start_date, end_date, fetcher, cache, adjust, skip_cached, logger):
    failed = []
    for i, symbol in enumerate(symbols):
        if skip_cached and cache.get_last_date(symbol):
            continue
        logger.info(f"[{i+1}/{len(symbols)}] Market: {symbol}")
        try:
            df = fetcher.fetch_daily(symbol, start_date, end_date, adjust=adjust)
            if df.empty:
                failed.append(symbol)
                continue
            cache.save_daily(symbol, df)
            logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)
    _report("Market", failed, len(symbols), logger)


def download_valuation_data(symbols, fetcher, cache, skip_cached, logger):
    failed = []
    for i, symbol in enumerate(symbols):
        if skip_cached and not cache.load_valuation(symbol).empty:
            continue
        logger.info(f"[{i+1}/{len(symbols)}] Valuation: {symbol}")
        try:
            df = fetcher.fetch_valuation_metrics(symbol)
            if df.empty:
                failed.append(symbol)
                continue
            cache.save_valuation(symbol, df)
            logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)
    _report("Valuation", failed, len(symbols), logger)


def download_financial_data(symbols, fetcher, cache, skip_cached, logger):
    failed = []
    for i, symbol in enumerate(symbols):
        if skip_cached and not cache.load_financial(symbol).empty:
            continue
        logger.info(f"[{i+1}/{len(symbols)}] Financial: {symbol}")
        try:
            df = fetcher.fetch_financial_summary(symbol)
            if df.empty:
                failed.append(symbol)
                continue
            cache.save_financial(symbol, df)
            logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)
    _report("Financial", failed, len(symbols), logger)


def download_statements(symbols, fetcher, cache, skip_cached, logger):
    """Download income/balance/cashflow statements for all symbols."""
    for stmt_name, fetch_fn, load_fn, save_fn in [
        ("Income", fetcher.fetch_income_statement, cache.load_income, cache.save_income),
        ("Balance", fetcher.fetch_balance_sheet, cache.load_balance, cache.save_balance),
        ("Cashflow", fetcher.fetch_cashflow_statement, cache.load_cashflow, cache.save_cashflow),
    ]:
        failed = []
        for i, symbol in enumerate(symbols):
            if skip_cached and not load_fn(symbol).empty:
                continue
            if (i + 1) % 50 == 1:
                logger.info(f"[{i+1}/{len(symbols)}] {stmt_name}: ...")
            try:
                df = fetch_fn(symbol)
                if df.empty:
                    failed.append(symbol)
                    continue
                save_fn(symbol, df)
                time.sleep(0.2)
            except Exception as e:
                failed.append(symbol)
                time.sleep(1)
        _report(stmt_name, failed, len(symbols), logger)


def download_index_data(start_date, end_date, fetcher, cache, logger):
    """Download index daily data for CSI 300 and CSI 500."""
    for idx_code, idx_name in [("000300", "CSI300"), ("000905", "CSI500")]:
        existing = cache.load_index(idx_code)
        if not existing.empty:
            logger.info(f"Index {idx_name} already cached ({len(existing)} rows), skipping")
            continue
        logger.info(f"Downloading index {idx_name} ({idx_code})")
        try:
            df = fetcher.fetch_index(idx_code, start_date, end_date)
            if df.empty:
                logger.warning(f"Index {idx_name}: no data")
                continue
            cache.save_index(idx_code, df)
            logger.info(f"Index {idx_name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Index {idx_name}: {e}")


def _report(name, failed, total, logger):
    if failed:
        path = Path("outputs/logs") / f"failed_{name.lower()}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(failed))
        logger.warning(f"{name} failed: {len(failed)}/{total}")
    logger.info(f"{name} done: {total - len(failed)}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Download A-share historical data")
    parser.add_argument("--all", action="store_true", help="Download ALL A-share stocks")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific stock symbols")
    parser.add_argument("--start", type=str, default="20150101", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")
    parser.add_argument("--no-market", action="store_true", help="Skip daily market data")
    parser.add_argument("--no-fundamental", action="store_true", help="Skip fina_indicator")
    parser.add_argument("--no-valuation", action="store_true", help="Skip daily_basic/valuation")
    parser.add_argument("--no-statements", action="store_true", help="Skip income/balance/cashflow")
    parser.add_argument("--no-index", action="store_true", help="Skip index data")
    parser.add_argument("--skip-cached", action="store_true", help="Skip stocks already in cache")
    parser.add_argument("--list-cache", action="store_true", help="List cached data and exit")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("download_data")

    end_date = args.end or datetime.now().strftime("%Y%m%d")
    adjust = config.get("data.adjust", "qfq")

    fetcher = AShareFetcher(
        source=config.get("data.source", "tushare"),
        tushare_token=config.get("data.tushare_token", ""),
    )
    cache = DataCache(cache_dir=config.get("data.cache_dir", "outputs/data_cache"))

    if args.list_cache:
        summary = cache.list_cached()
        if summary.empty:
            print("No cached data.")
        else:
            print(summary.to_string(index=False))
        return

    if args.all:
        symbols = get_all_symbols(fetcher, cache, logger)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get("data.symbols", ["000001"])

    logger.info(f"Download: {len(symbols)} stocks, {args.start} ~ {end_date}")

    if not args.no_market:
        download_market_data(symbols, args.start, end_date, fetcher, cache, adjust, args.skip_cached, logger)

    if not args.no_valuation:
        download_valuation_data(symbols, fetcher, cache, args.skip_cached, logger)

    if not args.no_fundamental:
        download_financial_data(symbols, fetcher, cache, args.skip_cached, logger)

    if not args.no_statements:
        download_statements(symbols, fetcher, cache, args.skip_cached, logger)

    if not args.no_index:
        download_index_data(args.start, end_date, fetcher, cache, logger)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()

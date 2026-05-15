"""
全量下载历史数据脚本。

用法:
  # 下载全 A 股行情 + 财报
  python scripts/download_data.py --all --start 20150101

  # 下载指定股票
  python scripts/download_data.py --symbols 000001 600519 --start 20150101

  # 只下载行情，跳过财报
  python scripts/download_data.py --all --no-fundamental

  # 跳过已有缓存的股票（增量场景）
  python scripts/download_data.py --all --skip-cached

  # 查看本地缓存状态
  python scripts/download_data.py --list-cache
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from data.fetcher import AShareFetcher
from data.cache import DataCache
from data.features import FeatureEngine
from data.fundamental import FundamentalEngine


def get_all_symbols(fetcher: AShareFetcher, logger) -> list:
    """获取全 A 股股票代码列表，过滤掉北交所和退市股。"""
    df = fetcher.fetch_stock_list()
    if df.empty:
        raise RuntimeError("Failed to fetch stock list from akshare")

    codes = df["symbol"].tolist()

    # 过滤: 只保留沪深主板 + 创业板 + 科创板
    # 60xxxx 沪主板, 00xxxx 深主板, 300xxx 创业板, 688xxx 科创板
    # 排除 8xxxxx 北交所, 4xxxxx 新三板
    valid = [c for c in codes if c[:2] in ("60", "00", "30", "68") or c[:3] in ("688")]

    logger.info(f"Total A-share stocks: {len(codes)}, after filter: {len(valid)}")
    return sorted(valid)


def download_market_data(
    symbols: list,
    start_date: str,
    end_date: str,
    fetcher: AShareFetcher,
    cache: DataCache,
    adjust: str,
    skip_cached: bool,
    logger,
):
    failed = []
    for i, symbol in enumerate(symbols):
        # 跳过已有缓存
        if skip_cached and cache.get_last_date(symbol):
            continue

        logger.info(f"[{i+1}/{len(symbols)}] Market: {symbol}")
        try:
            df = fetcher.fetch_daily(symbol, start_date, end_date, adjust=adjust)
            if df.empty:
                logger.warning(f"  {symbol}: no data returned")
                failed.append(symbol)
                continue

            cache.save_daily(symbol, df)
            logger.info(f"  {symbol}: {len(df)} rows, {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)

    if failed:
        logger.warning(f"Market data failed: {len(failed)} stocks")
        _save_failed(failed, "market", logger)
    logger.info(f"Market data done: {len(symbols) - len(failed)}/{len(symbols)}")


def download_valuation_data(
    symbols: list,
    fetcher: AShareFetcher,
    cache: DataCache,
    skip_cached: bool,
    logger,
):
    failed = []
    for i, symbol in enumerate(symbols):
        if skip_cached:
            existing = cache.load_valuation(symbol)
            if not existing.empty:
                continue

        logger.info(f"[{i+1}/{len(symbols)}] Valuation: {symbol}")
        try:
            df = fetcher.fetch_valuation_metrics(symbol)
            if df.empty:
                failed.append(symbol)
                continue
            cache.save_valuation(symbol, df)
            logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)

    if failed:
        logger.warning(f"Valuation failed: {len(failed)} stocks")
        _save_failed(failed, "valuation", logger)
    logger.info(f"Valuation done: {len(symbols) - len(failed)}/{len(symbols)}")


def download_financial_data(
    symbols: list,
    fetcher: AShareFetcher,
    cache: DataCache,
    skip_cached: bool,
    logger,
):
    failed = []
    for i, symbol in enumerate(symbols):
        if skip_cached:
            existing = cache.load_financial(symbol)
            if not existing.empty:
                continue

        logger.info(f"[{i+1}/{len(symbols)}] Financial: {symbol}")
        try:
            df = fetcher.fetch_financial_summary(symbol)
            if not df.empty:
                cache.save_financial(symbol, df)
                logger.info(f"  {symbol}: {len(df)} rows")
            else:
                failed.append(symbol)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")
            failed.append(symbol)
            time.sleep(1)

    if failed:
        logger.warning(f"Financial failed: {len(failed)} stocks")
        _save_failed(failed, "financial", logger)
    logger.info(f"Financial done: {len(symbols) - len(failed)}/{len(symbols)}")


def compute_and_save_features(
    symbols: list,
    cache: DataCache,
    feature_engine: FeatureEngine,
    fundamental_engine: FundamentalEngine,
    logger,
):
    import pandas as pd
    output_dir = Path("outputs/data_cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(symbols):
        if (i + 1) % 100 == 0:
            logger.info(f"Features: {i+1}/{len(symbols)}")

        df = cache.load_daily(symbol)
        if df.empty:
            continue

        df = feature_engine.compute(df)

        df_val = cache.load_valuation(symbol)
        if not df_val.empty:
            fund_feats = fundamental_engine.compute_features(df_val)
            if "datetime" in df.columns and "trade_date" in fund_feats.columns:
                fund_feats["datetime"] = pd.to_datetime(fund_feats["trade_date"])
                fund_feats = fund_feats.set_index("datetime")
                df = df.set_index("datetime")
                for col in fundamental_engine.feature_columns:
                    if col in fund_feats.columns:
                        df[col] = fund_feats[col]
                df = df.reset_index()

        df = df.dropna(subset=["close"])
        df.to_csv(output_dir / f"{symbol}_features.csv", index=False)

    logger.info(f"Features computed for {len(symbols)} symbols")


def _save_failed(failed: list, data_type: str, logger):
    path = Path("outputs/logs") / f"failed_{data_type}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(failed))
    logger.info(f"Failed list saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Download A-share historical data")
    parser.add_argument("--all", action="store_true", help="Download ALL A-share stocks")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific stock symbols")
    parser.add_argument("--start", type=str, default="20150101", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")
    parser.add_argument("--no-market", action="store_true", help="Skip market data")
    parser.add_argument("--no-fundamental", action="store_true", help="Skip fundamental data")
    parser.add_argument("--no-features", action="store_true", help="Skip feature computation")
    parser.add_argument("--skip-cached", action="store_true", help="Skip stocks already in cache")
    parser.add_argument("--list-cache", action="store_true", help="List cached data and exit")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("download_data")

    end_date = args.end or datetime.now().strftime("%Y%m%d")
    adjust = config.get("data.adjust", "qfq")

    fetcher = AShareFetcher(
        source=config.get("data.source", "akshare"),
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

    # 确定股票列表
    if args.all:
        symbols = get_all_symbols(fetcher, logger)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get("data.symbols", ["000001"])

    logger.info(f"Download: {len(symbols)} stocks, {args.start} ~ {end_date}")

    if not args.no_market:
        download_market_data(symbols, args.start, end_date, fetcher, cache, adjust, args.skip_cached, logger)

    if not args.no_fundamental:
        download_valuation_data(symbols, fetcher, cache, args.skip_cached, logger)
        download_financial_data(symbols, fetcher, cache, args.skip_cached, logger)

    if not args.no_features:
        feature_engine = FeatureEngine(config.raw)
        fundamental_engine = FundamentalEngine(fetcher, cache)
        compute_and_save_features(symbols, cache, feature_engine, fundamental_engine, logger)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()

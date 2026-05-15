"""
增量更新脚本：只拉取本地缓存最后日期之后的新数据。

用法:
  # 更新配置文件中所有股票
  python scripts/update_data.py

  # 更新指定股票
  python scripts/update_data.py --symbols 000001 600519

  # 只更新行情，不更新财报
  python scripts/update_data.py --no-fundamental

  # 强制全量更新（忽略本地缓存）
  python scripts/update_data.py --force
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from data.fetcher import AShareFetcher
from data.cache import DataCache
from data.features import FeatureEngine
from data.fundamental import FundamentalEngine


def update_market_data(
    symbols: list,
    fetcher: AShareFetcher,
    cache: DataCache,
    adjust: str,
    force: bool,
    default_start: str,
    logger,
):
    """增量更新行情数据：只拉取 last_date+1 到今天的新数据。"""
    today = datetime.now().strftime("%Y%m%d")
    updated_symbols = []

    for i, symbol in enumerate(symbols):
        last_date = cache.get_last_date(symbol) if not force else None

        if last_date:
            # 从缓存最后日期的下一天开始
            start = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            if start > today:
                logger.info(f"[{i+1}/{len(symbols)}] {symbol}: already up-to-date ({last_date})")
                continue
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}: incremental update {start} ~ {today}")
        else:
            start = default_start
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}: full download {start} ~ {today}")

        try:
            df_new = fetcher.fetch_daily(symbol, start, today, adjust=adjust)
            if df_new.empty:
                logger.info(f"  {symbol}: no new data")
                continue

            # 追加到本地缓存
            merged = cache.append_daily(symbol, df_new)
            new_rows = len(df_new)
            total_rows = len(merged)
            logger.info(f"  {symbol}: +{new_rows} new rows (total {total_rows})")
            updated_symbols.append(symbol)

            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")

    return updated_symbols


def update_valuation_data(
    symbols: list,
    fetcher: AShareFetcher,
    cache: DataCache,
    logger,
):
    """估值指标全量覆盖（日频数据量不大，直接覆盖更简单可靠）。"""
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] Updating valuation: {symbol}")
        try:
            df = fetcher.fetch_valuation_metrics(symbol)
            if not df.empty:
                cache.save_valuation(symbol, df)
                logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")


def update_financial_data(
    symbols: list,
    fetcher: AShareFetcher,
    cache: DataCache,
    logger,
):
    """财报数据全量覆盖（季频，数据量极小）。"""
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] Updating financial reports: {symbol}")
        try:
            # 财报摘要
            df = fetcher.fetch_financial_summary(symbol)
            if not df.empty:
                cache.save_financial(symbol, df)
                logger.info(f"  {symbol}: {len(df)} rows")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {symbol}: {e}")


def recompute_features(
    symbols: list,
    cache: DataCache,
    config: dict,
    logger,
):
    """对更新后的数据重新计算特征。"""
    import pandas as pd
    feature_engine = FeatureEngine(config)
    fundamental_engine = FundamentalEngine()
    output_dir = Path("outputs/data_cache")

    for symbol in symbols:
        df = cache.load_daily(symbol)
        if df.empty:
            continue

        df = feature_engine.compute(df)

        # 合并基本面
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
        logger.info(f"{symbol}: features recomputed ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Incremental update A-share data")
    parser.add_argument("--all", action="store_true", help="Update ALL A-share stocks")
    parser.add_argument("--symbols", nargs="+", default=None, help="Stock symbols")
    parser.add_argument("--default-start", type=str, default="20150101", help="Start date if no cache exists")
    parser.add_argument("--no-market", action="store_true", help="Skip market data update")
    parser.add_argument("--no-fundamental", action="store_true", help="Skip fundamental data update")
    parser.add_argument("--no-features", action="store_true", help="Skip feature recompute")
    parser.add_argument("--force", action="store_true", help="Force full re-download (ignore cache)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("update_data")

    fetcher = AShareFetcher(
        source=config.get("data.source", "akshare"),
        tushare_token=config.get("data.tushare_token", ""),
    )

    if args.all:
        from scripts.download_data import get_all_symbols
        symbols = get_all_symbols(fetcher, logger)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get("data.symbols", ["000001"])
    adjust = config.get("data.adjust", "qfq")

    cache = DataCache(cache_dir=config.get("data.cache_dir", "outputs/data_cache"))

    logger.info(f"Update: {len(symbols)} stocks, force={args.force}")

    # 1. 行情增量更新
    updated = []
    if not args.no_market:
        updated = update_market_data(
            symbols, fetcher, cache, adjust, args.force, args.default_start, logger
        )

    # 2. 基本面更新
    if not args.no_fundamental:
        update_valuation_data(symbols, fetcher, cache, logger)
        update_financial_data(symbols, fetcher, cache, logger)

    # 3. 重算特征（只对有更新的 + 有基本面更新的股票）
    if not args.no_features:
        recompute_symbols = list(set(updated + symbols)) if not args.no_fundamental else updated
        if recompute_symbols:
            recompute_features(recompute_symbols, cache, config.raw, logger)

    # 4. 打印缓存摘要
    summary = cache.list_cached()
    if not summary.empty:
        logger.info("\nCache summary:\n" + summary.to_string(index=False))

    logger.info("Update complete.")


if __name__ == "__main__":
    main()

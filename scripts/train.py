import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger
from data.fetcher import AShareFetcher
from data.cache import DataCache
from data.features import FeatureEngine
from data.liquidity import LiquidityFeatureEngine
from data.fundamental import FundamentalEngine
from data.preprocessor import Preprocessor, DataPipeline, DataQualityError
from data.dataset import TimeSeriesDataset, ReturnBucketizer
from model.transformer import StockTransformer
from model.trainer import Trainer
from model.evaluator import Evaluator
from model.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Train Stock Transformer model")
    parser.add_argument("--symbol", type=str, default="000001", help="Stock symbol")
    parser.add_argument("--start", type=str, default="20150101")
    parser.add_argument("--end", type=str, default="20260515")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("train")

    # --- Data ---
    fetcher = AShareFetcher(
        source=config.get("data.source", "tushare"),
        tushare_token=config.get("data.tushare_token", ""),
    )
    cache = DataCache(cache_dir=config.get("data.cache_dir", "outputs/data_cache"))

    logger.info(f"Loading data for {args.symbol}")
    df = cache.load_daily(args.symbol)
    if df.empty:
        logger.info(f"No cache, fetching from API")
        df = fetcher.fetch_daily(args.symbol, args.start, args.end)
        if not df.empty:
            cache.save_daily(args.symbol, df)

    if df.empty:
        raise RuntimeError(f"No data available for {args.symbol}")

    # --- Step 1: Feature computation ---
    # Phase 1: only tech + liquidity engines (no market/valuation/fundamental/industry/cross-sectional)
    tech_engine = FeatureEngine()
    liq_engine = LiquidityFeatureEngine()

    df = tech_engine.compute(df)
    df = liq_engine.compute(df)

    # Collect feature columns from all active engines
    feature_cols = tech_engine.feature_columns + liq_engine.feature_columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"After feature computation: {len(df)} rows, {len(feature_cols)} features")

    # Classify features: always-available vs conditional
    # Tech + liquidity are always-available (from OHLCV + daily_basic)
    always_available_cols = feature_cols
    conditional_cols = []

    # --- Steps 2-3: Data-independent preprocessing ---
    pipeline = DataPipeline(config.raw)
    df = pipeline.preprocess_stock(df, feature_cols, always_available_cols, conditional_cols)
    logger.info(f"After preprocessing: {len(df)} rows, {len(feature_cols)} features")

    # --- Walk-forward training ---
    preprocessor = Preprocessor(config.raw)
    folds = preprocessor.generate_folds(len(df))
    if not folds:
        raise RuntimeError("Not enough data for walk-forward splits")

    model_cfg = config.raw.get("model", {})
    training_cfg = config.raw.get("training", {})
    seq_len = model_cfg.get("seq_len", 250)
    pred_len = model_cfg.get("pred_len", 5)
    num_classes = model_cfg.get("num_classes", 8)
    batch_size = training_cfg.get("batch_size", 32)
    bucket_type = config.raw.get("features", {}).get("bucket_type", "5d")
    buckets = ReturnBucketizer.BUCKETS_5D if bucket_type == "5d" else ReturnBucketizer.BUCKETS_20D

    all_metrics = []
    checkpoint_mgr = CheckpointManager()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for fold in folds:
        logger.info(f"\n{'='*40} Fold {fold.fold_idx} {'='*40}")

        # --- Steps 4-6: Fold-dependent preprocessing ---
        try:
            train_norm, val_norm, test_norm = pipeline.prepare_fold(
                df, feature_cols, fold
            )
        except DataQualityError as e:
            logger.warning(f"Fold {fold.fold_idx} skipped: {e}")
            continue

        close_arr = df["close"].values.astype(np.float32)
        train_close = close_arr[fold.train_start : fold.train_end]
        val_close = close_arr[fold.val_start : fold.val_end]
        test_close = close_arr[fold.test_start : fold.test_end]

        try:
            train_ds = TimeSeriesDataset(
                train_norm, train_close,
                seq_len=seq_len, pred_len=pred_len, bucket_type=bucket_type,
            )
            val_ds = TimeSeriesDataset(
                val_norm, val_close,
                seq_len=seq_len, pred_len=pred_len, bucket_type=bucket_type,
            )
            test_ds = TimeSeriesDataset(
                test_norm, test_close,
                seq_len=seq_len, pred_len=pred_len, bucket_type=bucket_type,
            )
        except ValueError as e:
            logger.warning(f"Fold {fold.fold_idx} skipped: {e}")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Build model
        model = StockTransformer(
            num_features=len(feature_cols),
            d_model=model_cfg.get("d_model", 256),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 6),
            d_ff=model_cfg.get("d_ff", 1024),
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.15),
            seq_len=seq_len,
            pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
            pooling=model_cfg.get("pooling", "attention"),
        )

        trainer = Trainer(model, config.raw, device=device)

        history = trainer.train(
            train_loader, val_loader,
            symbol=args.symbol, fold=fold.fold_idx,
        )

        # Evaluate on test set
        logits = trainer.predict(test_loader).numpy()
        targets = np.concatenate([y.numpy() for _, y in test_loader], axis=0)

        metrics = Evaluator.compute_classification_metrics(logits, targets, buckets)
        all_metrics.append(metrics)
        logger.info(
            f"Fold {fold.fold_idx} test: "
            f"top1={metrics['top1_accuracy']:.3f} "
            f"top2={metrics['top2_accuracy']:.3f} "
            f"F1={metrics['weighted_f1']:.3f} "
            f"IC={metrics['ic']:.4f} "
            f"dir_acc={metrics['directional_accuracy']:.3f}"
        )

    # Summary
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        logger.info(f"\n{'='*40} Summary {'='*40}")
        logger.info(f"Avg top1 accuracy: {avg_metrics['top1_accuracy']:.3f}")
        logger.info(f"Avg top2 accuracy: {avg_metrics['top2_accuracy']:.3f}")
        logger.info(f"Avg weighted F1:   {avg_metrics['weighted_f1']:.3f}")
        logger.info(f"Avg IC:            {avg_metrics['ic']:.4f}")
        logger.info(f"Avg dir accuracy:  {avg_metrics['directional_accuracy']:.3f}")


if __name__ == "__main__":
    main()

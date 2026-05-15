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
from data.preprocessor import Preprocessor
from data.dataset import TimeSeriesDataset
from model.transformer import StockTransformer
from model.trainer import Trainer
from model.evaluator import Evaluator
from model.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Train Stock Transformer model")
    parser.add_argument("--symbol", type=str, default="000001", help="Stock symbol")
    parser.add_argument("--start", type=str, default="20200101")
    parser.add_argument("--end", type=str, default="20241231")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("train")

    # --- Data ---
    fetcher = AShareFetcher(source=config.get("data.source", "akshare"))
    cache = DataCache(cache_dir=config.get("data.cache_dir", "outputs/data_cache"))
    feature_engine = FeatureEngine(config.raw)

    logger.info(f"Fetching data for {args.symbol}")
    df = cache.get(args.symbol, args.start, args.end)
    if df.empty:
        df = fetcher.fetch_daily(args.symbol, args.start, args.end)
        if not df.empty:
            cache.put(args.symbol, args.start, args.end, df)

    if df.empty:
        logger.error("No data available")
        return

    df = feature_engine.compute(df)
    feature_cols = [c for c in feature_engine.feature_columns if c in df.columns]
    df = df.dropna(subset=feature_cols)
    logger.info(f"Data: {len(df)} rows, {len(feature_cols)} features")

    # --- Walk-forward training ---
    preprocessor = Preprocessor(config.raw)
    folds = preprocessor.generate_folds(len(df))
    if not folds:
        logger.error("Not enough data for walk-forward splits")
        return

    model_cfg = config.raw.get("model", {})
    training_cfg = config.raw.get("training", {})
    seq_len = model_cfg.get("seq_len", 60)
    pred_len = model_cfg.get("pred_len", 5)
    batch_size = training_cfg.get("batch_size", 64)

    all_metrics = []
    checkpoint_mgr = CheckpointManager()

    for fold in folds:
        logger.info(f"\n{'='*40} Fold {fold.fold_idx} {'='*40}")

        train_norm, val_norm, test_norm = preprocessor.prepare_fold_data(
            df, feature_cols, fold
        )

        # Targets: close price returns
        close_prices = df["close"].values.astype(np.float32)
        train_targets = np.diff(close_prices[fold.train_start:fold.train_end]) / close_prices[fold.train_start:fold.train_end-1]
        val_targets = np.diff(close_prices[fold.val_start:fold.val_end]) / close_prices[fold.val_start:fold.val_end-1]
        test_targets = np.diff(close_prices[fold.test_start:fold.test_end]) / close_prices[fold.test_start:fold.test_end-1]

        # Pad targets to match data length
        train_targets = np.concatenate([train_targets, np.zeros(1)]).astype(np.float32)
        val_targets = np.concatenate([val_targets, np.zeros(1)]).astype(np.float32)
        test_targets = np.concatenate([test_targets, np.zeros(1)]).astype(np.float32)

        try:
            train_ds = TimeSeriesDataset(train_norm, train_targets, seq_len, pred_len)
            val_ds = TimeSeriesDataset(val_norm, val_targets, seq_len, pred_len)
            test_ds = TimeSeriesDataset(test_norm, test_targets, seq_len, pred_len)
        except ValueError as e:
            logger.warning(f"Fold {fold.fold_idx} skipped: {e}")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Build model
        model = StockTransformer(
            num_features=len(feature_cols),
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 4),
            d_ff=model_cfg.get("d_ff", 512),
            pred_len=pred_len,
            dropout=model_cfg.get("dropout", 0.1),
            seq_len=seq_len,
            pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
            pooling=model_cfg.get("pooling", "attention"),
        )

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(model, config.raw, device=device)

        history = trainer.train(
            train_loader, val_loader,
            symbol=args.symbol, fold=fold.fold_idx,
        )

        # Evaluate on test set
        predictions = trainer.predict(test_loader).numpy()
        actuals = np.concatenate([y.numpy() for _, y in test_loader], axis=0)

        metrics = Evaluator.compute_metrics(predictions, actuals)
        all_metrics.append(metrics)
        logger.info(
            f"Fold {fold.fold_idx} test: "
            f"dir_acc={metrics['directional_accuracy']:.3f} "
            f"IC={metrics['ic']:.4f} "
            f"RMSE={metrics['rmse']:.6f}"
        )

    # Summary
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        logger.info(f"\n{'='*40} Summary {'='*40}")
        logger.info(f"Avg directional accuracy: {avg_metrics['directional_accuracy']:.3f}")
        logger.info(f"Avg IC: {avg_metrics['ic']:.4f}")
        logger.info(f"Avg Rank IC: {avg_metrics['rank_ic']:.4f}")
        logger.info(f"Avg RMSE: {avg_metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()

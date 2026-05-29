"""快速测试：数据加载 → 训练 → 回测，验证全流程。"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from pipeline.config import load_experiment
from pipeline.data_pipeline import FEATURE_CACHE_VERSION
from pipeline.factory import create_model, create_trainer
from data.features.v1_45 import V1_45FeatureEngine, BINS_1D, BINS_5D, BINS_20D
from data.stock_selector import load_symbols
from backtest.engine_v2 import BacktestEngine
from torch.utils.data import Dataset
class NumpyDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.FloatTensor(self.X[i]), torch.LongTensor(self.Y[i])


def _sliding_samples(feat, close, seq_len):
    """向量化滑动窗口。（与 data_pipeline 一致）"""
    T, F = feat.shape
    if T < seq_len + 25:
        return None, None
    shape = (T - seq_len + 1, seq_len, F)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    X = np.lib.stride_tricks.as_strided(feat, shape=shape, strides=strides).copy()

    usable = len(X) - 20
    if usable <= 0:
        return None, None
    X = X[:usable]

    t = np.arange(usable) + seq_len - 1
    r1 = close[t + 1] / close[t] - 1
    r5 = close[t + 5] / close[t] - 1
    r20 = close[t + 20] / close[t] - 1

    def bkt(r, bins):
        out = np.full(len(r), len(bins) - 2, dtype=np.int64)
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            out[(r >= lo) & (r < hi)] = i
        return out

    Y = np.stack([bkt(r1, BINS_1D), bkt(r5, BINS_5D), bkt(r20, BINS_20D)], axis=1)
    return X.astype(np.float32), Y.astype(np.int64)


def main():
    logger = setup_logger("quick_test")
    config = load_experiment("test")
    logger.info(f"Config: {config['name']}")

    dc = config["data"]
    cache_dir = dc.get("cache_dir", "outputs/data_cache")
    seq_len = config["features"].get("seq_len", 120)
    n_samples = 10  # only use first 10 stocks

    # ---- load 10 stocks ----
    symbols = load_symbols(dc.get("stock_list", "liquid100"))[:n_samples]
    logger.info(f"Symbols: {symbols}")

    feat_dir = Path("outputs/features")
    version_path = feat_dir / "_feature_cache_version.txt"
    if not version_path.exists() or version_path.read_text().strip() != FEATURE_CACHE_VERSION:
        raise RuntimeError("Feature cache is stale. Run pipeline/data_pipeline.py before quick_test.")
    feature_dfs = {}
    for s in symbols:
        p = feat_dir / f"{s}.parquet"
        if p.exists():
            feature_dfs[s] = pd.read_parquet(p)

    from data.features.v1_45 import V1_45FeatureEngine
    fcols = V1_45FeatureEngine(config, None).feature_columns
    logger.info(f"Features: {len(fcols)} cols, {len(feature_dfs)} stocks")

    # ---- build numpy samples ----
    ds_cfg = config["data_split"]
    train_end = pd.Timestamp(ds_cfg["train_end"])
    val_end = pd.Timestamp(ds_cfg["val_end"])

    all_X, all_Y, splits = [], [], []
    for s, df in feature_dfs.items():
        for c in fcols:
            if c not in df.columns:
                df[c] = 0.0
        feat = df[fcols].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)
        X, Y = _sliding_samples(feat, close, seq_len)
        if X is None:
            continue
        dt = pd.to_datetime(df["datetime"]).values
        sample_dates = dt[seq_len - 1 : seq_len - 1 + len(X)]
        train_mask = sample_dates <= train_end.to_numpy()
        val_mask = (sample_dates > train_end.to_numpy()) & (sample_dates <= val_end.to_numpy())

        for mask, sid in [(train_mask, 0), (val_mask, 1)]:
            if mask.sum() > 0:
                all_X.append(X[mask])
                all_Y.append(Y[mask])
                splits.extend([sid] * mask.sum())

    X_all = np.concatenate(all_X).astype(np.float32)
    Y_all = np.concatenate(all_Y).astype(np.int64)
    splits = np.array(splits)
    train_mask, val_mask = splits == 0, splits == 1
    logger.info(f"Samples: train={train_mask.sum()}, val={val_mask.sum()}")

    # ---- preprocessing (simple) ----
    lo = np.nanpercentile(X_all[train_mask], 1, axis=(0, 1))
    hi = np.nanpercentile(X_all[train_mask], 99, axis=(0, 1))
    X = np.clip(X_all, lo, hi)
    col_means = np.nanmean(X[train_mask], axis=(0, 1))
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = col_means[nan_idx[2]]
    mean = X[train_mask].mean(axis=(0, 1))
    std = X[train_mask].std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    X = (X - mean) / std

    train_ds = NumpyDataset(X[train_mask], Y_all[train_mask])
    val_ds = NumpyDataset(X[val_mask], Y_all[val_mask])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ---- train ----
    model = create_model(config)
    trainer = create_trainer(model, config)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    history = trainer.train(train_loader, val_loader, name="test", fold=0)
    torch.save(model.state_dict(), "outputs/checkpoints/test_best.pt")

    # ---- eval ----
    from training.evaluator import compute_metrics
    pred_1d, pred_5d, pred_20d = trainer.predict(val_loader)
    actuals = np.stack([val_ds[i][1].numpy() for i in range(len(val_ds))])
    metrics = compute_metrics(pred_1d, pred_5d, pred_20d,
                              actuals[:, 0].astype(float),
                              actuals[:, 1].astype(float),
                              actuals[:, 2].astype(float))
    logger.info(f"Rank IC: 1d={metrics['rank_ic_1d']:.4f}, 5d={metrics['rank_ic_5d']:.4f}, 20d={metrics['rank_ic_20d']:.4f}")

    # ---- backtest ----
    engine = BacktestEngine(config)
    bt_metrics, equity, trades = engine.run(model, feature_dfs,
                                            {"feature_columns": fcols, "mean": mean.tolist(), "std": std.tolist()})

    logger.info(f"Backtest: return={bt_metrics.get('total_return', 0):.2%}, sharpe={bt_metrics.get('sharpe_ratio', 0):.2f}, n_trades={bt_metrics.get('n_trades', 0)}")

    with open("outputs/results/quick_test.json", "w") as f:
        json.dump({"eval": metrics, "backtest": bt_metrics}, f, indent=2, default=str)
    logger.info("Done. results saved to outputs/results/quick_test.json")


if __name__ == "__main__":
    main()

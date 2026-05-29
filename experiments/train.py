"""
训练脚本。用法:
  python experiments/train.py --config baseline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from pipeline.config import load_experiment
from pipeline.factory import create_model, create_trainer
from pipeline.data_pipeline import FEATURE_CACHE_VERSION
from data.stock_selector import load_symbols
from data.features.v1_45 import V1_45FeatureEngine, BINS_1D, BINS_5D, BINS_20D


def _sliding_samples(feat, close, seq_len):
    """向量化滑动窗口生成样本。X: [N,seq_len,F], Y: [N,3]"""
    T, F = feat.shape
    if T < seq_len + 25:
        return None, None
    X = np.lib.stride_tricks.as_strided(
        feat, shape=(T - seq_len + 1, seq_len, F),
        strides=(feat.strides[0], feat.strides[0], feat.strides[1])
    ).copy()
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


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.FloatTensor(self.X[i]), torch.LongTensor(self.Y[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logger = setup_logger("train")
    config = load_experiment(args.config)
    logger.info(f"Config: {config['name']}")

    dc = config["data"]
    cache_dir = dc.get("cache_dir", "outputs/data_cache")
    seq_len = config["features"].get("seq_len", 120)
    stock_list = dc.get("stock_list", "liquid100")
    symbols = load_symbols(stock_list)
    logger.info(f"Symbols: {len(symbols)} stocks")

    # ---- load features ----
    feat_dir = Path("outputs/features")
    version_path = feat_dir / "_feature_cache_version.txt"
    if not version_path.exists() or version_path.read_text().strip() != FEATURE_CACHE_VERSION:
        raise RuntimeError("Feature cache is stale. Run pipeline/data_pipeline.py before training.")
    fcols = V1_45FeatureEngine(config, None).feature_columns
    feature_dfs = {}
    for s in symbols:
        p = feat_dir / f"{s}.parquet"
        if p.exists():
            feature_dfs[s] = pd.read_parquet(p)
    logger.info(f"Loaded {len(feature_dfs)} stocks × {len(fcols)} features")

    # ---- build samples ----
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
        sd = dt[seq_len - 1 : seq_len - 1 + len(X)]
        train_mask = sd <= train_end.to_numpy()
        val_mask = (sd > train_end.to_numpy()) & (sd <= val_end.to_numpy())
        for mask, sid in [(train_mask, 0), (val_mask, 1)]:
            if mask.sum() > 0:
                all_X.append(X[mask])
                all_Y.append(Y[mask])
                splits.extend([sid] * mask.sum())

    X_all = np.concatenate(all_X).astype(np.float32)
    Y_all = np.concatenate(all_Y).astype(np.int64)
    splits = np.array(splits, dtype=np.int8)
    train_mask, val_mask = splits == 0, splits == 1
    logger.info(f"Samples: train={train_mask.sum()}, val={val_mask.sum()}")

    # ---- preprocess (per-feature mean/std from train) ----
    Xt = X_all[train_mask]
    all_nan_features = np.isnan(Xt).all(axis=(0, 1))
    if all_nan_features.any():
        missing_cols = [c for c, missing in zip(fcols, all_nan_features) if missing]
        logger.warning(f"Train-only all-NaN features filled with 0: {missing_cols}")
        X_all[:, :, all_nan_features] = 0.0
        Xt = X_all[train_mask]
    lo = np.nanpercentile(Xt, 1, axis=(0, 1))
    hi = np.nanpercentile(Xt, 99, axis=(0, 1))
    X_all = np.clip(X_all, lo, hi)
    col_means = np.nanmean(X_all[train_mask], axis=(0, 1))
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_idx = np.where(np.isnan(X_all))
    X_all[nan_idx] = col_means[nan_idx[2]]
    Xt = X_all[train_mask]
    mean = Xt.mean(axis=(0, 1))
    std = Xt.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    for i, c in enumerate(fcols):
        if "rank" in c or "max_norm" in c:
            mean[i] = 0.0; std[i] = 1.0
    X_all = (X_all - mean) / std

    train_ds = ArrayDataset(X_all[train_mask], Y_all[train_mask])
    val_ds = ArrayDataset(X_all[val_mask], Y_all[val_mask])
    tcfg = config["training"]
    bc = tcfg.get("batch_size", 256)
    num_workers = tcfg.get("num_workers", 4)
    loader_kwargs = {
        "batch_size": bc,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = tcfg.get("prefetch_factor", 4)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ---- train ----
    model = create_model(config)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = create_trainer(model, config)
    history = trainer.train(train_loader, val_loader, name=config["name"], fold=0)
    device = trainer.device

    # ---- save ----
    out = Path("outputs/checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    best_fold_path = out / f"{config['name']}_fold0_best.pt"
    best_meta = {}
    if best_fold_path.exists():
        best_meta = torch.load(best_fold_path, map_location=device, weights_only=False)
        model.load_state_dict(best_meta["model_state"])

    ckpt_path = out / f"{config['name']}_best.pt"
    torch.save({"model_state": model.state_dict(),
                 "mean": mean, "std": std, "feature_columns": fcols,
                 "epoch": best_meta.get("epoch"),
                 "val_loss": best_meta.get("val_loss")}, ckpt_path)

    # ---- eval ----
    logger.info("Evaluating...")
    model.eval()
    correct_1d, correct_5d, correct_20d, total = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            o = model(x)
            correct_1d += (o["logits_1d"].argmax(-1) == y[:, 0]).sum().item()
            correct_5d += (o["logits_5d"].argmax(-1) == y[:, 1]).sum().item()
            correct_20d += (o["logits_20d"].argmax(-1) == y[:, 2]).sum().item()
            total += len(y)
    logger.info(f"Val accuracy: 1d={correct_1d/total:.4f} 5d={correct_5d/total:.4f} 20d={correct_20d/total:.4f}")
    from collections import Counter
    logger.info(f"1d pred dist: {Counter(o['logits_1d'].argmax(-1).cpu().numpy())}")
    logger.info(f"1d true dist: {Counter(Y_all[val_mask][:, 0].tolist())}")
    logger.info(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()

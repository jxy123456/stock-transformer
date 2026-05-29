"""实验主流程：加载配置 → 准备数据 → 训练 → 评估 → 回测 → 保存结果。"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from data.cache import DataCache
from data.features.v1_45 import V1_45FeatureEngine
from pipeline.config import load_experiment
from pipeline.factory import create_model, create_trainer, load_checkpoint
from training.dataset import StockDataset, winsorize, normalize_zscore
from training.evaluator import compute_metrics, expected_returns
from data.features.v1_45 import CENTERS_5D, CENTERS_20D


def run_experiment(exp_name: str, backtest_only=False, eval_only=False):
    cfg = load_experiment(exp_name)
    name = cfg.get("name", exp_name)
    logger.info(f"Experiment: {name}")

    # output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/results") / name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    import yaml
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # load data
    dc = cfg.get("data", {})
    cache = DataCache(dc.get("cache_dir", "outputs/data_cache"))
    symbols = dc.get("symbols", ["000001", "000002"])
    start = dc.get("start_date", "2015-01-01")
    end = dc.get("end_date", "2025-12-31")

    # load index data from cache
    index_data = {}
    for idx in dc.get("indexes", ["000300"]):
        df = cache.load_index(str(idx))
        if not df.empty:
            index_data[str(idx)] = df

    # feature engine
    engine = V1_45FeatureEngine(cfg, cache, index_data=index_data)

    if not backtest_only:
        _do_train(cfg, engine, symbols, start, end, out_dir, cache)

    if not eval_only:
        _do_backtest(cfg, engine, symbols, start, end, out_dir)

    logger.info(f"Results saved to {out_dir}")


def _compute_features(cfg, engine, symbols, start, end):
    """Compute features for all symbols, with caching."""
    all_dfs = {}
    for i, s in enumerate(symbols):
        if (i + 1) % 200 == 1:
            logger.info(f"  Features [{i+1}/{len(symbols)}] ...")
        df = engine.load(s)
        if df.empty:
            df = engine.compute(s, start, end)
            if not df.empty:
                engine.save(s, df)
        if not df.empty:
            all_dfs[s] = df
    return all_dfs


def _do_train(cfg, engine, symbols, start, end, out_dir, cache):
    """Full training pipeline."""
    logger.info("Computing features...")
    all_dfs = _compute_features(cfg, engine, symbols, start, end)
    logger.info(f"  {len(all_dfs)} stocks with features")

    fc = cfg.get("features", {})
    seq_len = fc.get("seq_len", 120)
    fcols = engine.feature_columns
    ds_cfg = cfg.get("data_split", {})
    train_end = pd.Timestamp(ds_cfg.get("train_end", "2022-12-31"))
    val_end = pd.Timestamp(ds_cfg.get("val_end", "2023-12-31"))

    # split by time
    train_dfs, val_dfs = {}, {}
    for s, df in all_dfs.items():
        df["datetime"] = pd.to_datetime(df["datetime"])
        train_part = df[df["datetime"] <= train_end]
        val_part = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)]
        if len(train_part) > seq_len + 20:
            train_dfs[s] = train_part
        if len(val_part) > seq_len + 20:
            val_dfs[s] = val_part

    # build datasets
    from training.dataset import MultiStockDataset
    train_ds = MultiStockDataset(train_dfs, seq_len=seq_len, feature_columns=fcols)
    val_ds = MultiStockDataset(val_dfs, seq_len=seq_len, feature_columns=fcols)

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # winsorize + normalize on training data
    all_x = np.stack([train_ds[i][0].numpy() for i in range(min(50000, len(train_ds)))])
    all_x = winsorize(all_x, 0.01, 0.99)
    all_x, mean, std = normalize_zscore(all_x)
    # Apply normalization directly to dataset samples
    for i in range(len(train_ds)):
        x, y = train_ds.samples[i]
        x = torch.FloatTensor(winsorize(x.numpy(), 0.01, 0.99))
        x = torch.FloatTensor((x.numpy() - mean) / std)
        train_ds.samples[i] = (x, y)
    for i in range(len(val_ds)):
        x, y = val_ds.samples[i]
        x = torch.FloatTensor(winsorize(x.numpy(), 0.01, 0.99))
        x = torch.FloatTensor((x.numpy() - mean) / std)
        val_ds.samples[i] = (x, y)

    batch_size = cfg.get("training", {}).get("batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # model
    model = create_model(cfg)
    trainer = create_trainer(model, cfg)
    history = trainer.train(train_loader, val_loader, name=cfg.get("name", "model"))
    torch.save(model.state_dict(), out_dir / "model.pt")

    # evaluate
    pred_5d, pred_20d = trainer.predict(val_loader)
    actuals = np.stack([val_ds[i][1].numpy() for i in range(len(val_ds))])
    actual_5d = actuals[:, 0]
    actual_20d = actuals[:, 1]
    metrics = compute_metrics(pred_5d, pred_20d, actual_5d, actual_20d)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics: rank_ic_5d={metrics.get('rank_ic_5d', 0):.4f}, "
                f"rank_ic_score={metrics.get('rank_ic_score', 0):.4f}")


def _do_backtest(cfg, engine, symbols, start, end, out_dir):
    logger.info("Running backtest...")
    all_dfs = _compute_features(cfg, engine, symbols, start, end)

    model = create_model(cfg)
    model_path = out_dir / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    from backtest.event_driven import EventDrivenBacktest
    bt = EventDrivenBacktest()
    metrics = bt.run(model, all_dfs, cfg)

    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if "equity_curve" in metrics:
        eq = pd.DataFrame(metrics.pop("equity_curve"), columns=["date", "value"])
        eq.to_csv(out_dir / "equity_curve.csv", index=False)

    logger.info(f"Backtest: sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                f"max_dd={metrics.get('max_drawdown', 0):.2%}")

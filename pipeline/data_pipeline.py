"""数据处理流水线：全 numpy 向量化，一步到位。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from data.features.v1_45 import V1_45FeatureEngine, BINS_1D, BINS_5D, BINS_20D, bucketize
from data.stock_selector import load_symbols


def _load_parquet(cache_dir: str, prefix: str, symbol: str) -> pd.DataFrame:
    p = Path(cache_dir) / f"{prefix}_{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


class NumpyDataset(Dataset):
    """纯索引式 Dataset：X[N,120,F] + Y[N,3]，getitem 只做切片，零 Python 开销。"""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor(self.Y[idx])


def _sliding_samples(feature_arr, close_arr, seq_len):
    """numpy stride 滑动窗口，一次生成所有样本。

    feature_arr: [T, F]
    close_arr:   [T]
    返回: X [N, seq_len, F], Y [N, 3]
    """
    T, F = feature_arr.shape
    if T < seq_len + 25:
        return None, None

    # X: sliding window
    shape = (T - seq_len + 1, seq_len, F)
    strides = (feature_arr.strides[0], feature_arr.strides[0], feature_arr.strides[1])
    X = np.lib.stride_tricks.as_strided(feature_arr, shape=shape, strides=strides)
    X = X.copy()  # make contiguous (strides trick view can cause issues)

    # trim last 20 samples that can't have complete labels
    usable = len(X) - 20
    if usable <= 0:
        return None, None
    X = X[:usable]

    # Y: future returns, vectorized
    t_indices = np.arange(usable) + seq_len - 1
    future_close_1d = close_arr[t_indices + 1]
    future_close_5d = close_arr[t_indices + 5]
    future_close_20d = close_arr[t_indices + 20]
    current_close = close_arr[t_indices]

    ret_1d = future_close_1d / current_close - 1
    ret_5d = future_close_5d / current_close - 1
    ret_20d = future_close_20d / current_close - 1

    y1 = np.array([bucketize(r, BINS_1D) for r in ret_1d], dtype=np.int64)
    y5 = np.array([bucketize(r, BINS_5D) for r in ret_5d], dtype=np.int64)
    y20 = np.array([bucketize(r, BINS_20D) for r in ret_20d], dtype=np.int64)
    Y = np.stack([y1, y5, y20], axis=1).astype(np.int64)

    return X.astype(np.float32), Y


def run_data_pipeline(config: dict = None):
    if config is None:
        from pipeline.config import load_experiment
        config = load_experiment("baseline")

    dc = config.get("data", {})
    cache_dir = dc.get("cache_dir", "outputs/data_cache")
    start_date = dc.get("start_date", "2015-01-01")
    end_date = dc.get("end_date", "2025-12-31")
    seq_len = config.get("features", {}).get("seq_len", 120)
    ds_cfg = config.get("data_split", {})
    train_end = pd.Timestamp(ds_cfg.get("train_end", "2022-12-31"))
    val_end = pd.Timestamp(ds_cfg.get("val_end", "2023-12-31"))

    # ---- stock list ----
    symbols = load_symbols(dc.get("stock_list", "liquid100"))
    logger.info(f"Stock list: {len(symbols)} symbols")

    # ---- index & industry ----
    index_data = {}
    for idx in dc.get("indexes", ["000300"]):
        df = _load_parquet(cache_dir, "index", str(idx))
        if not df.empty:
            index_data[str(idx)] = df
    industry_map = {}
    ind_csv = Path(cache_dir) / "_industry_map.csv"
    if ind_csv.exists():
        ind_df = pd.read_csv(ind_csv)
        industry_map = {str(s).zfill(6): ind for s, ind in zip(ind_df["symbol"], ind_df["industry"])}

    # ---- feature engine ----
    class _FC:
        def __init__(s, d): s.dir = Path(d)
        def load_daily(s, sym): return _load_parquet(cache_dir, "daily", sym)
        def load_valuation(s, sym): return _load_parquet(cache_dir, "valuation", sym)
        def load_financial(s, sym): return _load_parquet(cache_dir, "financial", sym)

    engine = V1_45FeatureEngine(config, _FC(cache_dir), index_data=index_data, industry_map=industry_map)
    feat_dir = Path("outputs/features")
    feat_dir.mkdir(parents=True, exist_ok=True)

    # ---- step 1: individual features ----
    logger.info("=== Step 1: 个股特征 ===")
    feature_dfs = {}
    for i, s in enumerate(symbols):
        if (i + 1) % 30 == 1:
            logger.info(f"  [{i+1}/{len(symbols)}] ...")
        fpath = feat_dir / f"{s}.parquet"
        if fpath.exists():
            feature_dfs[s] = pd.read_parquet(fpath)
        else:
            df = engine.compute(s, start_date, end_date)
            if not df.empty:
                df.to_parquet(fpath)
                feature_dfs[s] = df
    logger.info(f"  {len(feature_dfs)} stocks")

    # ---- step 2: cross-sectional ranks (numpy) ----
    logger.info("=== Step 2: 截面排名 ===")
    from scipy.stats import rankdata

    all_dates = sorted(set().union(*[
        set(pd.to_datetime(df["datetime"]).values) for df in feature_dfs.values()
    ]))
    n_dates, n_stocks = len(all_dates), len(feature_dfs)
    logger.info(f"  {n_dates} × {n_stocks}")

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    syms = list(feature_dfs.keys())

    def _build_matrix(col):
        mat = np.full((n_dates, n_stocks), np.nan)
        for j, s in enumerate(syms):
            df = feature_dfs[s]
            if col not in df.columns:
                continue
            ser = pd.Series(df[col].values, index=pd.to_datetime(df["datetime"]))
            mat[:, j] = ser.reindex(all_dates).values
        return mat

    def _rank_pct_matrix(mat):
        res = np.full_like(mat, np.nan)
        for i in range(len(mat)):
            row = mat[i]
            v = ~np.isnan(row)
            if v.sum() < 2:
                continue
            res[i, v] = rankdata(row[v], method="average") / v.sum()
        return res

    mat_mcap = _build_matrix("log_total_market_cap_max_norm")
    mat_ret20 = _build_matrix("ret_20d")
    mat_ey = _build_matrix("earnings_yield")

    rank_mcap = _rank_pct_matrix(mat_mcap)
    rank_ret20 = _rank_pct_matrix(mat_ret20)

    for j, s in enumerate(syms):
        df = feature_dfs[s]
        idxs = [date_to_idx[d] for d in pd.to_datetime(df["datetime"]).values if d in date_to_idx]
        if idxs:
            df["total_market_cap_rank_market"] = rank_mcap[idxs, j]
            df["ret_20d_rank_market"] = rank_ret20[idxs, j]

    # industry ranks
    ind_groups = {}
    for s in syms:
        ind_groups.setdefault(industry_map.get(s, "unknown"), []).append(s)

    for base_col, out_col in [("log_total_market_cap_max_norm", "total_market_cap_rank_industry"),
                               ("earnings_yield", "pe_rank_industry"),
                               ("ret_20d", "ret_20d_rank_industry")]:
        mat = _build_matrix(base_col)
        for ind, ind_syms in ind_groups.items():
            if len(ind_syms) < 3:
                continue
            cols = [syms.index(x) for x in ind_syms]
            sub_rank = _rank_pct_matrix(mat[:, cols])
            for s, ci in zip(ind_syms, range(len(ind_syms))):
                df = feature_dfs[s]
                idxs = [date_to_idx[d] for d in pd.to_datetime(df["datetime"]).values if d in date_to_idx]
                if idxs:
                    df[out_col] = sub_rank[idxs, ci]

    for s, df in feature_dfs.items():
        df.to_parquet(feat_dir / f"{s}.parquet")

    # ---- step 3: numpy samples ----
    logger.info("=== Step 3: numpy 批量生成样本 ===")
    fcols = engine.feature_columns
    all_X, all_Y, all_splits = [], [], []  # splits: 0=train, 1=val, 2=test

    for s, df in feature_dfs.items():
        dt = pd.to_datetime(df["datetime"]).values
        # ensure all feature columns exist, fill missing with 0
        for c in fcols:
            if c not in df.columns:
                df[c] = 0.0
        feat = df[fcols].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)

        X, Y = _sliding_samples(feat, close, seq_len)
        if X is None:
            continue

        # assign split per sample by date
        sample_dates = dt[seq_len - 1: seq_len - 1 + len(X)]
        train_mask = sample_dates <= train_end.to_numpy()
        val_mask = (sample_dates > train_end.to_numpy()) & (sample_dates <= val_end.to_numpy())
        test_mask = sample_dates > val_end.to_numpy()

        for mask, split_id in [(train_mask, 0), (val_mask, 1), (test_mask, 2)]:
            if mask.sum() > 0:
                all_X.append(X[mask])
                all_Y.append(Y[mask])
                all_splits.extend([split_id] * mask.sum())

    X_all = np.concatenate(all_X, axis=0).astype(np.float32)
    Y_all = np.concatenate(all_Y, axis=0).astype(np.int64)
    splits = np.array(all_splits, dtype=np.int8)

    logger.info(f"  Total: {len(X_all)} samples, Train: {(splits==0).sum()}, Val: {(splits==1).sum()}, Test: {(splits==2).sum()}")

    # ---- step 4: preprocessing (numpy, vectorized) ----
    logger.info("=== Step 4: 预处理 ===")
    train_mask = splits == 0

    # winsorize per feature on train
    X_train = X_all[train_mask]
    lo = np.nanpercentile(X_train, 1, axis=(0, 1))
    hi = np.nanpercentile(X_train, 99, axis=(0, 1))
    X_all = np.clip(X_all, lo, hi)

    # fill remaining NaN with column mean from train
    col_means = np.nanmean(X_all[train_mask], axis=(0, 1))
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_mask = np.isnan(X_all)
    nan_idx = np.where(nan_mask)
    X_all[nan_idx] = col_means[nan_idx[2]]

    # zscore on train
    mean = X_all[train_mask].mean(axis=(0, 1))
    std = X_all[train_mask].std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)

    # skip rank/norm columns
    for i, c in enumerate(fcols):
        if "rank" in c or "max_norm" in c:
            mean[i] = 0.0; std[i] = 1.0

    X_all = (X_all - mean) / std

    # ---- step 5: save as .npy (no pickle size limit) ----
    logger.info("=== Step 5: 保存 ===")
    out = Path("outputs/datasets")
    out.mkdir(parents=True, exist_ok=True)

    for name, mask in [("train", 0), ("val", 1), ("test", 2)]:
        m = splits == mask
        np.save(out / f"{name}_X.npy", X_all[m])
        np.save(out / f"{name}_Y.npy", Y_all[m])
        logger.info(f"  {name}: {m.sum()} samples, X {X_all[m].shape}, Y {Y_all[m].shape}")

    meta = {
        "feature_columns": fcols,
        "seq_len": seq_len,
        "n_features": len(fcols),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
    }
    with open(out / "norm_stats.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    from collections import Counter
    for horizon, idx in [("1d", 0), ("5d", 1), ("20d", 2)]:
        cnt = Counter(Y_all[splits == 2, idx].tolist())
        logger.info(f"  {horizon} buckets: {dict(sorted(cnt.items()))}")

    logger.info(f"Done. {out}/train_X.npy train_Y.npy val_X.npy ...")


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger("data_pipeline")
    run_data_pipeline()

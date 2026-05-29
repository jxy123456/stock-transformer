"""滑动窗口 Dataset：从特征 DataFrame 生成 [B, 120, 45] 训练样本。"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data.features.v1_45 import BINS_5D, BINS_20D, bucketize


class StockDataset(Dataset):
    """单只股票的时序样本生成器。

    每个样本 = (输入序列, 未来标签)。
    输入: [seq_len, num_features] float32
    标签: (target_5d, target_20d) 均为 int64 bucket index
    """

    def __init__(self, feature_df, seq_len=120, feature_columns=None):
        if feature_columns is None:
            feature_cols = [c for c in feature_df.columns
                            if c not in ("datetime", "symbol", "close")]
        else:
            feature_cols = [c for c in feature_columns if c in feature_df.columns]
        self.data = feature_df[feature_cols].values.astype(np.float32)
        self.close = feature_df["close"].values.astype(np.float32)
        self.seq_len = seq_len
        self.n_samples = max(len(self.data) - seq_len - 20, 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        t = idx + self.seq_len - 1  # 预测日 index

        ret_5d = self._future_mean_ret(t, 5)
        ret_20d = self._future_mean_ret(t, 20)

        y5 = bucketize(ret_5d, BINS_5D)
        y20 = bucketize(ret_20d, BINS_20D)

        return (torch.FloatTensor(x),
                torch.LongTensor([y5, y20]))

    def _future_mean_ret(self, t, horizon):
        """未来 horizon 日平均收盘价 / 当前收盘价 - 1。"""
        f = t + horizon
        if f >= len(self.close):
            return np.nan
        if self.close[t] <= 0:
            return np.nan
        return float(np.nanmean(self.close[t + 1: f + 1]) / self.close[t] - 1)


class MultiStockDataset(Dataset):
    """多只股票拼接的数据集，每只股票贡献其有效样本。"""

    def __init__(self, feature_dfs: dict, seq_len=120, feature_columns=None):
        self.samples = []
        for symbol, df in feature_dfs.items():
            if df.empty:
                continue
            ds = StockDataset(df, seq_len=seq_len, feature_columns=feature_columns)
            for i in range(len(ds)):
                self.samples.append(ds[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def winsorize(data: np.ndarray, lo=0.01, hi=0.99) -> np.ndarray:
    """按分位数截尾。"""
    lo_val = np.nanpercentile(data, lo * 100, axis=0)
    hi_val = np.nanpercentile(data, hi * 100, axis=0)
    return np.clip(data, lo_val, hi_val)


def normalize_zscore(data: np.ndarray, mean=None, std=None):
    """Z-score 标准化，返回 (normalized, mean, std)。"""
    if mean is None:
        mean = np.nanmean(data, axis=0)
    if std is None:
        std = np.nanstd(data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std, mean, std

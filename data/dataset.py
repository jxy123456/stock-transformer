import numpy as np
import torch
from torch.utils.data import Dataset


class ReturnBucketizer:
    """Convert future returns to discrete bucket indices."""

    BUCKETS_5D = [
        (-np.inf, -0.05), (-0.05, -0.02), (-0.02, -0.01),
        (-0.01, 0.0), (0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, np.inf),
    ]

    BUCKETS_20D = [
        (-np.inf, -0.10), (-0.10, -0.05), (-0.05, -0.02),
        (-0.02, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, np.inf),
    ]

    @staticmethod
    def bucketize(future_return: float, buckets: list) -> int:
        for i, (lo, hi) in enumerate(buckets):
            if lo <= future_return < hi:
                return i
        return len(buckets) - 1

    @staticmethod
    def bucketize_array(returns: np.ndarray, buckets: list) -> np.ndarray:
        result = np.full(len(returns), len(buckets) - 1, dtype=np.int64)
        for i, (lo, hi) in enumerate(buckets):
            mask = (returns >= lo) & (returns < hi)
            result[mask] = i
        return result

    @staticmethod
    def bucket_centers(buckets: list) -> np.ndarray:
        """Midpoint of each bucket, for computing expected return."""
        centers = []
        for i, (lo, hi) in enumerate(buckets):
            if lo == -np.inf:
                lo = -2 * (buckets[1][1] - buckets[1][0])
            if hi == np.inf:
                hi = -lo
            centers.append((lo + hi) / 2)
        return np.array(centers)


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for return distribution classification.

    Input:  (seq_len, num_features) float tensor
    Target: single int (bucket index)
    """

    def __init__(
        self,
        data: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int = 250,
        pred_len: int = 5,
        bucket_type: str = "5d",
    ):
        self.data = data.astype(np.float32)
        self.close = close_prices.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.buckets = (
            ReturnBucketizer.BUCKETS_5D
            if bucket_type == "5d"
            else ReturnBucketizer.BUCKETS_20D
        )
        self.valid_len = len(self.data) - self.seq_len - self.pred_len + 1

    def __len__(self) -> int:
        return max(self.valid_len, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]

        future_close = self.close[idx + self.seq_len + self.pred_len - 1]
        current_close = self.close[idx + self.seq_len - 1]
        future_ret = future_close / current_close - 1

        y = ReturnBucketizer.bucketize(future_ret, self.buckets)
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray = None,
        seq_len: int = 60,
        pred_len: int = 5,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        self.data = data.astype(np.float32)

        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            self.targets = targets.astype(np.float32)
        else:
            self.targets = self.data[:, 0]  # default: close price column

        self.valid_len = len(self.data) - seq_len - pred_len + 1
        if self.valid_len <= 0:
            raise ValueError(
                f"Data length {len(self.data)} too short for seq_len={seq_len} + pred_len={pred_len}"
            )

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

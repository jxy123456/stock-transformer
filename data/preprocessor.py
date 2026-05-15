from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class Preprocessor:
    def __init__(self, config: dict = None):
        cfg = config or {}
        training = cfg.get("training", {})
        wf = training.get("walk_forward", {})
        self.train_window = wf.get("train_window", 500)
        self.val_window = wf.get("val_window", 100)
        self.test_window = wf.get("test_window", 50)
        self.step = wf.get("step", 50)
        self.normalization = training.get("normalization", "zscore")

        self._mean: np.ndarray = None
        self._std: np.ndarray = None

    def generate_folds(self, total_length: int) -> List[WalkForwardFold]:
        folds = []
        min_required = self.train_window + self.val_window + self.test_window
        if total_length < min_required:
            logger.warning(
                f"Data length {total_length} < min required {min_required}"
            )
            return folds

        start = 0
        fold_idx = 0
        while start + min_required <= total_length:
            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=start,
                train_end=start + self.train_window,
                val_start=start + self.train_window,
                val_end=start + self.train_window + self.val_window,
                test_start=start + self.train_window + self.val_window,
                test_end=start + self.train_window + self.val_window + self.test_window,
            )
            folds.append(fold)
            start += self.step
            fold_idx += 1

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds

    def fit(self, data: np.ndarray):
        if self.normalization == "zscore":
            self._mean = np.nanmean(data, axis=0)
            self._std = np.nanstd(data, axis=0)
            self._std[self._std < 1e-8] = 1.0
        elif self.normalization == "minmax":
            self._mean = np.nanmin(data, axis=0)
            self._std = np.nanmax(data, axis=0) - self._mean
            self._std[self._std < 1e-8] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return data
        return (data - self._mean) / self._std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return data
        return data * self._std + self._mean

    def prepare_fold_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fold: WalkForwardFold,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = df[feature_cols].values.astype(np.float32)

        train_data = data[fold.train_start : fold.train_end]
        val_data = data[fold.val_start : fold.val_end]
        test_data = data[fold.test_start : fold.test_end]

        self.fit(train_data)
        train_norm = self.transform(train_data)
        val_norm = self.transform(val_data)
        test_norm = self.transform(test_data)

        return train_norm, val_norm, test_norm

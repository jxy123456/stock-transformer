from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataQualityError(Exception):
    """Raised when data contains NaN or inf after all preprocessing steps."""


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
    """Walk-forward fold generation only. Normalization moved to DataPipeline."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        training = cfg.get("training", {})
        wf = training.get("walk_forward", {})
        self.train_window = wf.get("train_window", 500)
        self.val_window = wf.get("val_window", 100)
        self.test_window = wf.get("test_window", 50)
        self.step = wf.get("step", 50)

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


class DataPipeline:
    """6-step preprocessing pipeline.

    Steps 2-3 are data-independent (run once per stock).
    Steps 4-5 are fold-dependent (fit on train, transform on val/test).
    Step 6 is a validation gate that crashes on any remaining NaN/inf.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        prep = cfg.get("preprocessing", {})
        self.warmup_rows = prep.get("warmup_rows", 250)
        self.winsorize_sigma = prep.get("winsorize_sigma", 5.0)

        self._winsorize_bounds: dict = None
        self._zscore_mean: np.ndarray = None
        self._zscore_std: np.ndarray = None

    # ---- Step 2: inf → NaN ----

    def replace_inf_with_nan(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        result = df.copy()
        for col in feature_cols:
            if col not in result.columns:
                continue
            n_inf = int(np.isinf(result[col]).sum())
            if n_inf > 0:
                logger.warning(f"Column '{col}': {n_inf} inf values replaced with NaN")
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        return result

    # ---- Step 3a: drop warmup rows ----

    def drop_warmup_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.warmup_rows:
            raise DataQualityError(
                f"Data length {len(df)} <= warmup {self.warmup_rows}. "
                f"Need at least {self.warmup_rows + 1} rows."
            )
        logger.info(
            f"Dropping {self.warmup_rows} warmup rows ({len(df)} -> {len(df) - self.warmup_rows})"
        )
        return df.iloc[self.warmup_rows :].reset_index(drop=True)

    # ---- Step 3b: forward-fill conditional features ----

    def forward_fill_conditional(
        self, df: pd.DataFrame, conditional_cols: List[str]
    ) -> pd.DataFrame:
        existing = [c for c in conditional_cols if c in df.columns]
        if not existing:
            return df

        result = df.copy()
        result[existing] = result[existing].ffill()

        remaining_nan = result[existing].isna().any(axis=1)
        if remaining_nan.any():
            if remaining_nan.iloc[-1]:
                bad_cols = [
                    c for c in existing if result[c].isna().iloc[-1]
                ]
                raise DataQualityError(
                    f"NaN in conditional features at end of series after ffill: {bad_cols}"
                )
            first_valid = remaining_nan.idxmin()
            n_trimmed = first_valid
            result = result.iloc[first_valid:].reset_index(drop=True)
            logger.info(
                f"Trimmed {n_trimmed} leading rows with missing conditional features"
            )

        return result

    # ---- Step 3c: assert always-available features are clean ----

    def assert_always_available_clean(
        self, df: pd.DataFrame, always_available_cols: List[str]
    ) -> None:
        existing = [c for c in always_available_cols if c in df.columns]
        if not existing:
            return
        nan_counts = df[existing].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if not bad.empty:
            raise DataQualityError(
                f"NaN in always-available features after warmup drop: {bad.to_dict()}"
            )

    # ---- Step 4: winsorization ----

    def fit_winsorize(
        self, train_df: pd.DataFrame, feature_cols: List[str]
    ) -> dict:
        bounds = {}
        for col in feature_cols:
            if col not in train_df.columns:
                continue
            series = train_df[col].dropna()
            if len(series) < 2:
                bounds[col] = (None, None)
                continue
            mean = series.mean()
            std = series.std()
            if std < 1e-8:
                bounds[col] = (None, None)
                continue
            lower = mean - self.winsorize_sigma * std
            upper = mean + self.winsorize_sigma * std
            bounds[col] = (lower, upper)
        self._winsorize_bounds = bounds
        n_clipped = sum(1 for v in bounds.values() if v[0] is not None)
        logger.info(
            f"Winsorize: {n_clipped}/{len(feature_cols)} features clipped at ±{self.winsorize_sigma}σ"
        )
        return bounds

    def transform_winsorize(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        result = df.copy()
        for col in feature_cols:
            if col not in result.columns or col not in self._winsorize_bounds:
                continue
            lower, upper = self._winsorize_bounds[col]
            if lower is not None and upper is not None:
                result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    # ---- Step 5: z-score normalization ----

    def fit_zscore(self, train_data: np.ndarray) -> None:
        self._zscore_mean = np.nanmean(train_data, axis=0)
        self._zscore_std = np.nanstd(train_data, axis=0)
        zero_std_mask = self._zscore_std < 1e-8
        if zero_std_mask.any():
            zero_cols = np.where(zero_std_mask)[0].tolist()
            raise DataQualityError(
                f"Near-zero variance in feature columns {zero_cols} "
                f"after warmup drop. These features carry no information."
            )

    def transform_zscore(self, data: np.ndarray) -> np.ndarray:
        return (data - self._zscore_mean) / self._zscore_std

    # ---- Step 6: final data quality assert ----

    def assert_clean(self, data: np.ndarray, label: str) -> None:
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        n_nan = int(nan_mask.sum())
        n_inf = int(inf_mask.sum())
        if n_nan > 0 or n_inf > 0:
            nan_per_col = nan_mask.sum(axis=0)
            inf_per_col = inf_mask.sum(axis=0)
            bad = []
            for i in range(data.shape[1]):
                if nan_per_col[i] > 0 or inf_per_col[i] > 0:
                    bad.append(f"col_{i}(nan={nan_per_col[i]},inf={inf_per_col[i]})")
            raise DataQualityError(
                f"{label}: {n_nan} NaN, {n_inf} inf remain after preprocessing. "
                f"Bad columns: {bad[:10]}"
            )

    # ---- Orchestration: data-independent steps (per stock) ----

    def preprocess_stock(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        always_available_cols: List[str],
        conditional_cols: List[str],
    ) -> pd.DataFrame:
        """Run Steps 2-3 for one stock. Returns cleaned DataFrame."""
        df = self.replace_inf_with_nan(df, feature_cols)
        df = self.drop_warmup_rows(df)
        df = self.forward_fill_conditional(df, conditional_cols)
        self.assert_always_available_clean(df, always_available_cols)
        return df

    # ---- Orchestration: fold-dependent steps ----

    def prepare_fold(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fold: WalkForwardFold,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Steps 4-6 for one fold. Returns (train_norm, val_norm, test_norm)."""
        train_df = df.iloc[fold.train_start : fold.train_end]
        val_df = df.iloc[fold.val_start : fold.val_end]
        test_df = df.iloc[fold.test_start : fold.test_end]

        # Step 4: winsorize (fit on train)
        self.fit_winsorize(train_df, feature_cols)
        train_df = self.transform_winsorize(train_df, feature_cols)
        val_df = self.transform_winsorize(val_df, feature_cols)
        test_df = self.transform_winsorize(test_df, feature_cols)

        # Step 5: z-score (fit on train)
        train_data = train_df[feature_cols].values.astype(np.float32)
        val_data = val_df[feature_cols].values.astype(np.float32)
        test_data = test_df[feature_cols].values.astype(np.float32)

        self.fit_zscore(train_data)
        train_norm = self.transform_zscore(train_data)
        val_norm = self.transform_zscore(val_data)
        test_norm = self.transform_zscore(test_data)

        # Step 6: final assert
        self.assert_clean(train_norm, f"fold-{fold.fold_idx}/train")
        self.assert_clean(val_norm, f"fold-{fold.fold_idx}/val")
        self.assert_clean(test_norm, f"fold-{fold.fold_idx}/test")

        return train_norm, val_norm, test_norm

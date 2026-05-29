"""特征引擎抽象基类。"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


FEATURE_CACHE_VERSION = "v3_no_industry"


class BaseFeatureEngine(ABC):
    """特征计算接口：输入原始数据，输出特征DataFrame。

    新增特征方案 = 继承此类，实现 compute() 和 compute_batch()。
    """

    def __init__(self, config: dict, cache):
        self.config = config
        self.cache = cache

    @property
    @abstractmethod
    def feature_columns(self) -> list:
        """返回特征列名列表，顺序固定。"""
        ...

    @abstractmethod
    def compute(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """计算单只股票的特征。返回 DataFrame，index 为日期。"""
        ...

    @abstractmethod
    def compute_batch(
        self, symbols: list, date: str
    ) -> dict:
        """批量计算截面特征（rank类）。返回 {symbol: Series}。"""
        ...

    def save(self, symbol: str, df: pd.DataFrame):
        path = Path("outputs/features") / f"{symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        (path.parent / "_feature_cache_version.txt").write_text(FEATURE_CACHE_VERSION)

    def load(self, symbol: str) -> pd.DataFrame:
        path = Path("outputs/features") / f"{symbol}.parquet"
        version_path = path.parent / "_feature_cache_version.txt"
        if not version_path.exists() or version_path.read_text().strip() != FEATURE_CACHE_VERSION:
            return pd.DataFrame()
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

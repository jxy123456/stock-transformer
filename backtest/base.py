"""回测抽象基类。"""

from abc import ABC, abstractmethod


class BaseBacktest(ABC):
    @abstractmethod
    def run(self, model, data: dict, config: dict) -> dict:
        """返回 metrics 字典。"""
        ...

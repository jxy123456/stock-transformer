"""模型抽象基类。"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """所有预测模型的基类。forward 必须返回 logits_1d, logits_5d, logits_20d。"""

    @abstractmethod
    def forward(self, x, padding_mask=None) -> dict:
        ...

from typing import List

import numpy as np
import torch
import torch.nn as nn


class EnsembleStrategy:
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, x: torch.Tensor) -> np.ndarray:
        predictions = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                pred = model(x).cpu().numpy()
                predictions.append(weight * pred)
        return sum(predictions)

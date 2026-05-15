from collections import namedtuple
from typing import List

import numpy as np

Signal = namedtuple("Signal", ["symbol", "direction", "strength", "price"])


class SignalGenerator:
    def __init__(self, config: dict = None):
        cfg = config or {}
        strategy = cfg.get("strategy", {})
        self.buy_threshold = strategy.get("buy_threshold", 0.005)
        self.sell_threshold = strategy.get("sell_threshold", -0.005)

    def generate_signals(
        self,
        predictions: dict,
        current_positions: dict,
        current_prices: dict,
    ) -> List[Signal]:
        signals = []
        for symbol, pred_return in predictions.items():
            if isinstance(pred_return, np.ndarray):
                pred_return = pred_return[0]  # use first day prediction

            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            is_long = symbol in current_positions

            if pred_return > self.buy_threshold and not is_long:
                strength = min(pred_return / self.buy_threshold, 2.0)
                signals.append(Signal(symbol, 1, strength, price))

            elif pred_return < self.sell_threshold and is_long:
                strength = min(abs(pred_return / self.sell_threshold), 2.0)
                signals.append(Signal(symbol, -1, strength, price))

        return signals

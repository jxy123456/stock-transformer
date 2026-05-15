import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngine:
    def __init__(self, config: dict = None):
        cfg = config or {}
        feat = cfg.get("features", {})
        self.ma_periods = feat.get("ma_periods", [5, 10, 20, 60])
        self.ema_periods = feat.get("ema_periods", [12, 26])
        self.macd_fast = feat.get("macd_fast", 12)
        self.macd_slow = feat.get("macd_slow", 26)
        self.macd_signal = feat.get("macd_signal", 9)
        self.rsi_period = feat.get("rsi_period", 14)
        self.bollinger_period = feat.get("bollinger_period", 20)
        self.bollinger_std = feat.get("bollinger_std", 2.0)
        self.atr_period = feat.get("atr_period", 14)
        self.volume_ma_period = feat.get("volume_ma_period", 20)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()
        c = result["close"]
        h = result["high"]
        l = result["low"]
        v = result["volume"]

        # --- Moving Averages ---
        for period in self.ma_periods:
            result[f"ma_{period}"] = c.rolling(period).mean()
            result[f"close_ma{period}_ratio"] = c / result[f"ma_{period}"] - 1

        for period in self.ema_periods:
            result[f"ema_{period}"] = c.ewm(span=period, adjust=False).mean()

        # --- MACD ---
        ema_fast = c.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = c.ewm(span=self.macd_slow, adjust=False).mean()
        result["macd_dif"] = ema_fast - ema_slow
        result["macd_dea"] = result["macd_dif"].ewm(
            span=self.macd_signal, adjust=False
        ).mean()
        result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

        # --- RSI ---
        result["rsi"] = self._compute_rsi(c, self.rsi_period)

        # --- Bollinger Bands ---
        bb_mid = c.rolling(self.bollinger_period).mean()
        bb_std = c.rolling(self.bollinger_period).std()
        result["bb_upper"] = bb_mid + self.bollinger_std * bb_std
        result["bb_lower"] = bb_mid - self.bollinger_std * bb_std
        result["bb_pct"] = (c - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )

        # --- ATR ---
        result["atr"] = self._compute_atr(h, l, c, self.atr_period)

        # --- Volume features ---
        vol_ma = v.rolling(self.volume_ma_period).mean()
        result["volume_ratio"] = v / vol_ma
        result["volume_log"] = np.log1p(v)

        # --- Returns ---
        result["return_1d"] = c.pct_change(1)
        result["return_5d"] = c.pct_change(5)
        result["return_20d"] = c.pct_change(20)

        # --- Volatility ---
        result["vol_5d"] = result["return_1d"].rolling(5).std()
        result["vol_20d"] = result["return_1d"].rolling(20).std()

        # --- Intra-day features ---
        result["amplitude"] = (h - l) / c.shift(1)
        result["oc_ratio"] = (result["open"] - c.shift(1)) / c.shift(1)

        return result

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return atr

    @property
    def feature_columns(self) -> list:
        features = ["open", "high", "low", "close", "volume"]
        for p in self.ma_periods:
            features += [f"ma_{p}", f"close_ma{p}_ratio"]
        for p in self.ema_periods:
            features.append(f"ema_{p}")
        features += [
            "macd_dif", "macd_dea", "macd_hist",
            "rsi",
            "bb_upper", "bb_lower", "bb_pct",
            "atr",
            "volume_ratio", "volume_log",
            "return_1d", "return_5d", "return_20d",
            "vol_5d", "vol_20d",
            "amplitude", "oc_ratio",
        ]
        return features

import numpy as np
import pandas as pd


class ValuationFeatureEngine:
    """5 valuation features from daily_basic (PE/PB/PS/market_cap)."""

    def compute(self, stock_df: pd.DataFrame, valuation_df: pd.DataFrame) -> pd.DataFrame:
        result = stock_df.copy()

        val = valuation_df.set_index("datetime")
        val = val.reindex(result["datetime"], method="ffill")

        pe = val.get("pe", pd.Series(np.nan, index=result.index))
        pb = val.get("pb", pd.Series(np.nan, index=result.index))
        ps = val.get("ps", pd.Series(np.nan, index=result.index))

        result["earnings_yield"] = 1.0 / pe.replace(0, np.nan)
        result["book_to_price"] = 1.0 / pb.replace(0, np.nan)
        result["sales_to_price"] = 1.0 / ps.replace(0, np.nan)

        result["pe_percentile_3y"] = pe.rolling(750, min_periods=250).rank(pct=True)
        result["pb_percentile_3y"] = pb.rolling(750, min_periods=250).rank(pct=True)

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "earnings_yield", "book_to_price", "sales_to_price",
            "pe_percentile_3y", "pb_percentile_3y",
        ]

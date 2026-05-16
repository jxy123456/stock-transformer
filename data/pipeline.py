import pandas as pd
from loguru import logger

from data.features import FeatureEngine
from data.liquidity import LiquidityFeatureEngine
from data.fundamental import FundamentalEngine
from data.market import MarketFeatureEngine
from data.valuation import ValuationFeatureEngine
from data.cross_sectional import CrossSectionalEngine
from data.industry import IndustryFeatureEngine
from data.cache import DataCache
from data.fetcher import AShareFetcher


class FeaturePipeline:
    """Orchestrates all feature engines for the 80-feature system."""

    def __init__(self, config: dict, cache: DataCache = None, fetcher: AShareFetcher = None):
        self.config = config
        self.tech_engine = FeatureEngine()
        self.liq_engine = LiquidityFeatureEngine()
        self.val_engine = ValuationFeatureEngine()
        self.fund_engine = FundamentalEngine(fetcher, cache)

        self._market_engine = None
        self._cs_engine = None
        self._ind_engine = None

    def setup(
        self,
        index_data: pd.DataFrame,
        industry_map: dict,
        industry_returns: dict,
    ):
        """Initialize engines that require external data."""
        self._market_engine = MarketFeatureEngine(index_data)
        self._cs_engine = CrossSectionalEngine(industry_map)
        self._ind_engine = IndustryFeatureEngine(industry_map, industry_returns)

    def compute_single(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        valuation_df: pd.DataFrame = None,
        financial_df: pd.DataFrame = None,
        income_df: pd.DataFrame = None,
        balance_df: pd.DataFrame = None,
        cashflow_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Compute all features for one stock."""
        df = self.tech_engine.compute(stock_df)
        df = self.liq_engine.compute(df)

        if self._market_engine is not None:
            df = self._market_engine.compute(df)

        if valuation_df is not None and not valuation_df.empty:
            df = self.val_engine.compute(df, valuation_df)

        df = self.fund_engine.compute(df, financial_df, income_df, balance_df, cashflow_df)

        if self._ind_engine is not None:
            df = self._ind_engine.compute(df, symbol)

        return df

    def compute_cross_sectional(self, all_stock_dfs: dict) -> dict:
        """Compute cross-sectional features across all stocks."""
        if self._cs_engine is not None:
            return self._cs_engine.compute_batch(all_stock_dfs)
        return all_stock_dfs

    @property
    def feature_columns(self) -> list:
        cols = self.tech_engine.feature_columns
        cols += self.liq_engine.feature_columns
        if self._market_engine is not None:
            cols += self._market_engine.feature_columns
        cols += self.val_engine.feature_columns
        cols += self.fund_engine.feature_columns
        if self._ind_engine is not None:
            cols += self._ind_engine.feature_columns
        if self._cs_engine is not None:
            cols += self._cs_engine.feature_columns
        return cols

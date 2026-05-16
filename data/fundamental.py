import numpy as np
import pandas as pd
from loguru import logger

from data.fetcher import AShareFetcher
from data.cache import DataCache


class FundamentalEngine:
    """16 financial features: 10 from fina_indicator + 6 from detailed statements."""

    def __init__(self, fetcher: AShareFetcher = None, cache: DataCache = None):
        self.fetcher = fetcher or AShareFetcher()
        self.cache = cache

    def compute(
        self,
        stock_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        income_df: pd.DataFrame = None,
        balance_df: pd.DataFrame = None,
        cashflow_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        result = stock_df.copy()

        # ---- From fina_indicator (10 features) ----
        if financial_df.empty:
            for col in self._fina_indicator_cols:
                result[col] = np.nan
        else:
            fin_daily = self._align_to_daily(financial_df, result)

            roe = fin_daily.get("roe_dt", fin_daily.get("roe", pd.Series(np.nan, index=result.index)))
            result["roe_ttm"] = roe.values

            result["gross_margin_ttm"] = fin_daily.get("grossprofit_margin", pd.Series(np.nan, index=result.index)).values
            result["net_margin_ttm"] = fin_daily.get("netprofit_margin", pd.Series(np.nan, index=result.index)).values

            revenue_yoy = fin_daily.get("revenue_yoy", pd.Series(np.nan, index=result.index))
            profit_yoy = fin_daily.get("profit_yoy", pd.Series(np.nan, index=result.index))
            result["revenue_yoy"] = revenue_yoy.values
            result["net_profit_yoy"] = profit_yoy.values
            result["revenue_yoy_acceleration"] = revenue_yoy.diff().values
            result["profit_yoy_acceleration"] = profit_yoy.diff().values

            result["ocf_to_net_profit"] = fin_daily.get("ocf_to_netprofit", pd.Series(np.nan, index=result.index)).values
            result["debt_to_asset"] = fin_daily.get("debt_to_assets", pd.Series(np.nan, index=result.index)).values
            result["current_ratio"] = fin_daily.get("current_ratio", pd.Series(np.nan, index=result.index)).values

        # ---- From detailed statements (6 features) ----
        has_detail = (
            income_df is not None and not income_df.empty
            and balance_df is not None and not balance_df.empty
            and cashflow_df is not None and not cashflow_df.empty
        )

        if not has_detail:
            for col in self._detail_cols:
                result[col] = np.nan
        else:
            inc_daily = self._align_to_daily(income_df, result)
            bal_daily = self._align_to_daily(balance_df, result)
            cf_daily = self._align_to_daily(cashflow_df, result)

            total_assets = bal_daily.get("total_assets", pd.Series(np.nan, index=result.index))
            equity = bal_daily.get("total_hldr_eqy_exc_min_int", pd.Series(np.nan, index=result.index))
            ocf = cf_daily.get("n_cashflow_act", pd.Series(np.nan, index=result.index))
            revenue = inc_daily.get("total_revenue", pd.Series(np.nan, index=result.index))
            netprofit = inc_daily.get("netprofit", pd.Series(np.nan, index=result.index))
            inventories = bal_daily.get("inventories", pd.Series(np.nan, index=result.index))
            accounts_rec = bal_daily.get("accounts_rec", pd.Series(np.nan, index=result.index))

            result["roa_ttm"] = (netprofit / (total_assets + 1e-8)).values
            result["ocf_to_revenue"] = (ocf / (revenue.abs() + 1e-8)).values
            result["accrual_to_assets"] = ((netprofit - ocf) / (total_assets.abs() + 1e-8)).values
            result["equity_ratio"] = (equity / (total_assets + 1e-8)).values

            inv_growth = inventories.pct_change(4)
            rec_growth = accounts_rec.pct_change(4)
            rev_growth = revenue.pct_change(4)
            result["inventory_growth_minus_revenue"] = (inv_growth - rev_growth).values
            result["receivable_growth_minus_revenue"] = (rec_growth - rev_growth).values

        return result

    _fina_indicator_cols = [
        "roe_ttm", "gross_margin_ttm", "net_margin_ttm",
        "revenue_yoy", "net_profit_yoy",
        "revenue_yoy_acceleration", "profit_yoy_acceleration",
        "ocf_to_net_profit", "debt_to_asset", "current_ratio",
    ]

    _detail_cols = [
        "roa_ttm", "ocf_to_revenue", "accrual_to_assets", "equity_ratio",
        "inventory_growth_minus_revenue", "receivable_growth_minus_revenue",
    ]

    @property
    def feature_columns(self) -> list:
        return self._fina_indicator_cols + self._detail_cols

    @staticmethod
    def _align_to_daily(financial: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        if financial.empty or daily.empty:
            return pd.DataFrame(index=daily.index)

        date_col = "report_date" if "report_date" in financial.columns else "end_period"
        if date_col not in financial.columns:
            return pd.DataFrame(index=daily.index)

        fin = financial.copy()
        fin[date_col] = pd.to_datetime(fin[date_col])
        fin = fin.set_index(date_col).sort_index()
        fin = fin[~fin.index.duplicated(keep="last")]

        daily_dates = pd.to_datetime(daily["datetime"])
        aligned = fin.reindex(daily_dates, method="ffill")
        return aligned

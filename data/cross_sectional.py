import numpy as np
import pandas as pd
from scipy.stats import rankdata


class CrossSectionalEngine:
    """9 cross-sectional rank features + market_up_stock_ratio_1d."""

    def __init__(self, industry_map: dict):
        self.industry_map = industry_map

    def compute_batch(
        self,
        all_stock_features: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Compute cross-sectional features for all stocks.

        For each trading date, rank metrics across all stocks and within industry.
        Also compute market_up_stock_ratio_1d (fraction of stocks with positive 1d return).
        """
        metrics_needed = [
            "log_ret_20d", "log_ret_60d", "realized_vol_60d",
            "log_amount", "market_cap",
            "roe_ttm", "earnings_yield", "book_to_price",
            "debt_to_asset",
        ]

        # Build panel: (symbol, metric) -> Series indexed by datetime
        panels = {}
        for symbol, df in all_stock_features.items():
            if "datetime" not in df.columns:
                continue
            for metric in metrics_needed:
                if metric not in df.columns:
                    continue
                key = (symbol, metric)
                panels[key] = df.set_index("datetime")[metric]

        # Get union of all dates
        all_dates = set()
        for symbol, df in all_stock_features.items():
            if "datetime" in df.columns:
                all_dates.update(df["datetime"].values)
        all_dates = sorted(all_dates)

        # For each date, compute ranks
        rank_results = {symbol: {} for symbol in all_stock_features}

        for date in all_dates:
            # Collect values for this date
            date_vals = {}
            for symbol in all_stock_features:
                for metric in metrics_needed:
                    key = (symbol, metric)
                    if key in panels and date in panels[key].index:
                        val = panels[key].loc[date]
                        if not np.isnan(val):
                            date_vals.setdefault(metric, {})[symbol] = val

            # Rank across all stocks
            for metric, vals in date_vals.items():
                if metric == "debt_to_asset":
                    continue  # handled in industry ranking
                symbols_list = list(vals.keys())
                values = np.array([vals[s] for s in symbols_list])
                ranks = rankdata(values, method="average") / len(values)
                for sym, rank in zip(symbols_list, ranks):
                    rank_results[sym].setdefault(date, {})[f"rank_{metric}_all"] = rank

            # Rank within industry
            for metric in ["roe_ttm", "earnings_yield", "book_to_price"]:
                if metric not in date_vals:
                    continue
                by_industry = {}
                for sym, val in date_vals[metric].items():
                    ind = self.industry_map.get(sym, "unknown")
                    by_industry.setdefault(ind, {})[sym] = val

                for industry, ind_vals in by_industry.items():
                    if len(ind_vals) < 3:
                        continue
                    syms = list(ind_vals.keys())
                    vals = np.array([ind_vals[s] for s in syms])
                    ranks = rankdata(vals, method="average") / len(vals)
                    for sym, rank in zip(syms, ranks):
                        rank_results[sym].setdefault(date, {})[f"rank_{metric}_industry"] = rank

            # rank_debt_to_asset_industry_reverse: low debt = high rank
            if "debt_to_asset" in date_vals:
                by_industry = {}
                for sym, val in date_vals["debt_to_asset"].items():
                    ind = self.industry_map.get(sym, "unknown")
                    by_industry.setdefault(ind, {})[sym] = val

                for industry, ind_vals in by_industry.items():
                    if len(ind_vals) < 3:
                        continue
                    syms = list(ind_vals.keys())
                    vals = np.array([ind_vals[s] for s in syms])
                    ranks = rankdata(-vals, method="average") / len(vals)
                    for sym, rank in zip(syms, ranks):
                        rank_results[sym].setdefault(date, {})[
                            "rank_debt_to_asset_industry_reverse"
                        ] = rank

            # market_up_stock_ratio_1d: fraction of stocks with positive 1d return
            if "log_ret_1d" in date_vals:
                ret_vals = date_vals["log_ret_1d"]
                n_up = sum(1 for v in ret_vals.values() if v > 0)
                n_total = len(ret_vals)
                up_ratio = n_up / n_total if n_total > 0 else np.nan
                for sym in all_stock_features:
                    rank_results[sym].setdefault(date, {})[
                        "market_up_stock_ratio_1d"
                    ] = up_ratio

        # Write rank columns back to each stock's DataFrame
        result = {}
        all_rank_cols = self.feature_columns + ["market_up_stock_ratio_1d"]
        for symbol, df in all_stock_features.items():
            df_out = df.copy()
            for col in all_rank_cols:
                df_out[col] = np.nan

            for date, ranks in rank_results.get(symbol, {}).items():
                mask = df_out["datetime"] == date
                for col, val in ranks.items():
                    if col in all_rank_cols:
                        df_out.loc[mask, col] = val

            result[symbol] = df_out

        return result

    @property
    def feature_columns(self) -> list:
        return [
            "rank_log_ret_20d_all", "rank_log_ret_60d_all",
            "rank_realized_vol_60d_all", "rank_log_amount_all",
            "rank_market_cap_all",
            "rank_roe_ttm_industry",
            "rank_earnings_yield_industry", "rank_book_to_price_industry",
            "rank_debt_to_asset_industry_reverse",
        ]

"""单独回测入口。

用法:
  python experiments/backtest.py --config baseline --checkpoint outputs/checkpoints/baseline_fold0_best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from pipeline.config import load_experiment
from pipeline.data_pipeline import FEATURE_CACHE_VERSION
from pipeline.factory import create_model, load_checkpoint
from backtest.engine_v2 import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger("backtest")
    config = load_experiment(args.config)

    # ---- load norm stats ----
    norm_path = Path("outputs/datasets/norm_stats.json")
    norm_stats = json.loads(norm_path.read_text()) if norm_path.exists() else {}

    # ---- load features ----
    feat_dir = Path("outputs/features")
    version_path = feat_dir / "_feature_cache_version.txt"
    if not version_path.exists() or version_path.read_text().strip() != FEATURE_CACHE_VERSION:
        raise RuntimeError("Feature cache is stale. Run pipeline/data_pipeline.py before backtesting.")
    feature_dfs = {}
    stock_list = config.get("data", {}).get("stock_list", "liquid100")
    from data.stock_selector import load_symbols
    symbols = load_symbols(stock_list)

    logger.info(f"Loading features for {len(symbols)} stocks...")
    for s in symbols:
        fpath = feat_dir / f"{s}.parquet"
        if fpath.exists():
            feature_dfs[s] = pd.read_parquet(fpath)

    # ensure 'open' column exists (merge from daily cache)
    cache_dir = config.get("data", {}).get("cache_dir", "outputs/data_cache")
    for s, df in list(feature_dfs.items()):
        if "open" not in df.columns:
            daily = Path(cache_dir) / f"daily_{s}.parquet"
            if daily.exists():
                daily_df = pd.read_parquet(daily)
                if "datetime" in daily_df.columns and "open" in daily_df.columns:
                    daily_df["datetime"] = pd.to_datetime(daily_df["datetime"])
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    o_map = daily_df.set_index("datetime")["open"]
                    dates = df["datetime"].values
                    open_vals = [float(o_map.get(d, np.nan)) for d in dates]
                    df["open"] = open_vals
                else:
                    df["open"] = df["close"]
            else:
                df["open"] = df["close"]

    logger.info(f"Loaded {len(feature_dfs)} stocks with features")

    # ---- create model ----
    model = create_model(config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # use checkpoint's norm stats if available (prefer over norm_stats.json)
    if "mean" in ckpt:
        norm_stats = {"feature_columns": ckpt.get("feature_columns", []),
                       "mean": ckpt["mean"], "std": ckpt["std"]}

    # ---- run backtest ----
    engine = BacktestEngine(config)
    metrics, equity, trades = engine.run(model, feature_dfs, norm_stats)

    # ---- save ----
    out = Path("outputs/results") / config.get("name", "backtest")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    eq_df = pd.DataFrame(equity, columns=["date", "value"])
    eq_df.to_csv(out / "equity_curve.csv", index=False)

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(out / "trades.csv", index=False)

    logger.info(f"\n=== 回测结果 ===")
    logger.info(f"  总收益:     {metrics.get('total_return', 0):.2%}")
    logger.info(f"  年化收益:   {metrics.get('annualized_return', 0):.2%}")
    logger.info(f"  Sharpe:    {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  最大回撤:   {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  胜率:       {metrics.get('win_rate', 0):.2%}")
    logger.info(f"  交易次数:   {metrics.get('n_trades', 0)}")
    logger.info(f"  最终净值:   {metrics.get('final_value', 0):,.0f}")
    logger.info(f"\n结果保存至: {out}")


if __name__ == "__main__":
    main()

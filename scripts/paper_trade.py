import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--symbols", nargs="+", default=["000001"], help="Stock symbols")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    logger = setup_logger("paper_trade")

    import torch
    from model.transformer import StockTransformer
    from model.checkpoint import CheckpointManager
    from data.features import FeatureEngine
    from execution.paper_trader import PaperTrader

    feature_engine = FeatureEngine(config.raw)
    model_cfg = config.raw.get("model", {})
    feature_cols = feature_engine.feature_columns

    model = StockTransformer(
        num_features=len(feature_cols),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 8),
        n_layers=model_cfg.get("n_layers", 4),
        d_ff=model_cfg.get("d_ff", 512),
        pred_len=model_cfg.get("pred_len", 5),
        dropout=0.0,  # no dropout in inference
        seq_len=model_cfg.get("seq_len", 60),
    )

    ckpt_mgr = CheckpointManager()
    ckpt_mgr.load(model, args.checkpoint)

    trader = PaperTrader(model, config.raw, feature_engine)

    from datetime import date
    today = date.today()
    trader.run_daily(args.symbols, today)

    logger.info(f"Portfolio: cash={trader.portfolio.cash:,.2f}, total={trader.portfolio.total_value:,.2f}")


if __name__ == "__main__":
    main()

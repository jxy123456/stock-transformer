"""实验入口。

用法:
  python -m experiments.run --config baseline
  python -m experiments.run --config baseline --backtest-only
  python -m experiments.run --config baseline --eval-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--config", type=str, required=True, help="Experiment name (e.g. baseline)")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_logger(f"exp_{args.config}")

    from pipeline.runner import run_experiment
    run_experiment(args.config, backtest_only=args.backtest_only,
                   eval_only=args.eval_only)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-darkgrid")


def plot_equity_curve(
    equity_curve: List[Tuple],
    benchmark_curve: List[Tuple] = None,
    title: str = "Equity Curve",
    save_path: str = None,
):
    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    ax1.plot(dates, values, label="Strategy", linewidth=1.5)
    if benchmark_curve:
        bm_dates = [e[0] for e in benchmark_curve]
        bm_values = [e[1] for e in benchmark_curve]
        ax1.plot(bm_dates, bm_values, label="Benchmark", alpha=0.7, linewidth=1)

    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax1.set_title(title)

    # Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = [(peak[i] - values[i]) / peak[i] for i in range(len(values))]
    ax2.fill_between(dates, drawdown, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(
    dates: list,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predicted vs Actual Returns",
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, label="Actual", alpha=0.7, linewidth=1)
    ax.plot(dates, predicted, label="Predicted", alpha=0.7, linewidth=1)
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_attention_weights(
    weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(
    train_loss: list,
    val_loss: list,
    title: str = "Training History",
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label="Train Loss")
    ax.plot(val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title(title)
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trade_signals(
    df: pd.DataFrame,
    trades: list,
    title: str = "Trade Signals",
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["datetime"], df["close"], linewidth=1, alpha=0.8)

    for trade in trades:
        if "date" not in trade or "direction" not in trade:
            continue
        color = "green" if trade["direction"] == "BUY" else "red"
        marker = "^" if trade["direction"] == "BUY" else "v"
        ax.scatter(trade["date"], trade["price"], color=color, marker=marker, s=100, zorder=5)

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

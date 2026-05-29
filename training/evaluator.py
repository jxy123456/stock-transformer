"""评估指标：Rank IC、分组收益、方向准确率。"""

import numpy as np
from scipy import stats
from scipy.special import softmax

from data.features.v1_45 import CENTERS_1D, CENTERS_5D, CENTERS_20D


def expected_returns(logits, centers):
    probs = softmax(logits, axis=-1)
    return probs @ centers


def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation。"""
    if len(pred) < 3 or np.std(pred) < 1e-8 or np.std(actual) < 1e-8:
        return 0.0
    ic, _ = stats.spearmanr(pred, actual)
    return float(ic) if not np.isnan(ic) else 0.0


def directional_accuracy(pred_sign: np.ndarray, actual_sign: np.ndarray) -> float:
    mask = actual_sign != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(pred_sign[mask] == actual_sign[mask]))


def group_returns(scores: np.ndarray, actual_rets: np.ndarray, n_groups=5) -> dict:
    """分组收益：按 score 从低到高分成 n_groups 组，计算每组平均收益。"""
    order = np.argsort(scores)
    group_size = len(order) // n_groups
    result = {}
    for g in range(n_groups):
        idx = order[g * group_size: (g + 1) * group_size] if g < n_groups - 1 else order[g * group_size:]
        result[f"group_{g+1}"] = float(np.mean(actual_rets[idx]))
    result["long_short"] = result[f"group_{n_groups}"] - result["group_1"]
    return result


def compute_metrics(logits_1d, logits_5d, logits_20d,
                    actual_1d, actual_5d, actual_20d) -> dict:
    """多周期综合评估。"""
    er_1d = expected_returns(logits_1d, CENTERS_1D)
    er_5d = expected_returns(logits_5d, CENTERS_5D)
    er_20d = expected_returns(logits_20d, CENTERS_20D)

    score = 0.2 * er_1d + 0.5 * er_5d + 0.3 * er_20d

    metrics = {}
    for name, pred, actual, centers in [
        ("1d", er_1d, actual_1d, CENTERS_1D),
        ("5d", er_5d, actual_5d, CENTERS_5D),
        ("20d", er_20d, actual_20d, CENTERS_20D),
    ]:
        metrics[f"rank_ic_{name}"] = rank_ic(pred, actual)
        metrics[f"dir_acc_{name}"] = directional_accuracy(np.sign(pred), np.sign(actual))
        metrics[f"mean_er_{name}"] = float(np.mean(pred))

    metrics["rank_ic_score"] = rank_ic(score, actual_5d)
    metrics["group_returns"] = group_returns(score, actual_5d)
    return metrics

import numpy as np
from scipy import stats


class Evaluator:
    @staticmethod
    def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
        pred = predictions.flatten()
        actual = actuals.flatten()

        # MSE / MAE / RMSE
        mse = np.mean((pred - actual) ** 2)
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(mse)

        # Directional accuracy
        pred_sign = np.sign(pred)
        actual_sign = np.sign(actual)
        non_zero = actual_sign != 0
        if non_zero.sum() > 0:
            dir_acc = np.mean(pred_sign[non_zero] == actual_sign[non_zero])
        else:
            dir_acc = 0.0

        # IC (Pearson correlation)
        if len(pred) > 2 and np.std(pred) > 1e-8 and np.std(actual) > 1e-8:
            ic, _ = stats.pearsonr(pred, actual)
        else:
            ic = 0.0

        # Rank IC (Spearman correlation)
        if len(pred) > 2:
            rank_ic, _ = stats.spearmanr(pred, actual)
        else:
            rank_ic = 0.0

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "directional_accuracy": float(dir_acc),
            "ic": float(ic),
            "rank_ic": float(rank_ic),
        }

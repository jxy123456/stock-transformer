import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


class Evaluator:
    @staticmethod
    def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
        pred = predictions.flatten()
        actual = actuals.flatten()

        mse = np.mean((pred - actual) ** 2)
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(mse)

        pred_sign = np.sign(pred)
        actual_sign = np.sign(actual)
        non_zero = actual_sign != 0
        if non_zero.sum() > 0:
            dir_acc = np.mean(pred_sign[non_zero] == actual_sign[non_zero])
        else:
            dir_acc = 0.0

        if len(pred) > 2 and np.std(pred) > 1e-8 and np.std(actual) > 1e-8:
            ic, _ = stats.pearsonr(pred, actual)
        else:
            ic = 0.0

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

    @staticmethod
    def compute_classification_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
        bucket_boundaries: list,
    ) -> dict:
        """Classification metrics for return distribution prediction.

        Args:
            logits: (N, num_classes) raw model outputs
            targets: (N,) int bucket indices
            bucket_boundaries: list of (lo, hi) tuples
        """
        from scipy.special import softmax

        probs = softmax(logits, axis=1)
        pred_classes = np.argmax(probs, axis=1)

        # Top-1 accuracy
        top1_acc = (pred_classes == targets).mean()

        # Top-2 accuracy
        top2_preds = np.argsort(probs, axis=1)[:, -2:]
        top2_acc = np.mean([t in top2 for t, top2 in zip(targets, top2_preds)])

        # Weighted F1
        f1 = f1_score(targets, pred_classes, average="weighted", zero_division=0)

        # Expected return from predicted distribution
        centers = np.array([(lo + hi) / 2 if lo != -np.inf and hi != np.inf
                            else (hi * 0.75 if lo == -np.inf else lo * 0.75)
                            for lo, hi in bucket_boundaries])
        expected_returns = probs @ centers

        # Actual returns (use bucket center as proxy)
        actual_returns = centers[targets]

        # IC between expected and actual
        if np.std(expected_returns) > 1e-8 and np.std(actual_returns) > 1e-8:
            ic, _ = stats.pearsonr(expected_returns, actual_returns)
        else:
            ic = 0.0

        # Directional accuracy: do expected and actual agree on sign?
        exp_sign = np.sign(expected_returns)
        act_sign = np.sign(actual_returns)
        non_zero = act_sign != 0
        dir_acc = np.mean(exp_sign[non_zero] == act_sign[non_zero]) if non_zero.sum() > 0 else 0.0

        return {
            "top1_accuracy": float(top1_acc),
            "top2_accuracy": float(top2_acc),
            "weighted_f1": float(f1),
            "ic": float(ic),
            "directional_accuracy": float(dir_acc),
            "mean_expected_return": float(expected_returns.mean()),
        }

import numpy as np
from sklearn.metrics import cohen_kappa_score


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score

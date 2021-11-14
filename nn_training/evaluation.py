import numpy as np


def MSE(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return ((y_pred - y_true) ** 2).mean()

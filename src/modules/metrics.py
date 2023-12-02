import numpy as np
from sklearn import metrics


def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    results = dict()
    results['RMSE'] = metrics.mean_squared_error(y_pred, y_true, squared=False)
    results['MAE'] = metrics.mean_absolute_error(y_pred, y_true)
    return results

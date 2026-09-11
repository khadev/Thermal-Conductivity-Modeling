"""Statistical metrics for model evaluation."""

import numpy as np


def calculate_statistics(y_true, y_pred):
    """Calculate R2, RMSE, MAE, and MAPE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(y_true - y_pred))

    y_safe = np.clip(np.abs(y_true), 1e-12, None)
    mape = 100.0 * np.mean(np.abs((y_true - y_pred) / y_safe))

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "residuals": y_true - y_pred,
    }

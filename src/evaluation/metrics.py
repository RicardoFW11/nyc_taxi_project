import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """Calculate metrics for model evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }


def print_metrics(metrics, title: str = "Model Metrics"):
    """Print metrics in a formatted way"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():10s}: {value:>10.4f}")
    print(f"{'=' * 50}\n")

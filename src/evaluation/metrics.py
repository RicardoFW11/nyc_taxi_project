"""
Metrics and Model Evaluation Module.

This module centralizes the calculation of key performance indicators (KPIs) for regression problems.
It standardizes comparative evaluation between different experiments,
ensuring that all models (Rate and Duration) are judged under the same criteria.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculates a comprehensive set of regression metrics to evaluate the quality of predictions.
    
    Generates the four fundamental indicators:
    1. R2 (Coefficient of Determination): Measures the proportion of variance explained by the model.
    2. MAE (Mean Absolute Error): Direct interpretation of the average error in the original units.
    3. MSE (Mean Squared Error): Penalizes large errors (useful for detecting outliers).
    4. RMSE (Root Mean Squared Error): Standard metric for comparison with the standard deviation of the target.

    Parameters:
    -----------
        y_true : array-like
            Actual values (Ground Truth) of the dataset.
    y_pred : array-like
            Values estimated by the model during inference.

    Returns:
    --------
    dict
            Dictionary with the calculated metrics rounded to 4 decimal places for readability.
    """
    
    # Ensures that input vectors have a one-dimensional structure (1D array).
    # This prevents broadcasting errors if ‘y_true’ or ‘y_pred’ come in shapes (N, 1).
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Performing statistical calculations
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }

def print_metrics(metrics: dict, title: str = "Model Metrics") -> None:
    """
    Renders the evaluation results in a standardized tabular format for the console.
    
    Visually organizes the metrics, prioritizing R2 and RMSE, which are usually the most relevant indicators
    for making decisions about the model's viability.

    Parameters:
    -----------
    metrics : dict
        Dictionary containing the key-value pairs of the calculated metrics.
    title : str
        Descriptive header for the report block (e.g., ‘Training Set’, ‘Validation Set’).
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    
    # Define a visualization hierarchy to facilitate quick reading of key KPIs.
    order = ["r2", "rmse", "mae", "mse"]
    
    # Iterate over the priority metrics if they are present in the dictionary
    for key in order:
        if key in metrics:
            print(f"{key.upper():10s}: {metrics[key]:>10.4f}")
            
    # Process and display any auxiliary metrics that are not on the priority list.
    for key, value in metrics.items():
        if key not in order:
             print(f"{key.upper():10s}: {value:>10.4f}")
             
    print(f"{'=' * 50}\n")
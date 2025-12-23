import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics for model evaluation.
    
    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.
        
    Returns:
        dict: Dictionary containing R2, MAE, MSE, and RMSE.
    """
    # Aseguramos que sean arrays de 1 dimensión para evitar errores de forma
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Cálculos
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "r2": round(r2, 4),   # <--- ¡Esto es lo que faltaba!
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }

def print_metrics(metrics, title: str = "Model Metrics"):
    """Print metrics in a formatted way"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    
    # Ordenamos un poco la salida para que R2 salga primero si existe
    order = ["r2", "rmse", "mae", "mse"]
    
    for key in order:
        if key in metrics:
            print(f"{key.upper():10s}: {metrics[key]:>10.4f}")
            
    # Imprimir cualquier otra métrica extra que no esté en la lista 'order'
    for key, value in metrics.items():
        if key not in order:
             print(f"{key.upper():10s}: {value:>10.4f}")
             
    print(f"{'=' * 50}\n")
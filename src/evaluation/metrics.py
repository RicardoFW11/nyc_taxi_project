from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           confusion_matrix, classification_report, 
                           precision_recall_fscore_support, accuracy_score)
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing various regression metrics
    """
    
    residuals = y_true - y_pred
    
    metrics = {
        # Basic metrics
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        
        # Residual metrics
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'median_ae': np.median(np.abs(residuals)),
        'max_error': np.max(np.abs(residuals)),
        'min_error': np.min(np.abs(residuals)),
        
        # Error percentiles
        'q25_error': np.percentile(np.abs(residuals), 25),
        'q75_error': np.percentile(np.abs(residuals), 75),
        'q95_error': np.percentile(np.abs(residuals), 95),
    }
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.inf
        
    # # Use sklearn directly for all metrics
    # conf_matrix = confusion_matrix(y_true, y_pred)
    # accuracy = accuracy_score(y_true, y_pred)
    
    # # Calculate precision, recall, f1 using sklearn
    # precision, recall, f1, support = precision_recall_fscore_support(
    #     y_true, y_pred, average=None, zero_division=0
    # )
    
    # sklearn_metrics = {
    #         'confusion_matrix': conf_matrix.tolist(),
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1,
    #         'support': support.tolist()
    #     }
    
    # metrics.update(sklearn_metrics)
        
    return metrics

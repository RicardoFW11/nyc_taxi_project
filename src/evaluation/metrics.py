"""
Módulo de Métricas y Evaluación de Modelos.

Este módulo centraliza el cálculo de indicadores clave de rendimiento (KPIs) para problemas
de regresión. Estandariza la evaluación comparativa entre diferentes experimentos,
asegurando que todos los modelos (Tarifa y Duración) sean juzgados bajo los mismos criterios.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calcula un conjunto integral de métricas de regresión para evaluar la calidad de las predicciones.
    
    Genera los cuatro indicadores fundamentales:
    1. R2 (Coeficiente de determinación): Mide la proporción de varianza explicada por el modelo.
    2. MAE (Error Absoluto Medio): Interpretación directa del error promedio en las unidades originales.
    3. MSE (Error Cuadrático Medio): Penaliza los errores grandes (útil para detectar outliers).
    4. RMSE (Raíz del Error Cuadrático Medio): Métrica estándar para comparar con la desviación estándar del target.

    Parámetros:
    -----------
    y_true : array-like
        Valores reales (Ground Truth) del conjunto de datos.
    y_pred : array-like
        Valores estimados por el modelo durante la inferencia.

    Retorna:
    --------
    dict
        Diccionario con las métricas calculadas redondeadas a 4 decimales para legibilidad.
    """
    
    # Garantiza que los vectores de entrada tengan una estructura unidimensional (1D array).
    # Esto previene errores de transmisión (broadcasting) si 'y_true' o 'y_pred' vienen con formas (N, 1).
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Ejecución de cálculos estadísticos
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
    Renderiza los resultados de la evaluación en un formato tabular estandarizado para la consola.
    
    Organiza visualmente las métricas priorizando el R2 y el RMSE, que suelen ser los indicadores
    más relevantes para la toma de decisiones sobre la viabilidad del modelo.

    Parámetros:
    -----------
    metrics : dict
        Diccionario conteniendo los pares clave-valor de las métricas calculadas.
    title : str
        Encabezado descriptivo para el bloque de reporte (ej. 'Training Set', 'Validation Set').
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    
    # Define una jerarquía de visualización para facilitar la lectura rápida de los KPIs principales.
    order = ["r2", "rmse", "mae", "mse"]
    
    # Itera sobre las métricas prioritarias si están presentes en el diccionario
    for key in order:
        if key in metrics:
            print(f"{key.upper():10s}: {metrics[key]:>10.4f}")
            
    # Procesa y muestra cualquier métrica auxiliar que no esté en la lista prioritaria
    for key, value in metrics.items():
        if key not in order:
             print(f"{key.upper():10s}: {value:>10.4f}")
             
    print(f"{'=' * 50}\n")
"""
Módulo de Modelos Base (Baseline Models) para Regresión.

Este módulo implementa algoritmos de aprendizaje supervisado clásicos que sirven como
punto de referencia para evaluar el rendimiento de modelos más complejos.
Incluye Regresión Lineal (para capturar relaciones lineales simples) y Árboles de Decisión
(para capturar no-linealidades básicas sin el costo computacional de ensambles complejos).
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics
import pandas as pd
import numpy as np

class LinearRegressionModel(BaseModel):
    def __init__(self, output_path:str, 
                 target: str = 'fare_amount',
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 positive: bool = False,
                 **kwargs
                 ):
        """
        Inicializa el Modelo de Regresión Lineal (Mínimos Cuadrados Ordinarios).
        
        Este modelo se utiliza principalmente como 'baseline' para establecer el límite inferior
        de rendimiento esperado. Su simplicidad y alta interpretabilidad lo hacen ideal para
        detectar si la complejidad añadida por otros modelos justifica su costo computacional.
        
        Parámetros:
        -----------
        output_path : str
            Directorio donde se persistirá el modelo serializado.
        target : str
            Variable dependiente a predecir.
        fit_intercept : bool
            Si se debe calcular el sesgo (intercepto) del modelo.
        positive : bool
            Si se debe forzar que los coeficientes sean positivos (útil en contextos de precios).
        kwargs : dict
            Argumentos adicionales para la clase LinearRegression de sklearn.
        """
        
        super().__init__('linear_regression', target, output_path, model_type='baseline')
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            n_jobs=-1 # Utiliza todos los núcleos de CPU disponibles para operaciones matriciales
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Ajusta los coeficientes del modelo lineal a los datos de entrenamiento.
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones lineales basadas en los coeficientes aprendidos.
        Aplica una función de activación ReLU (max(0, x)) a la salida para garantizar
        que no se generen valores negativos física o económicamente imposibles.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Calcula métricas de desempeño estándar comparando predicciones vs valores reales.
        Almacena los resultados internamente para su posterior serialización.
        """
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Devuelve la configuración actual de hiperparámetros del estimador."""
        return self.model.get_params(deep=deep)
        
class DecisionTreeModel(BaseModel):
    def __init__(self, output_path:str, target: str = 'fare_amount',
                 criterion: str = 'squared_error',
                 splitter: str = 'best',
                 max_depth: int = 10,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 max_leaf_nodes: int = 5,
                 min_impurity_decrease: float = 0.001,
                 max_features: str = 'sqrt',
                 **kwargs):
        """
        Inicializa el Modelo de Árbol de Decisión (CART).
        
        Configurado como un modelo base no lineal capaz de capturar interacciones simples
        entre variables. Los hiperparámetros por defecto están restringidos (ej. max_depth=10)
        para prevenir el sobreajuste (overfitting), un problema común en árboles individuales.
        
        Parámetros:
        -----------
        output_path : str
            Ruta de almacenamiento del modelo.
        target : str
            Variable objetivo.
        criterion : str
            Función de pérdida para medir la calidad de una división ('squared_error' para MSE).
        max_depth : int
            Profundidad máxima del árbol. Limita la complejidad del modelo.
        min_samples_split : int
            Número mínimo de muestras necesarias para dividir un nodo interno.
        min_samples_leaf : int
            Número mínimo de muestras requeridas en un nodo hoja (suavizado).
        """
        # Inicialización de la clase padre
        super().__init__('decision_tree', target, output_path, model_type='baseline')
        
        self.model = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=RANDOM_STATE, # Garantiza reproducibilidad determinista
            **kwargs
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Construye el árbol de decisión a partir del conjunto de entrenamiento.
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Infiere valores objetivo para nuevas observaciones.
        Aplica rectificación de valores negativos en la salida.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Ejecuta la evaluación del modelo utilizando el conjunto de métricas estandarizado.
        """
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Recupera los parámetros de configuración del árbol."""
        return self.model.get_params(deep=deep)
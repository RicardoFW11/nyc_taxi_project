"""
Módulo de Definición de Arquitectura Base para Modelos de Machine Learning.

Este módulo establece la interfaz abstracta (ABC) que estandariza el comportamiento
de todos los modelos predictivos dentro del sistema. Define el contrato obligatorio
para métodos críticos como entrenamiento, predicción y evaluación, asegurando
interoperabilidad y consistencia en el pipeline de MLOps.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Literal, Optional
import pickle
import os

class BaseModel(ABC):
    """
    Clase Abstracta Base (ABC) para la implementación de modelos de regresión.
    
    Proporciona la estructura esqueleto para el ciclo de vida del modelo, incluyendo:
    - Inicialización de metadatos y rutas de artefactos.
    - Definición de métodos abstractos para entrenamiento (fit) e inferencia (predict).
    - Mecanismos concretos para la serialización (guardado/carga) del estado del modelo.
    - Gestión estandarizada de métricas de rendimiento.
    
    Cualquier algoritmo nuevo (e.g., XGBoost, RandomForest) debe heredar de esta clase
    e implementar sus métodos abstractos.
    """
    
    def __init__(self, model_name: str, target: str, output_path: str, model_type: Literal['baseline','advanced']='baseline'):
        """
        Inicializa la configuración base del modelo.

        Parámetros:
        -----------
        model_name : str
            Identificador único del algoritmo (ej. 'xgboost', 'random_forest').
        target : str
            Nombre de la variable objetivo que el modelo predecirá (ej. 'fare_amount').
        output_path : str
            Directorio base donde se persistirán los artefactos binarios (.pkl) del modelo.
        model_type : Literal['baseline', 'advanced']
            Categorización del modelo para propósitos de benchmarking y reporte.
        """
        self.model_name = model_name
        self.target = target
        self.is_trained = False
        self.model = None
        self.metrics = {}
        self.model_type = model_type
        
        # Construcción de la ruta absoluta para el artefacto serializado.
        # Se utiliza una convención de nombrado sistemática para facilitar el versionado.
        self.model_output_path = os.path.join(output_path, f"{self.model_name}_{self.target}_{self.model_type}.pkl")
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Método abstracto para el entrenamiento del modelo.
        Debe ser implementado por las clases hijas para ajustar los pesos del algoritmo
        a los datos de entrenamiento proporcionados.
        """
        pass
        
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Método abstracto para la generación de predicciones (inferencia).
        Debe transformar las características de entrada y devolver las estimaciones del modelo.
        """
        pass
    
    def get_params(self, deep=True) -> Dict[str, float]:
        """
        Recupera los hiperparámetros actuales del estimador subyacente.
        Útil para registro de experimentos y auditoría de configuración.
        """
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params(deep=deep)
        return {}
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Método abstracto para la evaluación del rendimiento.
        Debe calcular y retornar métricas estandarizadas (RMSE, MAE, R2) comparando
        las predicciones contra los valores reales.
        """
        pass
    
    def save_model(self) -> str:
        """
        Serializa y persiste el estado completo del objeto modelo en disco.
        
        Utiliza el protocolo pickle para guardar no solo el estimador entrenado,
        sino también sus metadatos asociados (métricas, configuración, tipo),
        permitiendo una reconstrucción total del contexto del experimento.
        
        Returns:
            str: Ruta del archivo generado.
            
        Raises:
            ValueError: Si se intenta guardar un modelo que aún no ha sido entrenado.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before saving it")
        
        # Garantiza la existencia del directorio de destino antes de la escritura
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        
        with open(self.model_output_path, 'wb') as f:
            pickle.dump({
                'model': self,
                'model_name': self.model_name,
                'target': self.target,
                'metrics': self.metrics,
                'model_type': self.model_type
            }, f)
            
        return self.model_output_path
    
    def load_model(self) -> None:
        """
        Reconstruye el estado del modelo desde un archivo serializado.
        
        Restaura el estimador, la configuración y las métricas históricas,
        dejando el objeto listo para realizar inferencias sin necesidad de reentrenamiento.
        """
        with open(self.model_output_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_name = data['model_name']
            self.target = data['target']
            self.metrics = data.get('metrics', {})
            self.model_type = data.get('model_type', self.model_type)
            self.model_output_path = data.get('model_output_path', self.model_output_path)
            self.is_trained = True
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Método opcional para recuperar la importancia de las variables predictoras.
        Retorna None por defecto; las clases hijas deben sobrescribirlo si el algoritmo
        soporta interpretabilidad (ej. árboles de decisión).
        """
        return None
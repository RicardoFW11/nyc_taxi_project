from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import KBinsDiscretizer

class BaseModel(ABC):
    def __init__(self, model_name: str, target: str, model_path:str):
        self.model_name = model_name
        self.target = target
        self.is_trained = False
        self.model_path = model_path
        self.model = None
        self.metrics = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
        
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evalúa el modelo y retorna métricas"""
        pass
    
    def save_model(self) -> None:
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo debe estar entrenado antes de guardarlo")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_name': self.model_name,
                'target': self.target,
                'metrics': self.metrics,
                'model_path': self.model_path
            }, f)
    
    def load_model(self) -> None:
        """Carga un modelo guardado"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_name = data['model_name']
            self.target = data['target']
            self.metrics = data.get('metrics', {})
            self.model_path = data.get('model_path', self.model_path)
            self.is_trained = True
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna la importancia de las características si está disponible"""
        return None

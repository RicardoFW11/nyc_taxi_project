from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Literal, Optional
import pickle
import os

class BaseModel(ABC):
    def __init__(self, model_name: str, target: str, output_path:str, model_type:Literal['baseline','advanced']='baseline'):
        """
            Base Class for all machine learning models
            
            Args:
                model_name (str): Name of the model
                target (str): Target variable
                output_path (str): Path where the model will be saved
                model_type (Literal['baseline', 'advanced']): Type of model ('baseline' or 'advanced')
        """
        self.model_name = model_name
        self.target = target
        self.is_trained = False
        self.model = None
        self.metrics = {}
        self.model_type = model_type
        
        self.model_output_path = os.path.join(output_path, f"{self.model_name}_{self.target}_{self.model_type}.pkl")
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
        
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def get_params(self, deep=True) -> Dict[str, float]:
        """Return model parameters"""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model and return metrics"""
        pass
    
    def save_model(self) -> str:
        """
            Save the trained model to a file
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before saving it")
        
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        
        with open(self.model_output_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_name': self.model_name,
                'target': self.target,
                'metrics': self.metrics,
                'model_type': self.model_type
            }, f)
            
        return self.model_output_path
    
    def load_model(self) -> None:
        """
            Load a saved model from a file
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
        """Return feature importance if available"""
        return None

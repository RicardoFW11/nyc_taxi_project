from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, model_name: str, target: str):
        self.model_name = model_name
        self.target = target
        self.is_trained = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
        
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
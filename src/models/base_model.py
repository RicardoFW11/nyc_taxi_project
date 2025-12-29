"""
Base Architecture Definition Module for Machine Learning Models.

This module establishes the abstract interface (ABC) that standardizes the behavior
of all predictive models within the system. It defines the mandatory contract
for critical methods such as training, prediction, and evaluation, ensuring
interoperability and consistency in the MLOps pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Literal, Optional
import pickle
import os

class BaseModel(ABC):
    """
    Abstract Base Class (ABC) for implementing regression models.
    
    Provides the skeleton structure for the model lifecycle, including:
    - Initialization of metadata and artifact paths.
    - Definition of abstract methods for training (fit) and inference (predict).
    - Concrete mechanisms for serialization (saving/loading) of the model state.
    - Standardized management of performance metrics.
    
    Any new algorithm (e.g., XGBoost, RandomForest) must inherit from this class
    and implement its abstract methods.

    """
    
    def __init__(self, model_name: str, target: str, output_path: str, model_type: Literal['baseline','advanced']='baseline'):
        """
        Initializes the base configuration of the model.

        Parameters:
        -----------
        model_name : str
            Unique identifier of the algorithm (e.g., ‘xgboost’, ‘random_forest’).
        target : str
            Name of the target variable that the model will predict (e.g., ‘fare_amount’).
        output_path : str
            Base directory where the model's binary artifacts (.pkl) will be stored.
        model_type : Literal[‘baseline’, ‘advanced’]
            Model categorization for benchmarking and reporting purposes.
        """
        self.model_name = model_name
        self.target = target
        self.is_trained = False
        self.model = None
        self.metrics = {}
        self.model_type = model_type
        
        # Building the absolute path for the serialized artifact.
        # A systematic naming convention is used to facilitate versioning.
        self.model_output_path = os.path.join(output_path, f"{self.model_name}_{self.target}_{self.model_type}.pkl")
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Abstract method for training the model.
        It must be implemented by child classes to adjust the algorithm weights
        to the training data provided.
        """
        pass
        
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Abstract method for generating predictions (inference).
        It must transform the input features and return the model estimates.
        """
        pass
    
    def get_params(self, deep=True) -> Dict[str, float]:
        """
        Retrieves the current hyperparameters of the underlying estimator.
        Useful for experiment logging and configuration auditing.
        """
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params(deep=deep)
        return {}
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Abstract method for performance evaluation.
        It must calculate and return standardized metrics (RMSE, MAE, R2) by comparing
        the predictions against the actual values.
        """
        pass
    
    def save_model(self) -> str:
        """
        Serializes and persists the complete state of the model object to disk.
        
        Uses the pickle protocol to save not only the trained estimator,
        but also its associated metadata (metrics, configuration, type),
        allowing for a complete reconstruction of the experiment context.
        
        Returns:
            str: Path of the generated file.
            
        Raises:
            ValueError: If an attempt is made to save a model that has not yet been trained.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before saving it")
        
        # Ensure that the destination directory exists before writing
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
        Rebuilds the model state from a serialized file.
        
        Restores the estimator, configuration, and historical metrics,
        leaving the object ready to perform inferences without the need for retraining.
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
        Optional method to retrieve the importance of predictor variables.
        Returns None by default; child classes should override it if the algorithm
        supports interpretability (e.g., decision trees).
        """
        return None
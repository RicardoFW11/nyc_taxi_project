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
            Initialize the Linear Regression Model
            
            Args:
                output_path (str): Path base donde se guardará el modelo.
                target (str): Variable objetivo ('fare_amount' o 'trip_duration_minutes').
        """
        
        # Inicializa la clase base con el nombre del modelo, el target y el tipo 'baseline'
        super().__init__('linear_regression', target, output_path, model_type='baseline')
        
        # Crea la instancia del modelo de scikit-learn
        self.model = LinearRegression(fit_intercept=fit_intercept,
                                     copy_X=copy_X,
                                     positive=positive,
                                     n_jobs=-1)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo de regresión lineal."""
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        # Asegura que las predicciones sean no negativas para variables como tarifa/duración
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Evalúa el rendimiento del modelo y guarda las métricas."""
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Devuelve los parámetros del modelo."""
        return self.model.get_params(deep=deep)
        
class DecisionTreeModel(BaseModel):
    def __init__(self, output_path:str,target: str = 'fare_amount',
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
            Initialize the Decision Tree Model
            
            Args:
                output_path (str): Path base donde se guardará el modelo.
                target (str): Variable objetivo ('fare_amount' o 'trip_duration_minutes').
        """
        # Inicializa la clase base con el nombre del modelo, el target y el tipo 'baseline'
        super().__init__('decision_tree', target, output_path, model_type='baseline')
        
        # Crea la instancia del modelo de scikit-learn
        self.model = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=RANDOM_STATE,
            **kwargs
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo de árbol de decisión."""
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Evalúa el rendimiento del modelo y guarda las métricas."""
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Devuelve los parámetros del modelo."""
        return self.model.get_params(deep=deep)
"""
Módulo de Optimización de Hiperparámetros basado en Scikit-Learn.

Este módulo extiende las capacidades de optimización del sistema integrando herramientas
estándar de la industria como GridSearchCV y RandomizedSearchCV. Implementa una capa de
adaptación (Wrapper) que permite que los modelos personalizados del proyecto sean
compatibles con la API de estimadores de scikit-learn.

Incluye estrategias de gestión de memoria (GC, limpieza explícita) para manejar
grandes volúmenes de datos durante la búsqueda intensiva de parámetros.
"""

from scipy.stats import randint, uniform, loguniform
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, 
    KFold, 
    GridSearchCV, 
    RandomizedSearchCV,
    train_test_split
)
from sklearn.base import BaseEstimator
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import joblib
import time
from enum import Enum
import gc

from src.config.paths import LOGGER_NAME
from src.config.settings import RANDOM_STATE
from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel
from src.utils.logging import LoggerFactory

# Configuración del logger para el módulo
logger = LoggerFactory.create_logger(
    name=LOGGER_NAME,
    log_level='DEBUG',
    console_output=True,
    file_output=False
)

class OptimizationMethod(Enum):
    """Enumeración de estrategias de optimización disponibles."""
    GRID_SEARCH = "grid_search"       # Búsqueda exhaustiva en rejilla predefinida
    RANDOM_SEARCH = "random_search"   # Muestreo aleatorio del espacio de parámetros
    BAYESIAN_OPTUNA = "bayesian_optuna" # Optimización bayesiana (implementada en otro módulo)
    
class SklearnHyperparameterTuner(ABC):
    """
    Clase base abstracta para el tuning de hiperparámetros utilizando la suite de scikit-learn.
    
    Proporciona la infraestructura común para:
    1. Carga y gestión eficiente de datos en memoria.
    2. Adaptación de modelos propios a la interfaz de sklearn (fit/predict/get_params).
    3. Ejecución de búsquedas de parámetros (Grid/Random).
    """
    
    def __init__(
        self, 
        model_class, 
        data_path: str,
        output_path: str,
        target: str = 'fare_amount',
        cv_folds: int = 3,  # Valor reducido por defecto para acelerar iteraciones
        method: Union[str, OptimizationMethod] = OptimizationMethod.RANDOM_SEARCH,
        scoring: str = 'neg_mean_squared_error'
    ):
        """
        Inicializa el sintonizador de hiperparámetros.

        Args:
            model_class: Clase del modelo a optimizar (debe heredar de BaseModel).
            data_path (str): Ruta al archivo de datos serializado (.pkl).
            output_path (str): Directorio de destino para el modelo optimizado.
            target (str): Nombre de la variable objetivo.
            cv_folds (int): Número de pliegues para validación cruzada.
            method (Union[str, OptimizationMethod]): Estrategia de búsqueda.
            scoring (str): Métrica de evaluación compatible con sklearn.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.model_class = model_class
        self.output_path = output_path
        self.target = target
        
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = RANDOM_STATE
        
        # Normalización del método de optimización a Enum
        if isinstance(method, str):
            self.method = OptimizationMethod(method)
        else:
            self.method = method
            
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
        np.random.seed(RANDOM_STATE)

    def clear_data(self):
        """
        Libera explícitamente los recursos de memoria ocupados por los datasets.
        Crucial en entornos contenerizados (Docker) para evitar errores OOM (Out Of Memory)
        entre ejecuciones consecutivas.
        """
        if hasattr(self, 'X'):
            logger.info("Clearing X data from memory...")
            del self.X
        if hasattr(self, 'y'):
            logger.info("Clearing Y data from memory...")
            del self.y
        if hasattr(self, 'data'):
            logger.info("Clearing data from memory...")
            del self.data
            
        logger.info("✓ Data cleared from memory")
        gc.collect() # Invoca al recolector de basura de Python
        return True
        
    def load_data(self):
        """
        Carga, reconstruye y muestrea el dataset de entrenamiento.
        
        Combina los splits de entrenamiento, validación y prueba para permitir que
        GridSearchCV realice su propia validación cruzada interna completa.
        Aplica un límite estricto de filas (200k) para garantizar la viabilidad computacional.
        """
        logger.info("Loading data from data_splitter...")
        
        self.data = joblib.load(self.data_path)
        self.features = self.data['features']
        
        # Reconstrucción de DataFrames a partir de los splits
        X_train = pd.DataFrame(self.data['X_train'][self.features])
        X_test = pd.DataFrame(self.data['X_test'][self.features])
        X_val = pd.DataFrame(self.data['X_val'][self.features])

        # Concatenación vertical para formar el dataset completo de optimización
        self.X = pd.concat([X_train, X_test, X_val], axis=0, ignore_index=True)
        
        self.y = pd.concat([
            self.data['y_train'], 
            self.data['y_test'], 
            self.data['y_val']
        ], axis=0, ignore_index=True)
    
        self.feature_scores = self.data['feature_scores']
        
        # Estrategia de Muestreo Defensivo
        limit_size = 200000
        if len(self.X) > limit_size:
            logger.info(f"⚠️ Muestreando {limit_size} filas para evitar error de memoria (OOM)...")
            self.X = self.X.sample(n=limit_size, random_state=42)
            self.y = self.y.loc[self.X.index]

        logger.info(f"✓ Data loaded successfully:")
        logger.info(f"  - X shape: {self.X.shape}")
        logger.info(f"  - y shape: {self.y.shape}")
        logger.info(f"  - Features: {len(self.features)}")
        
        return True
    
    def _create_sklearn_estimator(self, **params):
        """
        Patrón Adapter: Crea un envoltorio (Wrapper) compatible con Scikit-Learn.
        
        Permite que las clases de modelo del proyecto (que tienen su propia API personalizada)
        sean consumidas por herramientas estándar como GridSearchCV, que esperan métodos
        específicos como get_params y set_params.
        """
        class SklearnCompatibleWrapper(BaseEstimator):
            def __init__(self, **hyperparams):
                # Almacena hiperparámetros en un diccionario interno para evitar recursión infinita
                # en getattr/setattr durante la clonación del estimador.
                self._hyperparams = hyperparams.copy()
                for key, value in hyperparams.items():
                    setattr(self, key, value)
                self._model = None
                
            def fit(self, X, y):
                """Delega el entrenamiento al modelo subyacente personalizado."""
                self._model = self.model_class(
                    output_path=self.output_path,
                    target=self.target,
                    **self._hyperparams
                )
                self._model.fit(X, y)
                return self
                
            def predict(self, X):
                """Delega la predicción al modelo subyacente."""
                if self._model is None:
                    raise ValueError("Model must be fitted before prediction")
                return self._model.predict(X)
                
            def get_params(self, deep=True):
                """Interfaz requerida por sklearn para inspeccionar parámetros."""
                return self._hyperparams.copy()
                
            def set_params(self, **params):
                """Interfaz requerida por sklearn para actualizar parámetros durante CV."""
                self._hyperparams.update(params)
                for key, value in params.items():
                    setattr(self, key, value)
                return self
        
        # Inyección de metadatos de clase al wrapper
        SklearnCompatibleWrapper.model_class = self.model_class
        SklearnCompatibleWrapper.output_path = self.output_path
        SklearnCompatibleWrapper.target = self.target
        
        return SklearnCompatibleWrapper(**params)

    def _grid_search(self) -> Dict[str, Any]:
        """
        Ejecuta la optimización mediante Búsqueda en Rejilla (Grid Search).
        Evalúa exhaustivamente todas las combinaciones definidas en el espacio de parámetros.
        """
        logger.info("🔍 Starting Grid Search optimization...")
        start_time = time.time()
        
        param_grid = self.suggest_hyperparameters()
        base_estimator = self._create_sklearn_estimator()
        
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=18, # Paralelización agresiva (ajustar según hardware disponible)
            verbose=2,
            error_score='raise'
        )
        
        search.fit(self.X, self.y)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.search_results = search
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Grid Search completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score:.6f}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': search.cv_results_,
            'method': 'grid_search'
        }

    def optimize(self) -> Dict[str, Any]:
        """
        Punto de entrada principal para ejecutar la optimización.
        Valida el estado y delega a la estrategia específica configurada.
        """
        logger.info(f"Starting hyperparameter optimization using {self.method.value}")
        
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Must load data first using load_data()")
        
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._grid_search()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            # Implementación futura o extensión
            raise ValueError(f"Unknown optimization method: {self.method}")
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
    def get_best_model(self, **additional_params):
        """
        Instancia un nuevo modelo configurado con los parámetros ganadores.
        """
        if self.best_params is None:
            raise ValueError("Must run optimization first")
        
        params = {**self.best_params, **additional_params}
        return self.model_class(
            output_path=self.output_path,
            target=self.target,
            **params
        )
        
    @abstractmethod
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """
        Método abstracto para definir el espacio de búsqueda.
        Debe ser implementado por cada subclase de sintonizador.
        """
        pass
    
class LinearRegressionTuner(SklearnHyperparameterTuner):
    """
    Sintonizador para Regresión Lineal.
    Espacio de búsqueda mínimo, enfocado en configuración estructural (intercepto, positividad).
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 3,
                 method: Union[str, OptimizationMethod] = OptimizationMethod.GRID_SEARCH,
                 scoring: str = 'neg_mean_squared_error'):
        super().__init__(
            model_class=LinearRegressionModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            method=method,
            scoring=scoring
        )
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        params = {
            'fit_intercept': [True],
            'copy_X': [True],
            'positive': [True]
        }
        if self.method == OptimizationMethod.GRID_SEARCH:
            return params
        else:
            raise ValueError(f"Unsupported method for Linear Regression: {self.method}")
        
class DecisionTreeTuner(SklearnHyperparameterTuner):
    """
    Sintonizador para Árboles de Decisión.
    Espacio de búsqueda orientado a controlar la complejidad y profundidad del árbol.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 3,
                 method: Union[str, OptimizationMethod] = OptimizationMethod.RANDOM_SEARCH,
                 scoring: str = 'neg_mean_squared_error'):
        super().__init__(
            model_class=DecisionTreeModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            method=method,
            scoring=scoring
        )
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        else:
            raise ValueError(f"Unsupported method for Decision Tree Regression: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Define la rejilla de parámetros para GridSearch."""
        return {
            'criterion': ['squared_error', 'friedman_mse'],
            'splitter': ['best', 'random'],
            'max_depth': [5, 7, 9],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [10, 20],
            'max_leaf_nodes': [10, 20, 30],
            'min_impurity_decrease': [0.01, 0.001],
            'max_features': ['sqrt', 'log2']
        }
        
class XGBoostTuner(SklearnHyperparameterTuner):
    """
    Sintonizador para XGBoost.
    Espacio de búsqueda optimizado para regularización y parámetros de boosting.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 3,
                 method: Union[str, OptimizationMethod] = OptimizationMethod.RANDOM_SEARCH,
                 scoring: str = 'neg_mean_squared_error'):
        super().__init__(
            model_class=XGBoostModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            method=method,
            scoring=scoring
        )
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        else:
            raise ValueError(f"Unsupported method for XGBoost: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """
        Define una rejilla limitada para XGBoost.
        Se ha reducido n_estimators a 50 para asegurar tiempos de ejecución viables en GridSearch.
        """
        return {
            'n_estimators': [50], # Reducido intencionalmente para optimizar tiempo
            'max_depth': [6, 8, 10],
            'colsample_bytree': [0.6, 0.8],
            'colsample_bylevel': [0.6, 0.8],
            'min_child_weight': [5, 7],
            'reg_alpha': [0.2, 0.5],
            'reg_lambda': [0.5, 1.5],
            'tree_method': ['hist']
        }    
    
class RandomForestTuner(SklearnHyperparameterTuner):
    """
    Sintonizador para Random Forest.
    Espacio de búsqueda enfocado en el tamaño del ensamble y las divisiones de nodos.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 3,
                 method: Union[str, OptimizationMethod] = OptimizationMethod.RANDOM_SEARCH,
                 scoring: str = 'neg_mean_squared_error'):
        super().__init__(
            model_class=RandomForestModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            method=method,
            scoring=scoring
        )
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        else:
            raise ValueError(f"Unsupported method for Random Forest: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Define la rejilla de parámetros para Random Forest."""
        return {
            'n_estimators': [100],
            'max_depth': [5, 7, 9],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True],
            'criterion': ['squared_error', 'friedman_mse']
        }
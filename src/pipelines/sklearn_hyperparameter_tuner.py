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
from src.config.paths import LOGGER_NAME
from src.config.settings import RANDOM_STATE
from pathlib import Path
import joblib
import time
from enum import Enum

from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel

from src.utils.logging import LoggerFactory
logger = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )

import gc

class OptimizationMethod(Enum):
    """Available optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search" 
    BAYESIAN_OPTUNA = "bayesian_optuna"
    
class SklearnHyperparameterTuner(ABC):
    """
    Flexible hyperparameter tuning supporting multiple optimization methods:
    - GridSearchCV (exhaustive search, good for small param spaces)
    - RandomizedSearchCV (random sampling, much faster)
    - Bayesian Optimization with Optuna (intelligent search)
    """
    
    def __init__(
        self, 
                 model_class, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 3,  # Reduced default for speed
                 method: Union[str, OptimizationMethod] = OptimizationMethod.RANDOM_SEARCH,
                 scoring: str = 'neg_mean_squared_error'
    ):
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.model_class = model_class
        self.output_path = output_path
        self.target = target
        
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = RANDOM_STATE
        
        # Convert string to enum if needed
        if isinstance(method, str):
            self.method = OptimizationMethod(method)
        else:
            self.method = method
            
        self.best_params = None
        self.best_score = None
        self.search_results = None
        
        np.random.seed(RANDOM_STATE)

    def clear_data(self):
        """Clear loaded data from memory"""
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

        gc.collect()

        return True
        
    def load_data(self):
        """Load split data from the data splitter output"""
        logger.info("Loading data from data_splitter...")
        
        self.data = joblib.load(self.data_path)
        
        self.features = self.data['features']
        
        X_train = pd.DataFrame(self.data['X_train'][self.features])
        X_test = pd.DataFrame(self.data['X_test'][self.features])
        X_val = pd.DataFrame(self.data['X_val'][self.features])
        
        # Concatenate all X datasets
        self.X = pd.concat([X_train, X_test, X_val], axis=0, ignore_index=True)
        
        # Concatenate all y datasets
        self.y = pd.concat([
            self.data['y_train'], 
            self.data['y_test'], 
            self.data['y_val']
        ], axis=0, ignore_index=True)
    
        self.feature_scores = self.data['feature_scores']
        
        # Sample data if requested for faster tuning
        # if self.sample_size and self.sample_size < len(self.X):
        #     logger.info(f"Sampling {self.sample_size} rows for faster tuning...")
        #     self.X, _, self.y, _ = train_test_split(
        #         self.X, self.y, 
        #         train_size=self.sample_size,
        #         random_state=self.random_state,
        #         stratify=None
        #     )
        
        logger.info(f"✓ Data loaded successfully:")
        logger.info(f"  - X shape: {self.X.shape}")
        logger.info(f"  - y shape: {self.y.shape}")
        logger.info(f"  - Features: {len(self.features)}")
        
        return True
    
    def _create_sklearn_estimator(self, **params):
        """Create sklearn-compatible estimator from model class"""
        # Create a wrapper class that sklearn can clone properly
        class SklearnCompatibleWrapper(BaseEstimator):
            def __init__(self, **hyperparams):
                # Explicitly store hyperparameters to avoid recursion issues
                self._hyperparams = hyperparams.copy()
                # Set each hyperparam as an attribute (required for sklearn)
                for key, value in hyperparams.items():
                    setattr(self, key, value)
                self._model = None
                
            def fit(self, X, y):
                # Create and fit the actual model using stored hyperparams
                self._model = self.model_class(
                    output_path=self.output_path,
                    target=self.target,
                    **self._hyperparams
                )
                self._model.fit(X, y)
                return self
                
            def predict(self, X):
                if self._model is None:
                    raise ValueError("Model must be fitted before prediction")
                return self._model.predict(X)
                
            def get_params(self, deep=True):
                # Return the stored hyperparameters to avoid recursion
                return self._hyperparams.copy()
                
            def set_params(self, **params):
                # Update stored hyperparams and set as attributes
                self._hyperparams.update(params)
                for key, value in params.items():
                    setattr(self, key, value)
                return self
        
        # Add required attributes to the wrapper
        SklearnCompatibleWrapper.model_class = self.model_class
        SklearnCompatibleWrapper.output_path = self.output_path
        SklearnCompatibleWrapper.target = self.target
        
        return SklearnCompatibleWrapper(**params)

    def _grid_search(self) -> Dict[str, Any]:
        """Run GridSearchCV optimization"""
        logger.info("🔍 Starting Grid Search optimization...")
        start_time = time.time()
        
        param_grid = self.suggest_hyperparameters()
        base_estimator = self._create_sklearn_estimator()
        
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=18, # Adjust based on your system
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
        """Run optimization using the specified method"""
        logger.info(f"Starting hyperparameter optimization using {self.method.value}")
        
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Must load data first using load_data()")
        
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._grid_search()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            raise ValueError(f"Unknown optimization method: {self.method}")
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
    def get_best_model(self, **additional_params):
        """Create and return model with best hyperparameters"""
        if self.best_params is None:
            raise ValueError("Must run optimization first")
        
        # Merge best params with any additional parameters
        params = {**self.best_params, **additional_params}
        return self.model_class(
            output_path=self.output_path,
            target=self.target,
            **params
        )
        
    @abstractmethod
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Define hyperparameter space for Optuna optimization"""
        pass
    
class LinearRegressionTuner(SklearnHyperparameterTuner):
    """Hyperparameter tuning for Linear Regression models"""
    
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
        """Suggest hyperparameters based on optimization method"""
        params = {
            'fit_intercept': [True],
            'copy_X': [True],
            'positive': [True]
        }
        if self.method == OptimizationMethod.GRID_SEARCH:
            return params
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            raise ValueError(f"Unsupported method for Linear Regression: {self.method}")
        else:
            raise ValueError(f"Unsupported method for Linear Regression: {self.method}")
        
class DecisionTreeTuner(SklearnHyperparameterTuner):
    """Hyperparameter tuning for Decision Tree models"""
    
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
        """Suggest hyperparameters based on optimization method"""
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            raise ValueError(f"Unsupported method for Decision Tree Regression: {self.method}")
        else:
            raise ValueError(f"Unsupported method for Decision Tree Regression: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Parameter grid for GridSearch (smaller for speed)"""
        return {
            'criterion': ['squared_error', 'friedman_mse'], # splitting criterion
            'splitter': ['best', 'random'], # splitting strategy
            'max_depth': [5, 7, 9], # depth of the tree
            'min_samples_split': [10, 20, 30], # min samples to split
            'min_samples_leaf': [10, 20], # min samples at leaf
            'max_leaf_nodes': [10, 20, 30], # max leaf nodes
            'min_impurity_decrease': [0.01, 0.001], # min impurity decrease
            'max_features': ['sqrt', 'log2'] # max features
        }
        
class XGBoostTuner(SklearnHyperparameterTuner):
    """Hyperparameter tuning for XGBoost models"""
    
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
        """Suggest hyperparameters based on optimization method"""
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            raise ValueError(f"Unsupported method for XGBoost: {self.method}")
        else:
            raise ValueError(f"Unsupported method for XGBoost: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Parameter grid for GridSearch (very limited for speed)"""
        return {
            'n_estimators': [100], # number of trees
            'max_depth': [6, 8, 10], # depth of each tree
            #'learning_rate': [0.01, 0.1], # step size
            #'subsample': [0.5, 0.9], # row sampling
            'colsample_bytree': [0.6, 0.8], # feature sampling
            'colsample_bylevel': [0.6, 0.8], # feature sampling per level
            'min_child_weight': [5, 7], # min sum hessian in leaf
            #'gamma': [0.2, 0.5], # min loss reduction
            'reg_alpha': [0.2, 0.5], # L1 regularization
            'reg_lambda': [0.5, 1.5], # L2 regularization
            'tree_method': ['hist'] # tree construction method
        }    
    
class RandomForestTuner(SklearnHyperparameterTuner):
    """Hyperparameter tuning for Random Forest models"""
    
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
        """Suggest hyperparameters based on optimization method"""
        if self.method == OptimizationMethod.GRID_SEARCH:
            return self._get_param_grid()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            raise ValueError(f"Unsupported method for Random Forest: {self.method}")
        else:
            raise ValueError(f"Unsupported method for Random Forest: {self.method}")
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Parameter grid for GridSearch (limited for speed)"""
        return {
            'n_estimators': [100],
            'max_depth': [5, 7, 9],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2'],
            'max_samples': [0.5, 0.7],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'friedman_mse']
            #'min_impurity_decrease': [0.01, 0.001],
            #'min_weight_fraction_leaf': [0.01, 0.05],
            #'max_leaf_nodes': [10, 20]
        }
    
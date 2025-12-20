import optuna
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict, Any, Optional
from src.config.paths import LOGGER_NAME
from src.config.settings import RANDOM_STATE
from pathlib import Path
import joblib

from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel

from src.utils.logging import LoggerFactory
logger = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna for BaseModel implementations"""
    
    def __init__(self, 
                 model_class, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Args:
            model_class: Class that inherits from BaseModel
            data_path (str): Path to the data file
            output_path (str): Path where the model will be saved
            target (str): Target variable name
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            direction: "minimize" or "maximize" the objective
        """
        
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.model_class = model_class
        self.output_path = output_path
        self.target = target
        
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.direction = direction
        self.random_state = RANDOM_STATE
        self.best_params = None
        self.best_score = None
        
    def load_data(self):
        """
            Load split data from the data splitter output
        """
        try:
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
            
            logger.info(f"✓ Data loaded and combined successfully:")
            logger.info(f"  - Combined X: {self.X.shape}")
            logger.info(f"  - Combined y: {self.y.shape}")
            logger.info(f"  - Features: {len(self.features)}")

        # AGREGAR ESTO AL FINAL DEL MÉTODO load_data:
            MAX_OPTIMIZE_SAMPLES = 200000 
            if len(self.X) > MAX_OPTIMIZE_SAMPLES:
                print(f"DEBUG: Limiting optimization data to {MAX_OPTIMIZE_SAMPLES} rows")
                self.X = self.X.sample(n=MAX_OPTIMIZE_SAMPLES, random_state=42)
                self.y = self.y.loc[self.X.index]

            return True

        except FileNotFoundError as e:
            logger.error(f"❌ Error: Split data files not found.")
            logger.error(f"Make sure to run data_splitter.py first.")
            logger.error(f"Missing file: {e}")
            return False



    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter space for optimization"""
        pass
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        try:
            # Get suggested hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Create model instance with suggested parameters
            model = self.model_class(
                output_path=self.output_path,
                target=self.target,
                **params
            )
            
            # Perform cross-validation
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            scores = []
            for train_idx, val_idx in kfold.split(self.X):
                X_train_fold = self.X.iloc[train_idx]
                X_val_fold = self.X.iloc[val_idx]
                y_train_fold = self.y.iloc[train_idx]
                y_val_fold = self.y.iloc[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                metrics = model.evaluate(X_val_fold, y_val_fold)
                
                # Extract primary metric (customize based on your needs)
                primary_metric = self._get_primary_metric(metrics)
                scores.append(primary_metric)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Trial failed with parameters {params}: {e}")
            # Return worst possible score for failed trials
            return float('inf') if self.direction == "minimize" else float('-inf')
    
    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Extract primary metric for optimization (override in subclass)"""
        if 'rmse' in metrics:
            return metrics['rmse']
        elif 'mae' in metrics:
            return metrics['mae']
        elif 'mse' in metrics:
            return metrics['mse']
        else:
            raise ValueError("No suitable metric found for optimization")
    
    def optimize(self, 
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None) -> optuna.Study:
        """Run hyperparameter optimization"""
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            storage=storage
        )
        
        # Add pruner for early stopping of unpromising trials
        study.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return study
    
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
        
class LinearRegressionTuner(HyperparameterTuner):
    """Hyperparameter tuning for Linear Regression models"""
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Initialize Linear Regression tuner with same parameters as base class
        """
        super().__init__(
            model_class=LinearRegressionModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            n_trials=n_trials,
            direction=direction
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter space for Linear Regression
        Note: LinearRegression has very few hyperparameters to tune
        """
        return {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }
        
class DecisionTreeTuner(HyperparameterTuner):
    """Hyperparameter tuning for Decision Tree models"""
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Initialize Decision Tree tuner with same parameters as base class
        """
        super().__init__(
            model_class=DecisionTreeModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            n_trials=n_trials,
            direction=direction
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter space for Decision Tree
        """
        return {
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 200, step=10),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01, step=0.005),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
class XGBoostTuner(HyperparameterTuner):
    """Hyperparameter tuning for XGBoost models"""
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Initialize XGBoost tuner with same parameters as base class
        """
        super().__init__(
            model_class=XGBoostModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            n_trials=n_trials,
            direction=direction
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter space for XGBoost
        """
        
        # Tree method selection affects which other parameters are valid
        tree_method = trial.suggest_categorical('tree_method', ['hist', 'exact', 'approx'])
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.05),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5, step=0.01),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0, step=0.05),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0, step=0.1),
            'tree_method': tree_method
        }
        
        # Add grow_policy and max_leaves only for tree methods that support them
        if tree_method in ['hist', 'gpu_hist']:
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            
            # max_leaves is only meaningful with lossguide policy
            if params['grow_policy'] == 'lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 16, 256, step=16)
        
        return params
    
class RandomForestTuner(HyperparameterTuner):
    """Hyperparameter tuning for Random Forest models"""
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Initialize Random Forest tuner with same parameters as base class
        """
        super().__init__(
            model_class=RandomForestModel,
            data_path=data_path,
            output_path=output_path,
            target=target,
            cv_folds=cv_folds,
            n_trials=n_trials,
            direction=direction
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter space for Random Forest
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0, step=0.1),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01, step=0.001),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1, step=0.01),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 500, step=10)
        }
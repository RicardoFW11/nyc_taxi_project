from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics

class XGBoostModel(BaseModel):
    def __init__(self, output_path:str,target: str = 'fare_amount',
                 n_estimators: int = 100,
                 max_depth: int = 8,
                 learning_rate: float = 0.05,
                 subsample: float = 0.85,
                 colsample_bytree: float = 0.85,
                 colsample_bylevel: float = 0.8,
                 min_child_weight: int = 3,
                 gamma: float = 0.1,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 tree_method: str = 'hist',
                 grow_policy: str = 'depthwise',
                 max_leaves: int = 0,
                 **kwargs):
        """
            Initialize the XGBoost Model
        """
        super().__init__('xgboost', target, output_path, model_type='advanced')
        
        xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'tree_method': tree_method,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_jobs': -1, # Utilize multiple CPU cores
            'random_state': RANDOM_STATE,
            'verbosity': 0,
            # 'device': 'gpu' # Usar solo si está disponible y configurado
        }
        
        if tree_method in ['hist', 'gpu_hist']:
            xgb_params['grow_policy'] = grow_policy
            if max_leaves > 0:
                xgb_params['max_leaves'] = max_leaves
        
        xgb_params.update(kwargs)
        
        self.model = XGBRegressor(**xgb_params)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature preparation specific to XGBoost"""
        features = data.copy()
        # No se necesita drop si X_train ya tiene solo las features correctas
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the XGBoost model with optional validation"""
        X_prepared = self._prepare_features(X_train)

        self.model.fit(X_prepared, y_train)
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost"""
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        # Ensure predictions are not negative for fares/duration
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate the XGBoost model"""
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Add XGBoost-specific metrics
        self.metrics.update({
            'n_estimators_used': self.model.get_booster().num_boosted_rounds(),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        feature_names = self.model.get_booster().feature_names
        
        # Different types of importance
        gain_importance = self.model.get_booster().get_score(importance_type='gain')
        weight_importance = self.model.get_booster().get_score(importance_type='weight')
        cover_importance = self.model.get_booster().get_score(importance_type='cover')
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain_importance': [gain_importance.get(f, 0) for f in feature_names],
            'weight_importance': [weight_importance.get(f, 0) for f in feature_names],
            'cover_importance': [cover_importance.get(f, 0) for f in feature_names]
        }).sort_values('gain_importance', ascending=False)
        
        return importance_df

class RandomForestModel(BaseModel):
    def __init__(self, output_path:str,target: str = 'fare_amount',
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 criterion: str = 'squared_error',
                 min_impurity_decrease: float = 0.001,
                 min_weight_fraction_leaf: float = 0.01,
                 max_leaf_nodes: int = None,
                 **kwargs):
        
        """
            Initialize the Random Forest Model
        """
        
        super().__init__('random_forest', target, output_path, model_type='advanced')
        
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'criterion': criterion,
            'min_impurity_decrease': min_impurity_decrease,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'warm_start': False,
            'verbose': 0
        }
        
        if max_leaf_nodes is not None:
            rf_params['max_leaf_nodes'] = max_leaf_nodes
            
        rf_params.update(kwargs)
        
        self.model = RandomForestRegressor(**rf_params)
        self.oob_score_ = None
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Random Forest"""
        features = data.copy()
        # No se necesita drop si X_train ya tiene solo las features correctas
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Random Forest model with OOB score"""
        # Asegurarse que oob_score esté activado antes de fit
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
            self.model.oob_score = True
        
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
             self.oob_score_ = self.model.oob_score_

        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest"""
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        # Ensure predictions are not negative for fares/duration
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate the Random Forest model"""
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Add Random Forest-specific metrics
        self.metrics.update({
            'oob_score': self.oob_score_,
            'n_trees': self.model.n_estimators
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            # Fallback: generar nombres genéricos
            n_features = len(self.model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Average importance and per tree
        importance_scores = self.model.feature_importances_
        
        # Calculate variability of importance among trees
        tree_importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        importance_std = np.std(tree_importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_scores,
            'importance_std': importance_std,
            'importance_cv': importance_std / (importance_scores + 1e-10) # Coefficient of variation
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
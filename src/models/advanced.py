from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics

class XGBoostModel(BaseModel):
    def __init__(self, output_path:str,target: str = 'fare_amount'):
        """
            Initialize the XGBoost Model
            Args:
                output_path (str): Path where the model will be saved
                target (str): Target variable
        """
        super().__init__('xgboost', target, output_path, model_type='advanced')
        self.model = XGBRegressor(
            n_estimators=200, 
            max_depth=8, 
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=-1,
            early_stopping_rounds=20,
            random_state=RANDOM_STATE,
            verbosity=0
        )
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature preparation specific to XGBoost"""
        features = data.copy()
        
        # XGBoost handles numerical features directly
        if 'pickup_datetime' in features.columns:
            features = features.drop('pickup_datetime', axis=1)
        if self.target in features.columns:
            features = features.drop(self.target, axis=1)
            
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
        # Ensure predictions are not negative for fares
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
    def __init__(self, output_path:str,target: str = 'fare_amount'):
        super().__init__('random_forest', target, output_path, model_type='advanced')
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            oob_score=True,  # Configurar aquí directamente
            bootstrap=True,  # Asegurar bootstrap para OOB
            random_state=RANDOM_STATE,
            n_jobs=-1,
            warm_start=False,  # Para posibles entrenamientos incrementales
            verbose=0
        )
        
        self.oob_score_ = None
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Random Forest"""
        features = data.copy()
        
        if 'pickup_datetime' in features.columns:
            features = features.drop('pickup_datetime', axis=1)
        if self.target in features.columns:
            features = features.drop(self.target, axis=1)
            
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the Random Forest model with OOB score"""
        # Enable OOB score
        self.model.oob_score = True
        
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        self.oob_score_ = self.model.oob_score_
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest"""
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        # Ensure predictions are not negative for fares
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
            'importance_cv': importance_std / (importance_scores + 1e-10)  # Coefficient of variation
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics

class XGBoostModel(BaseModel):
    def __init__(self, model_path:str,target: str = 'fare_amount'):
        super().__init__('xgboost', target, model_path)
        self.model = XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE
        )
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara características específicas para XGBoost"""
        features = data.copy()
        
        # XGBoost maneja bien las características numéricas directamente
        if 'pickup_datetime' in features.columns:
            features = features.drop('pickup_datetime', axis=1)
        if self.target in features.columns:
            features = features.drop(self.target, axis=1)
            
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo XGBoost con validación opcional"""
        X_prepared = self._prepare_features(X_train)

        self.model.fit(X_prepared, y_train)
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones usando XGBoost"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        # Asegurar que las predicciones no sean negativas para tarifas
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evalúa el modelo XGBoost"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Agregar métricas específicas de XGBoost
        self.metrics.update({
            'n_estimators_used': self.model.get_booster().num_boosted_rounds(),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene la importancia de características"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        feature_names = self.model.get_booster().feature_names
        
        # Diferentes tipos de importancia
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
    def __init__(self, model_path:str,target: str = 'fare_amount'):
        super().__init__('random_forest', target, model_path)
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        self.oob_score_ = None
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara características para Random Forest"""
        features = data.copy()
        
        if 'pickup_datetime' in features.columns:
            features = features.drop('pickup_datetime', axis=1)
        if self.target in features.columns:
            features = features.drop(self.target, axis=1)
            
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo Random Forest con OOB score"""
        # Habilitar OOB score
        self.model.oob_score = True
        
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        self.oob_score_ = self.model.oob_score_
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones usando Random Forest"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        # Asegurar que las predicciones no sean negativas para tarifas
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evalúa el modelo Random Forest"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Agregar métricas específicas de Random Forest
        self.metrics.update({
            'oob_score': self.oob_score_,
            'n_trees': self.model.n_estimators
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene la importancia de características"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        feature_names = self.model.feature_names_in_
        
        # Importancia promedio y por árbol
        importance_scores = self.model.feature_importances_
        
        # Calcular variabilidad de importancia entre árboles
        tree_importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        importance_std = np.std(tree_importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_scores,
            'importance_std': importance_std,
            'importance_cv': importance_std / (importance_scores + 1e-10)  # Coeficiente de variación
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    
"""
Módulo de Implementación de Modelos Avanzados de Regresión.

Este módulo define las clases concretas para los algoritmos XGBoost y Random Forest,
heredando de la clase base abstracta. Encapsula la lógica de configuración, entrenamiento,
predicción y evaluación específica para cada arquitectura de modelo.
"""

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics

class XGBoostModel(BaseModel):
    def __init__(self, output_path:str, target: str = 'fare_amount',
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
        Inicializa el modelo de Gradient Boosting (XGBoost).
        
        Configura los hiperparámetros del regresor XGBoost para optimizar el rendimiento
        en tareas de regresión tabular. Los valores por defecto están ajustados para
        balancear la capacidad de generalización y el tiempo de entrenamiento.

        Parámetros:
        -----------
        output_path : str
            Ruta base para la persistencia de artefactos del modelo.
        target : str
            Nombre de la variable objetivo a predecir.
        n_estimators : int
            Número de árboles de decisión (boosting rounds).
        max_depth : int
            Profundidad máxima de cada árbol, controlando la complejidad del modelo.
        learning_rate : float
            Tasa de aprendizaje (eta) para la actualización de pesos.
        subsample : float
            Fracción de muestras utilizadas para entrenar cada árbol (bagging).
        colsample_bytree : float
            Fracción de características utilizadas por árbol.
        tree_method : str
            Algoritmo de construcción del árbol ('hist' optimizado para velocidad).
        kwargs : dict
            Parámetros adicionales pasados directamente al constructor de XGBRegressor.
        """
        super().__init__('xgboost', target, output_path, model_type='advanced')
        
        # Configuración detallada de hiperparámetros
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
            'objective': 'reg:squarederror', # Función de pérdida para regresión
            'eval_metric': 'rmse',           # Métrica de evaluación durante entrenamiento
            'n_jobs': -1,                    # Paralelización utilizando todos los núcleos disponibles
            'random_state': RANDOM_STATE,    # Reproducibilidad
            'verbosity': 0,
        }
        
        # Ajustes específicos para métodos basados en histogramas
        if tree_method in ['hist', 'gpu_hist']:
            xgb_params['grow_policy'] = grow_policy
            if max_leaves > 0:
                xgb_params['max_leaves'] = max_leaves
        
        xgb_params.update(kwargs)
        
        self.model = XGBRegressor(**xgb_params)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesamiento específico para XGBoost antes del entrenamiento/predicción.
        Actualmente pasa los datos directamente, pero permite inyectar lógica futura
        (ej. manejo de matrices dispersas).
        """
        features = data.copy()
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrena el modelo XGBoost con los datos proporcionados.
        
        Parámetros:
        -----------
        X_train : pd.DataFrame
            Matriz de características de entrenamiento.
        y_train : pd.Series
            Vector de la variable objetivo.
        """
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones utilizando el modelo entrenado.
        Aplica una rectificación (ReLU) para asegurar que no se produzcan valores negativos,
        ya que tarifas y duraciones deben ser estrictamente no negativas.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        
        # Rectificación de salida: max(0, predicción)
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evalúa el rendimiento del modelo en un conjunto de datos independiente.
        
        Retorna:
        --------
        dict
            Diccionario con métricas estándar (R2, MAE, RMSE) y metadatos específicos
            del modelo (número de estimadores utilizados, mejor iteración).
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Enriquecimiento con métricas internas de XGBoost
        self.metrics.update({
            'n_estimators_used': self.model.get_booster().num_boosted_rounds(),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extrae y organiza la importancia de las características según el modelo.
        Calcula múltiples tipos de importancia (gain, weight, cover) para un análisis robusto.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame ordenado por ganancia de información (gain_importance).
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        feature_names = self.model.get_booster().feature_names
        
        # Extracción de métricas de importancia nativas de XGBoost
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
    def __init__(self, output_path:str, target: str = 'fare_amount',
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
        Inicializa el modelo de Random Forest Regressor.
        
        Configura un ensamble de bagging robusto, ideal para establecer líneas base de rendimiento
        y analizar la importancia de características sin el riesgo de sobreajuste excesivo.

        Parámetros:
        -----------
        n_estimators : int
            Número de árboles en el bosque.
        max_depth : int
            Profundidad máxima de los árboles.
        min_samples_split : int
            Número mínimo de muestras requeridas para dividir un nodo interno.
        min_samples_leaf : int
            Número mínimo de muestras requeridas en un nodo hoja.
        max_features : str
            Número de características a considerar al buscar la mejor división ('sqrt' es estándar).
        bootstrap : bool
            Si se utilizan muestras bootstrap para construir árboles.
        oob_score : bool
            Si se utiliza out-of-bag samples para estimar la precisión de generalización.
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
        """Preprocesamiento específico para Random Forest."""
        features = data.copy()
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrena el modelo Random Forest.
        Captura el puntaje Out-of-Bag (OOB) si está habilitado, proporcionando una
        estimación de error de validación cruzada interna sin necesidad de un set separado.
        """
        # Verificación defensiva de configuración OOB
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
            self.model.oob_score = True
        
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        
        # Almacenamiento del OOB score post-entrenamiento
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
             self.oob_score_ = self.model.oob_score_

        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicción con Random Forest.
        Aplica rectificación de valores negativos (ReLU) en la salida.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evalúa el modelo y retorna métricas de rendimiento.
        Incluye el OOB score como métrica adicional de diagnóstico de sobreajuste.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        self.metrics.update({
            'oob_score': self.oob_score_,
            'n_trees': self.model.n_estimators
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calcula la importancia de características basada en la disminución media de impureza (MDI).
        
        Además de la media, calcula la desviación estándar de la importancia entre todos los árboles
        del bosque, permitiendo evaluar la estabilidad de la importancia de cada variable.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            # Generación de nombres genéricos si no están disponibles en el objeto modelo
            n_features = len(self.model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Importancia media global
        importance_scores = self.model.feature_importances_
        
        # Análisis de variabilidad de importancia entre árboles individuales
        tree_importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        importance_std = np.std(tree_importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_scores,
            'importance_std': importance_std,
            'importance_cv': importance_std / (importance_scores + 1e-10) # Coeficiente de variación
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
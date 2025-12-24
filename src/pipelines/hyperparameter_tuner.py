"""
Módulo de Optimización de Hiperparámetros (HPO).

Este módulo implementa un marco de trabajo flexible basado en Optuna para la búsqueda
automática de la configuración óptima de los modelos. Define una arquitectura base
abstracta que gestiona el ciclo de vida de la optimización (carga de datos, definición
de la función objetivo, validación cruzada) y clases concretas que definen el espacio
de búsqueda específico para cada algoritmo (XGBoost, Random Forest, etc.).
"""

import optuna
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict, Any, Optional
from pathlib import Path
import joblib

from src.config.paths import LOGGER_NAME
from src.config.settings import RANDOM_STATE
from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel
from src.utils.logging import LoggerFactory

# Configuración del logger para el módulo de optimización
logger = LoggerFactory.create_logger(
    name=LOGGER_NAME,
    log_level='DEBUG',
    console_output=True,
    file_output=False
)

class HyperparameterTuner(ABC):
    """
    Clase base abstracta para la orquestación de tuning de hiperparámetros.
    
    Centraliza la lógica repetitiva del proceso de optimización bayesiana:
    1. Carga y reconstrucción del dataset de entrenamiento.
    2. Definición de la función objetivo para Optuna.
    3. Evaluación robusta mediante K-Fold Cross-Validation.
    4. Gestión del estudio de optimización (pruning, sampling).
    """
    
    def __init__(self, 
                 model_class, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
        """
        Inicializa el orquestador de tuning.

        Args:
            model_class: Referencia a la clase del modelo (no instancia) que hereda de BaseModel.
            data_path (str): Ruta al archivo .pkl generado por el DataSplitter.
            output_path (str): Directorio donde se guardarán los artefactos del modelo.
            target (str): Nombre de la variable objetivo.
            cv_folds (int): Número de pliegues para la validación cruzada.
            n_trials (int): Presupuesto de intentos para la optimización.
            direction (str): Dirección de la optimización ('minimize' para error, 'maximize' para métricas como R2).
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
        Carga y prepara los datos para el proceso de validación cruzada.
        
        Reconstruye un único conjunto de datos (X, y) concatenando los splits de 
        entrenamiento, prueba y validación. Esto es necesario porque Optuna realizará
        su propia división interna mediante K-Fold para evaluar la generalización
        de cada configuración de hiperparámetros.
        
        Además, aplica un muestreo estratégico si el volumen de datos excede un umbral
        seguro, garantizando que la optimización se ejecute en tiempos razonables.
        """
        try:
            logger.info("Loading data from data_splitter...")
            
            self.data = joblib.load(self.data_path)
            self.features = self.data['features']
            
            # Recuperación de los subconjuntos procesados
            X_train = pd.DataFrame(self.data['X_train'][self.features])
            X_test = pd.DataFrame(self.data['X_test'][self.features])
            X_val = pd.DataFrame(self.data['X_val'][self.features])
            
            # Unificación de características para CV
            self.X = pd.concat([X_train, X_test, X_val], axis=0, ignore_index=True)
            
            # Unificación de targets para CV
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

            # Limitación de datos para optimización eficiente
            # Usar todo el dataset en cada trial de Optuna es computacionalmente costoso.
            # Se usa una muestra representativa para encontrar los mejores hiperparámetros.
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
        """
        Define el espacio de búsqueda de hiperparámetros específico para cada algoritmo.
        Debe ser implementado por las subclases.
        """
        pass
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo que Optuna intenta minimizar o maximizar.
        
        Flujo por intento (trial):
        1. Muestrea una configuración de hiperparámetros.
        2. Instancia el modelo con dicha configuración.
        3. Evalúa el modelo mediante K-Fold Cross-Validation.
        4. Retorna el promedio de la métrica de evaluación.
        """
        
        try:
            # 1. Sugerencia de parámetros
            params = self.suggest_hyperparameters(trial)
            
            # 2. Instanciación del modelo candidato
            model = self.model_class(
                output_path=self.output_path,
                target=self.target,
                **params
            )
            
            # 3. Validación Cruzada (K-Fold)
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            scores = []
            for train_idx, val_idx in kfold.split(self.X):
                X_train_fold = self.X.iloc[train_idx]
                X_val_fold = self.X.iloc[val_idx]
                y_train_fold = self.y.iloc[train_idx]
                y_val_fold = self.y.iloc[val_idx]
                
                # Entrenamiento en el fold actual
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluación en el fold de validación
                metrics = model.evaluate(X_val_fold, y_val_fold)
                
                # Extracción de la métrica principal
                primary_metric = self._get_primary_metric(metrics)
                scores.append(primary_metric)
            
            # Retorna el promedio de los folds como puntaje final del trial
            return np.mean(scores)
            
        except Exception as e:
            # Manejo robusto de errores para evitar que un trial fallido detenga todo el estudio
            logger.warning(f"Trial failed with parameters {params}: {e}")
            return float('inf') if self.direction == "minimize" else float('-inf')
    
    def _get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """
        Selecciona la métrica prioritaria para guiar la optimización.
        El orden de preferencia es RMSE -> MAE -> MSE.
        """
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
        """
        Ejecuta el ciclo completo de optimización.
        
        Configura el sampler TPE (Tree-structured Parzen Estimator) para una búsqueda bayesiana
        eficiente y un pruner para detener prematuramente los intentos no prometedores.
        """
        
        
        # Creación del estudio
        study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            storage=storage
        )
        
        # Configuración de estrategias de muestreo y poda
        study.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
        
        # Ejecución de la optimización
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Registro de resultados
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return study
    
    def get_best_model(self, **additional_params):
        """
        Instancia un nuevo modelo configurado con los mejores hiperparámetros encontrados.
        """
        if self.best_params is None:
            raise ValueError("Must run optimization first")
        
        # Fusión de parámetros optimizados con configuraciones adicionales
        params = {**self.best_params, **additional_params}
        return self.model_class(
            output_path=self.output_path,
            target=self.target,
            **params
        )
        
class LinearRegressionTuner(HyperparameterTuner):
    """
    Especialización para el ajuste de Regresión Lineal.
    Aunque tiene pocos hiperparámetros, el tuning puede ayudar a decidir sobre
    el ajuste del intercepto o la restricción de positividad.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
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
        """Definición del espacio de búsqueda para Regresión Lineal."""
        return {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }
        
class DecisionTreeTuner(HyperparameterTuner):
    """
    Especialización para Árboles de Decisión.
    Busca controlar la complejidad del árbol (profundidad, división mínima)
    para encontrar el balance entre sesgo y varianza.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
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
        """Definición del espacio de búsqueda para Árboles de Decisión."""
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
    """
    Especialización para XGBoost.
    Implementa un espacio de búsqueda complejo que abarca arquitectura del árbol,
    regularización (L1/L2) y estrategias de muestreo estocástico.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
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
        Definición del espacio de búsqueda para XGBoost.
        Incluye lógica condicional para parámetros que dependen del método de construcción del árbol.
        """
        
        # Selección del método de árbol (afecta la validez de otros parámetros)
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
        
        # Configuración específica para métodos basados en histogramas (más rápidos)
        if tree_method in ['hist', 'gpu_hist']:
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            
            # 'max_leaves' solo es relevante si la política de crecimiento es 'lossguide'
            if params['grow_policy'] == 'lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 16, 256, step=16)
        
        return params
    
class RandomForestTuner(HyperparameterTuner):
    """
    Especialización para Random Forest.
    Se enfoca en optimizar el tamaño del ensamble y las restricciones de los árboles individuales.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 target: str = 'fare_amount',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 direction: str = "minimize"):
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
        """Definición del espacio de búsqueda para Random Forest."""
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
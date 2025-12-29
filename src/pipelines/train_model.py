"""
Pipeline de Entrenamiento y Optimización de Modelos.

Este script actúa como el punto de entrada principal para el ciclo de vida del aprendizaje automático.
Soporta dos modos operativos distintos:
1. 'train': Entrenamiento directo de modelos base con hiperparámetros predeterminados.
2. 'optimize': Búsqueda exhaustiva de hiperparámetros óptimos (HPO) mediante Grid Search o algoritmos bayesianos.

Integra la gestión de datos, entrenamiento, evaluación y persistencia de modelos en un flujo unificado.
"""

import sys
import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

# Importación de componentes del sistema
from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel
from src.models.base_model import BaseModel as basemo
from src.config.paths import FARE_MODEL_DATA_FILE, DURATION_MODEL_DATA_FILE, \
    BASELINE_MODEL_PATH, ADVANCED_MODEL_PATH, LOGGER_NAME
from src.config.settings import RANDOM_STATE

from src.pipelines.sklearn_hyperparameter_tuner import (LinearRegressionTuner, 
                                                        DecisionTreeTuner,
                                                        XGBoostTuner,
                                                        RandomForestTuner,
                                                        SklearnHyperparameterTuner)

from src.utils.logging import LoggerFactory

# Configuración del logger centralizado
logger = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )

class ModelTrainer:
    """
    Clase orquestadora para el entrenamiento y evaluación de modelos.
    
    Encapsula la lógica para cargar datos preprocesados, instanciar múltiples arquitecturas
    de modelos, ejecutar el entrenamiento supervisado y recopilar métricas de rendimiento.
    """
    
    def __init__(self, data_path, models:list[basemo]):
        """
        Inicializa el entrenador.

        Args:
            data_path (str): Ruta al archivo .pkl generado por el DataSplitter.
            models (list): Lista de instancias de modelos (heredados de BaseModel) a entrenar.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        # Inicialización de contenedores de datos y resultados
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.features = None
        self.feature_scores = None
        
        self.trained_models = {}
        self.model_results = {}
        
        self.model_instances = models
        
    def load_data(self):
        """
        Carga los conjuntos de datos de entrenamiento, validación y prueba desde el disco.
        
        Implementa una estrategia de muestreo defensivo (downsampling) para limitar
        el tamaño del dataset de entrenamiento a 300k registros. Esto previene errores
        de memoria (OOM - Out Of Memory) en entornos con recursos limitados (ej. contenedores Docker),
        manteniendo una muestra estadísticamente significativa para el aprendizaje.
        """
        try:
            logger.info("Loading data from data_splitter...")
            
            self.data = joblib.load(self.data_path)
            
            self.features = self.data['features']
        
            self.X_train = self.data['X_train'][self.features]
            self.X_test = self.data['X_test'][self.features]
            self.X_val = self.data['X_val'][self.features]
            self.y_train = self.data['y_train']
            self.y_test = self.data['y_test']
            self.y_val = self.data['y_val']
            self.feature_scores = self.data['feature_scores']
            
            # --- GESTIÓN DE MEMORIA: MUESTREO ESTRATÉGICO ---
            MAX_SAMPLES = 300000 
            if len(self.X_train) > MAX_SAMPLES:
                logger.info(f"⚠️ Limitando datos de entrenamiento de {len(self.X_train)} a {MAX_SAMPLES} filas para estabilidad del sistema...")
                self.X_train = self.X_train.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
                self.y_train = self.y_train.loc[self.X_train.index]
            # ------------------------------------------------

            logger.info(f"✓ Data loaded successfully:")
            logger.info(f"  - X_train: {self.X_train.shape}")
            logger.info(f"  - X_test: {self.X_test.shape}")
            logger.info(f"  - y_train: {self.y_train.shape}")
            logger.info(f"  - y_test: {self.y_test.shape}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"❌ Error: Split data files not found.")
            logger.error(f"Make sure to run data_splitter.py first.")
            logger.error(f"Missing file: {e}")
            return False
        
    def train_model(self):
        """
        Itera sobre la lista de modelos configurados, ejecutando el ciclo de entrenamiento y evaluación.
        Registra métricas en los conjuntos de entrenamiento, validación y prueba para detectar sobreajuste.
        """
        
        logger.info("Training models...")
        
        for model_instance in self.model_instances:
            logger.info(f"\tTraining {model_instance.model_type} model: {model_instance.model_name}, Target: {model_instance.target}, Samples: {self.X_train.shape[0]}, Features: {self.X_train.shape[1]}")
            
            # Fase de ajuste (Fit)
            model_instance.fit(self.X_train, self.y_train)
            
            logger.info(f"\tEvaluating {model_instance.model_type} model: {model_instance.model_name} ...")
            
            # Fase de evaluación (Evaluate)
            train_metrics = model_instance.evaluate(self.X_train, self.y_train)
            test_metrics = model_instance.evaluate(self.X_test, self.y_test)
            val_metrics = model_instance.evaluate(self.X_val, self.y_val)
            
            results = {
                'model_instance': model_instance,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics
            }
            
            self.model_results[model_instance.model_name] = results
            logger.info(f" \t✓ {model_instance.model_name} trained successfully")
            
        logger.info("All models trained.")
    
    def save_trained_models(self):
        """
        Persiste los modelos entrenados en disco utilizando el protocolo pickle.
        Esto permite reutilizar los modelos para inferencia sin necesidad de reentrenamiento.
        """
        logger.info(f"\n💾 Saving models ...")
        
        saved_count = 0
        
        for model_name, results in self.model_results.items():
            try:
                model_instance = results['model_instance']
                model_path = model_instance.save_model()
                logger.info(f"  ✓ {model_name} saved to {model_path}")
                saved_count += 1
            except Exception as e:
                logger.error(f"  ❌ Error saving {model_name}: {e}")
        
        logger.info(f"✓ {saved_count} models saved successfully")
        return saved_count
    
    def print_results_summary(self):
        """
        Genera y muestra un reporte tabular comparativo del rendimiento de todos los modelos entrenados.
        Incluye una métrica de 'Overfitting' (diferencia entre R2 de Train y Test) para diagnóstico rápido.
        """
        if not self.model_results:
            logger.error("❌ No results to display")
            return
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*80)
        
        summary_data = []
        for model_name, results in self.model_results.items():
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            val_metrics = results['val_metrics']
            
            summary_data.append({
                'Modelo': model_name.replace('_', ' ').title(),
                'Train R²': f"{train_metrics.get('r2', 0):.4f}",
                'Test R²': f"{test_metrics.get('r2', 0):.4f}",
                'Test RMSE': f"{test_metrics.get('rmse', 0):.2f}",
                'Test MAE': f"{test_metrics.get('mae', 0):.2f}",
                'Val R²': f"{val_metrics.get('r2', 0):.4f}",
                'Val RMSE': f"{val_metrics.get('rmse', 0):.2f}",
                'Val MAE': f"{val_metrics.get('mae', 0):.2f}",
                'Overfitting': f"{(train_metrics.get('r2', 0) - test_metrics.get('r2', 0)):.4f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        logger.info("\n" + df_summary.to_string(index=False))
        
        # Identificación del mejor modelo basado en R2 del conjunto de prueba
        best_model_name, best_model = self.get_best_model()
        best_r2 = best_model['test_metrics'].get('r2', 0)
        logger.info(f"\n🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
        logger.info(f"  Test R²: {best_r2:.4f}")
        
    def get_best_model(self):
        """Identifica y retorna el modelo con mejor rendimiento en el set de prueba."""
        if not self.model_results:
            return None
        
        best_model_name = max(self.model_results.keys(), 
                              key=lambda x: self.model_results[x]['test_metrics'].get('r2', 0))
        
        return best_model_name, self.model_results[best_model_name]
    
class HyperparameterOptimizer:
    """
    Clase orquestadora para el flujo de trabajo de optimización de hiperparámetros (HPO).
    Gestiona múltiples 'tuners' (sintonizadores) para optimizar diferentes algoritmos en paralelo secuencial.
    """
    
    def __init__(self, data_path: str, tuners: list[SklearnHyperparameterTuner]):
        """
        Inicializa el optimizador.

        Args:
            data_path (str): Ruta al archivo de datos.
            tuners (list): Lista de objetos Tuner configurados para optimizar modelos específicos.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.tuner_instances = tuners
        self.optimization_results = {}
        self.best_models = {}
        
        
    def optimize_models(self):
        """
        Ejecuta el proceso de optimización para cada sintonizador registrado.
        Carga datos, ejecuta la búsqueda (Grid/Random) y limpia la memoria tras cada iteración.
        """
        logger.info("Starting hyperparameter optimization...")
        
        for i, tuner in enumerate(self.tuner_instances):
            model_name = f"{tuner.model_class.__name__}_{tuner.target}"
            
            logger.info(f"\n🔍 Optimizing {model_name} ({i+1}/{len(self.tuner_instances)})")
            logger.info(f"  Target: {tuner.target}")
            logger.info(f"  CV Folds: {tuner.cv_folds}")
            
            try:
                # Carga de datos específica para el tuner
                tuner.load_data()
                
                # Ejecución de la búsqueda de hiperparámetros
                study = tuner.optimize()
                study['tuner']= tuner

                # Liberación de recursos para el siguiente ciclo
                tuner.clear_data()
                
                # Almacenamiento de resultados
                self.optimization_results[model_name] = study
                
                logger.info(f"  ✓ Optimization completed")
                logger.info(f"  Best score: {tuner.best_score:.4f}")
                logger.info(f"  Best params: {tuner.best_params}")
                
            except Exception as e:
                logger.error(f"  ❌ Optimization failed for {model_name}: {e}")
                continue
        
        logger.info(f"\n✓ Hyperparameter optimization completed for {len(self.optimization_results)} models")
        
    def create_best_models(self):
        """Instancia los objetos de modelo finales utilizando los hiperparámetros óptimos encontrados."""
        logger.info("\n🏗️ Creating models with best hyperparameters...")
        
        for model_name, results in self.optimization_results.items():
            try:
                tuner = results['tuner']
                best_model = tuner.get_best_model()
                self.best_models[model_name] = best_model
                logger.info(f"  ✓ {model_name} created with optimized parameters")
            except Exception as e:
                logger.error(f"  ❌ Failed to create {model_name}: {e}")
        
        logger.info(f"✓ {len(self.best_models)} optimized models created")
        
    def train_best_models(self):
        """
        Entrena los modelos optimizados finales con el conjunto de datos completo (dentro de los límites de memoria).
        También calcula y registra la importancia de características si el modelo lo soporta.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("\n🚀 Training models with best hyperparameters...")
        
        self.trained_results = {}
        
        for model_name, model in self.best_models.items():
            try:
                logger.info(f"  Training {model_name}...")
                
                tuner = self.optimization_results[model_name]['tuner']
                tuner.load_data()
                
                # Validación de carga de datos
                if not (hasattr(tuner, 'X') and tuner.X is not None):
                    logger.error(f"❌ Error: No data loaded in tuner for {model_name}")
                    continue

                # Entrenamiento final
                model.fit(tuner.X, tuner.y)
                
                # Extracción de Feature Importance para interpretabilidad
                try:
                    # Intenta acceder al atributo feature_importances_ del modelo base o su wrapper
                    if hasattr(model, 'feature_importances_') or (hasattr(model, 'model') and hasattr(model.model, 'feature_importances_')):
                        base_model = model.model if hasattr(model, 'model') else model
                        if hasattr(base_model, 'feature_importances_'):
                            import pandas as pd
                            feature_imp = pd.DataFrame({
                                'Feature': tuner.X.columns,
                                'Importance': base_model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            logger.info(f"\n📊 Top 5 Features ({model_name}):\n{feature_imp.head(5)}")
                except Exception:
                    pass # La interpretabilidad es secundaria, no debe bloquear el flujo

                # Evaluación final
                metrics = model.evaluate(tuner.X, tuner.y)
                
                self.trained_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'best_params': self.optimization_results[model_name]['tuner'].best_params,
                    'best_score': self.optimization_results[model_name]['tuner'].best_score,
                    'cv_results': self.optimization_results[model_name].get('search_results', {}),
                    'method': self.optimization_results[model_name]['tuner'].method
                }

                tuner.clear_data()
                logger.info(f"  ✓ {model_name} trained successfully")
                
            except Exception as e:
                logger.error(f"  ❌ Training failed for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"✓ {len(self.trained_results)} models trained with optimal hyperparameters")

    def _convert_numpy_to_native(self, obj):
        """Utilería para convertir tipos NumPy a tipos nativos de Python para serialización JSON segura."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_native(item) for item in obj]
        else:
            return obj
        
    def save_optimized_models(self):
        """
        Guarda los modelos optimizados y un archivo JSON adjunto con metadatos del experimento
        (mejores parámetros, puntajes, fecha de ejecución).
        """
        logger.info(f"\n💾 Saving optimized models...")
        
        saved_count = 0
        
        for model_name, results in self.trained_results.items():
            try:
                model = results['model']
                model_path = model.save_model()
                
                cv_results_serializable = self._convert_numpy_to_native(results['cv_results'])
                
                optimization_info = {
                    'best_params': results['best_params'],
                    'best_score': float(results['best_score']),
                    'cv_results': cv_results_serializable,
                    'method': str(results['method']),
                    'optimization_date': datetime.now().isoformat(),
                    'model_path': str(model_path)
                }
                
                model_path = Path(model_path)
                info_path = model_path.parent / f"{model_path.stem}_optimization_info.json"
                import json
                with open(info_path, 'w') as f:
                    json.dump(optimization_info, f, indent=2)
                
                logger.info(f"  ✓ {model_name} saved to {model_path}")
                logger.info(f"  ✓ Optimization info saved to {info_path}")
                saved_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ Error saving {model_name}: {e}")
        
        logger.info(f"✓ {saved_count} optimized models saved successfully")
        return saved_count
    
    def print_optimization_summary(self):
        """Imprime un resumen ejecutivo de los resultados de la optimización."""
        if not self.optimization_results:
            logger.error("❌ No optimization results to display")
            return
        
        logger.info("\n" + "="*100)
        logger.info("HYPERPARAMETER OPTIMIZATION RESULTS SUMMARY")
        logger.info("="*100)
        
        summary_data = []
        for model_name, results in self.optimization_results.items():
            tuner = results['tuner']
            summary_data.append({
                'Model': model_name.replace('_', ' '),
                'Target': tuner.target,
                'Best Score': f"{tuner.best_score:.4f}",
                'CV Folds': tuner.cv_folds
            })
        
        df_summary = pd.DataFrame(summary_data)
        logger.info("\n" + df_summary.to_string(index=False))
        
        logger.info("\n" + "="*100)
        logger.info("BEST HYPERPARAMETERS")
        logger.info("="*100)
        
        for model_name, results in self.optimization_results.items():
            logger.info(f"\n🏆 {model_name}:")
            for param, value in results['tuner'].best_params.items():
                logger.info(f"  {param}: {value}")
                
    def get_best_model_overall(self):
        """Determina cuál fue el mejor modelo global basándose en el score de optimización."""
        if not self.optimization_results:
            return None
        
        best_model_name = min(self.optimization_results.keys(), 
                              key=lambda x: self.optimization_results[x]['tuner'].best_score)
        
        return best_model_name, self.optimization_results[best_model_name]['tuner']

def main(target_filter='both'):
    """
    Punto de entrada para el modo 'train' (Entrenamiento Baseline).
    Entrena modelos con configuración por defecto para establecer una línea base de rendimiento.
    """
    logger.info("🚀 Starting model training pipeline (Baseline Mode)")
    logger.info("="*50)
    
    trainers = []
    
    if target_filter in ['fare', 'both']:
        fare_amount_trainer = ModelTrainer(FARE_MODEL_DATA_FILE, 
                                         [LinearRegressionModel(output_path=BASELINE_MODEL_PATH, target='fare_amount'),
                                          DecisionTreeModel(output_path=BASELINE_MODEL_PATH, target='fare_amount'),
                                          XGBoostModel(output_path=ADVANCED_MODEL_PATH, target='fare_amount'),
                                          RandomForestModel(output_path=ADVANCED_MODEL_PATH, target='fare_amount')
                                         ])
        trainers.append(('Fare Amount', fare_amount_trainer))
    
    if target_filter in ['duration', 'both']:
        trip_duration_trainer = ModelTrainer(DURATION_MODEL_DATA_FILE, 
                                         [LinearRegressionModel(output_path=BASELINE_MODEL_PATH, target='trip_duration_minutes'),
                                          DecisionTreeModel(output_path=BASELINE_MODEL_PATH, target='trip_duration_minutes'),
                                          XGBoostModel(output_path=ADVANCED_MODEL_PATH, target='trip_duration_minutes'),
                                          RandomForestModel(output_path=ADVANCED_MODEL_PATH, target='trip_duration_minutes')
                                         ])
        trainers.append(('Trip Duration', trip_duration_trainer))
    
    for target_name, trainer in trainers:
        logger.info(f"\n🎯 Training models for {target_name}")
        if not trainer.load_data():
            logger.error(f"❌ Failed to load data for {target_name}. Skipping...")
            continue
        
        trainer.train_model()
        trainer.print_results_summary()
        trainer.save_trained_models()
        
    logger.info("\n✅ Training pipeline completed successfully!")
    
def main_optimization(target_filter='both', cv_folds=5, models='baseline'):
    """
    Punto de entrada para el modo 'optimize' (Búsqueda de Hiperparámetros).
    Configura y ejecuta la búsqueda de Grid Search o Random Search para los modelos seleccionados.
    """
    logger.info("🔍 Starting hyperparameter optimization pipeline (Optimization Mode)")
    logger.info("="*60)

    import joblib
    from src.config.paths import FARE_MODEL_DATA_FILE

    optimization_configs = []
    
    # Configuración de Tuners para Fare Amount
    if target_filter in ['fare', 'both']:
        fare_tuners = []
        if models in ['baseline', 'all']:
            if Path(FARE_MODEL_DATA_FILE).exists():
                fare_tuners.extend([
                    LinearRegressionTuner(
                        data_path=FARE_MODEL_DATA_FILE,
                        output_path=BASELINE_MODEL_PATH,
                        target='fare_amount',
                        method='grid_search'
                    ),
                    DecisionTreeTuner(
                        data_path=FARE_MODEL_DATA_FILE,
                        output_path=BASELINE_MODEL_PATH,
                        target='fare_amount',
                        method='grid_search',
                        cv_folds=cv_folds
                    )
                ])
        if models in ['advanced', 'all']:
             if Path(FARE_MODEL_DATA_FILE).exists():
                fare_tuners.extend([
                    XGBoostTuner(
                        data_path=FARE_MODEL_DATA_FILE,
                        output_path=ADVANCED_MODEL_PATH,
                        target='fare_amount',
                        method='grid_search',
                        cv_folds=cv_folds
                    ),
                    RandomForestTuner(
                        data_path=FARE_MODEL_DATA_FILE,
                        output_path=ADVANCED_MODEL_PATH,
                        target='fare_amount',
                        method='grid_search',
                        cv_folds=cv_folds
                    )
                ])
        
        if fare_tuners:
            optimization_configs.append(('Fare Amount', fare_tuners))
        else:
            logger.warning("No fare models or data available for optimization.")
    
    # Configuración de Tuners para Trip Duration
    if target_filter in ['duration', 'both']:
        duration_tuners = []
        if models in ['baseline', 'all']:
            if Path(DURATION_MODEL_DATA_FILE).exists():
                duration_tuners.extend([
                    LinearRegressionTuner(
                        data_path=DURATION_MODEL_DATA_FILE,
                        output_path=BASELINE_MODEL_PATH,
                        target='trip_duration_minutes',
                        method='grid_search'
                    ),
                    DecisionTreeTuner(
                        data_path=DURATION_MODEL_DATA_FILE,
                        output_path=BASELINE_MODEL_PATH,
                        target='trip_duration_minutes',
                        method='grid_search',
                        cv_folds=cv_folds
                    )
                ])

        if models in ['advanced', 'all']:
             if Path(DURATION_MODEL_DATA_FILE).exists():
                duration_tuners.extend([
                    XGBoostTuner(
                        data_path=DURATION_MODEL_DATA_FILE,
                        output_path=ADVANCED_MODEL_PATH,
                        target='trip_duration_minutes',
                        method='grid_search',
                        cv_folds=cv_folds
                    ),
                    RandomForestTuner(
                        data_path=DURATION_MODEL_DATA_FILE,
                        output_path=ADVANCED_MODEL_PATH,
                        target='trip_duration_minutes',
                        method='grid_search',
                        cv_folds=cv_folds
                    )
                ])
        
        if duration_tuners:
            optimization_configs.append(("Trip Duration", duration_tuners))
        else:
             logger.warning("No duration models or data available for optimization.")
    
    # Ejecución de la optimización
    for target_name, tuners in optimization_configs:
        logger.info(f"\n🎯 Optimizing models for {target_name}")
        logger.info("="*60)
        
        optimizer = HyperparameterOptimizer(tuners[0].data_path, tuners)
        optimizer.optimize_models()
        optimizer.create_best_models()
        optimizer.train_best_models()
        optimizer.save_optimized_models()
        optimizer.print_optimization_summary()
        
        best_model_name, best_tuner = optimizer.get_best_model_overall()
        logger.info(f"\n🏆 BEST {target_name.upper()} MODEL: {best_model_name}")
        logger.info(f"  Best Score (Neg MSE): {best_tuner.best_score:.4f}")
        
    logger.info("\n✅ Hyperparameter optimization pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NYC Taxi Model Training and Optimization Pipeline')
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'optimize'], 
        default='train',
        help='Modo de ejecución: entrenamiento estándar (train) o búsqueda de hiperparámetros (optimize)'
    )
    parser.add_argument(
        '--target',
        choices=['fare', 'duration', 'both'],
        default='both',
        help='Variable objetivo a modelar: tarifa (fare), duración (duration) o ambas'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Número de pliegues para validación cruzada (solo modo optimize)'
    )

    parser.add_argument(
        '--models',
        choices=['baseline', 'advanced', 'all'],
        default='advanced',
        help='Familia de modelos a entrenar/optimizar (baseline=lineal/árbol, advanced=xgb/rf)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info(f"🚀 Running in TRAINING mode for target: {args.target}")
        main(target_filter=args.target)
    elif args.mode == 'optimize':
        logger.info(f"🔍 Running in OPTIMIZATION mode for target: {args.target}")
        logger.info(f"  CV Folds: {args.cv_folds}")
        main_optimization(target_filter=args.target, cv_folds=args.cv_folds, models=args.models)
import sys
import os

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

# Importar tus clases de modelos
from src.models.baseline import LinearRegressionModel, DecisionTreeModel
from src.models.advanced import XGBoostModel, RandomForestModel
from src.models.base_model import BaseModel as basemo
from src.config.paths import FARE_MODEL_DATA_FILE, DURATION_MODEL_DATA_FILE, \
    BASELINE_MODEL_PATH, ADVANCED_MODEL_PATH, LOGGER_NAME
from src.config.settings import RANDOM_STATE

from src.utils.logging import LoggerFactory
logger = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )

class ModelTrainer:
    def __init__(self, data_path, models_output_path, models:list[basemo]):
        """
            Initialize the ModelTrainer with paths and data structures.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.models_output_path = Path(models_output_path)
        self.models_output_path.mkdir(parents=True, exist_ok=True)
        
        self.data = joblib.load(self.data_path)
        
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
            Load split data from the data splitter output
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
        logger.info("Training models...")
        
        for model_instance in self.model_instances:
            logger.info(f"Training model: {model_instance.model_name} - Target: {model_instance.target}")
            
            # train the model using your class method
            model_instance.fit(self.X_train, self.y_train)
            # Evaluar el modelo
            train_metrics = model_instance.evaluate(self.X_train, self.y_train)
            test_metrics = model_instance.evaluate(self.X_test, self.y_test)
            val_metrics = model_instance.evaluate(self.X_val, self.y_val)
            
            results = {
                'model_instance': model_instance,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics,
                'train_time': datetime.now()
            }
            
            self.model_results[model_instance.model_name] = results
            logger.info(f"  ✓ {model_instance.model_name} trained successfully")
            
        logger.info("All models trained.")
    
    def save_trained_models(self):
        """Guardar todos los modelos entrenados"""
        print(f"\n💾 Guardando modelos en {self.models_output_path}...")
        
        saved_count = 0
        
        for model_name, results in self.model_results.items():
            try:
                model_instance = results['model_instance']
                
                # Usar el método save de tu clase si existe, sino usar joblib
                model_path = self.models_output_path / f"{model_name}_model.joblib"
                
                if hasattr(model_instance, 'save'):
                    model_instance.save()
                else:
                    joblib.dump(model_instance, model_path)
                
                print(f"  ✓ {model_name} guardado en {model_path}")
                saved_count += 1
                
            except Exception as e:
                print(f"  ❌ Error guardando {model_name}: {e}")
        
        # Guardar resumen de resultados
        self._save_results_summary()
        
        print(f"✓ {saved_count} modelos guardados exitosamente")
        return saved_count
    
    def _save_results_summary(self):
        """Guardar resumen de resultados en CSV y pickle"""
        
        # Guardar pickle con todos los detalles
        pickle_path = self.models_output_path / 'training_results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.model_results, f)
        
        print(f"  ✓ Detalles guardados en {pickle_path}")
    
    def print_results_summary(self):
        """Imprimir resumen detallado de resultados"""
        if not self.model_results:
            print("❌ No hay resultados para mostrar")
            return
        
        print("\n" + "="*80)
        print("RESUMEN DE RESULTADOS DEL ENTRENAMIENTO")
        print("="*80)
        
        # Crear DataFrame para mostrar resultados
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
        print(df_summary.to_string(index=False))
        
        # Identificar mejor modelo
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_metrics'].get('r2', 0))
        best_r2 = self.model_results[best_model_name]['test_metrics'].get('r2', 0)
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name.replace('_', ' ').title()}")
        print(f"   R² en test: {best_r2:.4f}")
        
    def get_best_model(self):
        """Obtener el mejor modelo basado en R² de test"""
        if not self.model_results:
            return None
        
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_metrics'].get('r2', 0))
        
        return best_model_name, self.model_results[best_model_name]['model_instance']

def main():
    """Función principal para ejecutar el entrenamiento"""
    print("🚀 Iniciando pipeline de entrenamiento de modelos")
    print("="*50)
    
    # Inicializar trainer
    fare_base_trainer = ModelTrainer(FARE_MODEL_DATA_FILE, BASELINE_MODEL_PATH, 
                           [LinearRegressionModel(model_path=f"{BASELINE_MODEL_PATH}/fare_amount_linear_regression.pkl", target='fare_amount'),
                            DecisionTreeModel(model_path=f"{BASELINE_MODEL_PATH}/fare_amount_decision_tree.pkl", target='fare_amount')
                            ])
    fare_advanced_trainer = ModelTrainer(FARE_MODEL_DATA_FILE, ADVANCED_MODEL_PATH, 
                           [XGBoostModel(model_path=f"{ADVANCED_MODEL_PATH}/fare_amount_xgboost.pkl", target='fare_amount'),
                            RandomForestModel(model_path=f"{ADVANCED_MODEL_PATH}/fare_amount_random_forest.pkl", target='fare_amount')
                            ])
    
    trip_dur_base_trainer = ModelTrainer(DURATION_MODEL_DATA_FILE, BASELINE_MODEL_PATH, 
                           [LinearRegressionModel(model_path=f"{BASELINE_MODEL_PATH}/trip_duration_linear_regression.pkl", target='trip_duration_minutes'),
                            DecisionTreeModel(model_path=f"{BASELINE_MODEL_PATH}/trip_duration_decision_tree.pkl", target='trip_duration_minutes')
                            ])
    trip_dur_advanced_trainer = ModelTrainer(DURATION_MODEL_DATA_FILE, ADVANCED_MODEL_PATH, 
                           [XGBoostModel(model_path=f"{ADVANCED_MODEL_PATH}/trip_duration_xgboost.pkl", target='trip_duration_minutes'),
                            RandomForestModel(model_path=f"{ADVANCED_MODEL_PATH}/trip_duration_random_forest.pkl", target='trip_duration_minutes')
                            ])
    
    for trainer in [fare_base_trainer, fare_advanced_trainer, trip_dur_base_trainer, trip_dur_advanced_trainer]:
        # Cargar datos
        if not trainer.load_data():
            print("❌ No se pudieron cargar los datos. Saliendo...")
            return
        
        # Entrenar modelos
        trainer.train_model()
        
        # Mostrar resultados
        trainer.print_results_summary()
        
        # Guardar modelos
        trainer.save_trained_models()
        
        # Obtener mejor modelo
        best_name, best_model = trainer.get_best_model()
        if best_model:
            print(f"\n💡 Considera usar '{best_name}' para producción")
    
    print("\n✅ Pipeline de entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main()
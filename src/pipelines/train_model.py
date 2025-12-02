import sys
import os

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

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
    def __init__(self, data_path, models:list[basemo]):
        """
            Initialize the ModelTrainer with paths and data structures.
            
            Args:
                data_path (str): Path to the split data file.
                models (list): List of model instances to train.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
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
        """
            Train all provided models
        """
        
        logger.info("Training models...")
        
        for model_instance in self.model_instances:
            logger.info(f"\tTraining {model_instance.model_type} model: {model_instance.model_name}, Target: {model_instance.target}, Samples: {self.X_train.shape[0]}, Features: {self.X_train.shape[1]}")
            
            # train the model using your class method
            model_instance.fit(self.X_train, self.y_train)
            
            logger.info(f"\tEvaluating {model_instance.model_type} model: {model_instance.model_name} ...")
            # Eval the model
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
        """Save all trained models"""
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
        """Print detailed results summary"""
        if not self.model_results:
            logger.error("❌ No results to display")
            return
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*80)
        
        # Create DataFrame to display results
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
        
        # Identify best model
        best_model_name, best_model = self.get_best_model()
        
        best_r2 = best_model['test_metrics'].get('r2', 0)
        logger.info(f"\n🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
        logger.info(f"   Test R²: {best_r2:.4f}")
        
    def get_best_model(self):
        """Get the best model based on test R²"""
        if not self.model_results:
            return None
        
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_metrics'].get('r2', 0))
        
        return best_model_name, self.model_results[best_model_name]

def main():
    """Main function to execute the training pipeline"""
    logger.info("🚀 Starting model training pipeline")
    logger.info("="*50)
    
    # Initialize trainer
    fare_amount_trainer = ModelTrainer(FARE_MODEL_DATA_FILE, 
                           [LinearRegressionModel(output_path=BASELINE_MODEL_PATH, target='fare_amount'),
                            DecisionTreeModel(output_path=BASELINE_MODEL_PATH, target='fare_amount'),
                            XGBoostModel(output_path=ADVANCED_MODEL_PATH, target='fare_amount'),
                            RandomForestModel(output_path=ADVANCED_MODEL_PATH, target='fare_amount')
                            ])
    
    trip_duration_trainer = ModelTrainer(DURATION_MODEL_DATA_FILE, 
                           [LinearRegressionModel(output_path=BASELINE_MODEL_PATH, target='trip_duration_minutes'),
                            DecisionTreeModel(output_path=BASELINE_MODEL_PATH, target='trip_duration_minutes'),
                            XGBoostModel(output_path=ADVANCED_MODEL_PATH, target='trip_duration_minutes'),
                            RandomForestModel(output_path=ADVANCED_MODEL_PATH, target='trip_duration_minutes')
                            ])
    
    for trainer in [fare_amount_trainer, trip_duration_trainer]:
        # Load data
        if not trainer.load_data():
            logger.error("❌ Failed to load data. Exiting...")
            return
        
        # Train models
        trainer.train_model()
        
        # Show results
        trainer.print_results_summary()
        
        # Save models
        trainer.save_trained_models()
        
        # Get best model
        best_name, best_model = trainer.get_best_model()
        if best_model:
            logger.info(f"\n💡 Consider using '{best_name}' for production")
    
    logger.info("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
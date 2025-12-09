import sys
import os
import argparse

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

from src.pipelines.sklearn_hyperparameter_tuner import (LinearRegressionTuner, 
                                               DecisionTreeTuner,
                                               XGBoostTuner,
                                               RandomForestTuner,
                                               SklearnHyperparameterTuner)

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
    
class HyperparameterOptimizer:
    def __init__(self, data_path: str, tuners: list[SklearnHyperparameterTuner]):
        """
        Initialize the HyperparameterOptimizer with paths and tuner instances.
        
        Args:
            data_path (str): Path to the split data file.
            tuners (list): List of tuner instances to optimize.
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.tuner_instances = tuners
        self.optimization_results = {}
        self.best_models = {}
        
        
    def optimize_models(self):
        """
        Run hyperparameter optimization for all tuners
        """
        logger.info("Starting hyperparameter optimization...")
        
        for i, tuner in enumerate(self.tuner_instances):
            model_name = f"{tuner.model_class.__name__}_{tuner.target}"
            
            logger.info(f"\n🔍 Optimizing {model_name} ({i+1}/{len(self.tuner_instances)})")
            logger.info(f"   Target: {tuner.target}")
            logger.info(f"   CV Folds: {tuner.cv_folds}")
            
            try:
                # Create study name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Load data into tuner
                tuner.load_data()
                
                # Run optimization
                study = tuner.optimize()
                study['tuner']= tuner
                
                # Store results
                self.optimization_results[model_name] = study
                
                logger.info(f"   ✓ Optimization completed")
                logger.info(f"   Best score: {tuner.best_score:.4f}")
                logger.info(f"   Best params: {tuner.best_params}")
                
            except Exception as e:
                logger.error(f"   ❌ Optimization failed for {model_name}: {e}")
                continue
        
        logger.info(f"\n✓ Hyperparameter optimization completed for {len(self.optimization_results)} models")
        
    def create_best_models(self):
        """
        Create model instances with best hyperparameters
        """
        logger.info("\n🏗️ Creating models with best hyperparameters...")
        
        for model_name, results in self.optimization_results.items():
            try:
                tuner = results['tuner']
                best_model = tuner.get_best_model()
                
                self.best_models[model_name] = best_model
                
                logger.info(f"   ✓ {model_name} created with optimized parameters")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to create {model_name}: {e}")
        
        logger.info(f"✓ {len(self.best_models)} optimized models created")
        
    def train_best_models(self):
        """
        Train the best models with their optimized hyperparameters
        """
        logger.info("\n🚀 Training models with best hyperparameters...")
        
        self.trained_results = {}
        
        for model_name, model in self.best_models.items():
            try:
                logger.info(f"   Training {model_name}...")
                
                # Get the corresponding tuner to access data
                tuner = self.optimization_results[model_name]['tuner']
                
                # Load data into tuner
                tuner.load_data()
                
                # Train the model
                model.fit(tuner.X, tuner.y)
                
                # Evaluate on the same data (you might want to use separate test set)
                metrics = model.evaluate(tuner.X, tuner.y)
                
                self.trained_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'best_params': self.optimization_results[model_name]['best_params'],
                    'best_score': self.optimization_results[model_name]['best_score'],
                    'cv_results': self.optimization_results[model_name]['cv_results'],
                    'method': self.optimization_results[model_name]['method']
                }
                
                logger.info(f"   ✓ {model_name} trained successfully")
                
            except Exception as e:
                logger.error(f"   ❌ Training failed for {model_name}: {e}")
        
        logger.info(f"✓ {len(self.trained_results)} models trained with optimal hyperparameters")
        
    def _convert_numpy_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
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
        """Save all optimized models"""
        logger.info(f"\n💾 Saving optimized models...")
        
        saved_count = 0
        
        for model_name, results in self.trained_results.items():
            try:
                model = results['model']
                model_path = model.save_model()
                
                # Convert cv_results to JSON-serializable format
                cv_results_serializable = self._convert_numpy_to_native(results['cv_results'])
                
                # Also save the optimization results
                optimization_info = {
                    'best_params': results['best_params'],
                    'best_score': float(results['best_score']),
                    'cv_results': cv_results_serializable,
                    'method': results['method'],
                    'optimization_date': datetime.now().isoformat(),
                    'model_path': str(model_path)
                }
                
                # Save optimization info alongside the model
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
        """Print detailed optimization results summary"""
        if not self.optimization_results:
            logger.error("❌ No optimization results to display")
            return
        
        logger.info("\n" + "="*100)
        logger.info("HYPERPARAMETER OPTIMIZATION RESULTS SUMMARY")
        logger.info("="*100)
        
        # Create DataFrame to display results
        summary_data = []
        for model_name, results in self.optimization_results.items():
            tuner = results['tuner']
            
            summary_data.append({
                'Model': model_name.replace('_', ' '),
                'Target': tuner.target,
                'Best Score': f"{results['best_score']:.4f}",
                'CV Folds': tuner.cv_folds
            })
        
        df_summary = pd.DataFrame(summary_data)
        logger.info("\n" + df_summary.to_string(index=False))
        
        # Show best parameters for each model
        logger.info("\n" + "="*100)
        logger.info("BEST HYPERPARAMETERS")
        logger.info("="*100)
        
        for model_name, results in self.optimization_results.items():
            logger.info(f"\n🏆 {model_name}:")
            for param, value in results['best_params'].items():
                logger.info(f"   {param}: {value}")
                
    def get_best_model_overall(self):
        """Get the overall best model across all optimizations"""
        if not self.optimization_results:
            return None
        
        # Find best model based on optimization score
        best_model_name = min(self.optimization_results.keys(), 
                            key=lambda x: self.optimization_results[x]['best_score'])
        
        return best_model_name, self.optimization_results[best_model_name]

def main(target_filter='both'):
    """Main function to execute the training pipeline"""
    logger.info("🚀 Starting model training pipeline")
    logger.info("="*50)
    
    # Initialize trainer
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
        # Load data
        if not trainer.load_data():
            logger.error(f"❌ Failed to load data for {target_name}. Skipping...")
            continue
        
        # Train models
        trainer.train_model()
        
        # Show results
        trainer.print_results_summary()
        
        # Save models
        trainer.save_trained_models()
        
    logger.info("\n✅ Training pipeline completed successfully!")
    
def main_optimization(target_filter='both', cv_folds=5, models='baseline'):
    """Main function to execute the hyperparameter optimization pipeline"""
    logger.info("🔍 Starting hyperparameter optimization pipeline")
    logger.info("="*60)
    
    optimization_configs = []
    
    # Define tuners for fare amount models
    if target_filter in ['fare', 'both']:
        fare_tuners = []
        if models in ['baseline', 'all']:
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
        
        optimization_configs.append(('Fare Amount', fare_tuners))
    
    # Define tuners for trip duration models
    if target_filter in ['duration', 'both']:
        duration_tuners = []
        if models in ['baseline', 'all']:
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
        
        optimization_configs.append(("Trip Duration", duration_tuners))
    
    # Run optimization for both targets
    for target_name, tuners in optimization_configs:
        logger.info(f"\n🎯 Optimizing models for {target_name}")
        logger.info("="*60)
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(tuners[0].data_path, tuners)
        
        # Run optimization
        optimizer.optimize_models()
        
        # Create and train best models
        optimizer.create_best_models()
        optimizer.train_best_models()
        
        # Save optimized models
        optimizer.save_optimized_models()

        # Show results
        optimizer.print_optimization_summary()
        
        # Get overall best model for this target
        best_model_name, best_results = optimizer.get_best_model_overall()
        logger.info(f"\n🏆 BEST {target_name.upper()} MODEL: {best_model_name}")
        logger.info(f"   Best Score: {best_results['best_score']:.4f}")
    
    logger.info("\n✅ Hyperparameter optimization pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NYC Taxi Model Training and Optimization Pipeline')
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'optimize'], 
        default='train',
        help='Choose between training baseline models (train) or optimizing hyperparameters (optimize)'
    )
    parser.add_argument(
        '--target',
        choices=['fare', 'duration', 'both'],
        default='both',
        help='Choose target variable: fare_amount (fare), trip_duration_minutes (duration), or both'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (only for optimize mode)'
    )

    parser.add_argument(
        '--models',
        choices=['baseline', 'advanced', 'all'],
        default='baseline',
        help='Models to train (only for optimize mode) - baseline, advanced, or all'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logger.info(f"🚀 Running in TRAINING mode for target: {args.target}")
        main(target_filter=args.target)
    elif args.mode == 'optimize':
        logger.info(f"🔍 Running in OPTIMIZATION mode for target: {args.target}")
        logger.info(f"   CV Folds: {args.cv_folds}")
        main_optimization(target_filter=args.target, cv_folds=args.cv_folds, models=args.models)
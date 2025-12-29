import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

from src.data.feature_selector import FeatureSelector
from src.config.settings import RANDOM_STATE, TEST_SIZE, VAL_SIZE, TRAIN_SIZE
from src.config.paths import LOGGER_NAME

from src.utils.logging import LoggerFactory

class DataSplitter:
    def __init__(self, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
        """
        Orchestrates the partitioning of the dataset for training, validation, and testing
        of the rate and duration estimation models.
        
        Ensures that both models use exactly the same row indices for
        each subset, ensuring fair comparability in the evaluation metrics.

        Parameters:
        -----------
        test_size : float
            Proportion of the total dataset that will be reserved exclusively for final evaluation (Hold-out set).
        val_size : float
            Proportion of the total dataset allocated to hyperparameter tuning and early stopping.
        random_state : int
            Random seed to ensure deterministic reproducibility of the splits.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.logging = LoggerFactory.create_logger(
            name=LOGGER_NAME,
            log_level='DEBUG',
            console_output=True,
            file_output=False
        )
        
        # Verify the mathematical consistency of the partition ratios.
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Ejecuta verificaciones de integridad sobre el DataFrame antes de procesarlo.
        Asegura que existan las columnas objetivo y que el volumen de datos sea suficiente.
        """
        required_columns = ['fare_amount', 'trip_duration_minutes']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Analyzes the quality of the target variables to warn of
        # a possible excessive loss of information if there are too many null values.
        for col in required_columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.5:
                self.logging.warning(f"High missing values in {col}: {missing_pct:.2%}")
                
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera la matriz de características (Features) eliminando las variables objetivo
        y aquellas columnas que podrían introducir 'Data Leakage' (fugas de información).
        """
        exclude_columns = ['fare_amount', 'trip_duration_minutes']
        
        # Explicitly exclude variables that contain future information 
        # (such as the actual arrival date) or direct components of the final price
        # (tips, tolls, total amount) to maintain the honesty of the model.
        additional_exclude = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'total_amount', 'tip_amount', 'tolls_amount'
        ]
        
        exclude_columns.extend([col for col in additional_exclude if col in df.columns])
        X_all = df.drop(columns=exclude_columns)
        
        return X_all
                
    def split_data_for_both_models(self, df: pd.DataFrame, 
                                 sample_for_feature_selection: int = 10000) -> Dict:
        """
        Execute the complete preparation flow: cleaning, stratified splitting, and
        selection of specific features for each objective.

        Strategy:
        1. Clean records without a valid target.
        2. Split Train/Val/Test while maintaining the same indices for both problems.
        3. Perform independent feature selection for ‘fare’ and ‘duration’,
        as the relevant predictors may vary between cost and time.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Complete dataset with features and objectives.
        sample_for_feature_selection : int
            Maximum number of samples to use during the feature selection stage
            to optimize computation time without sacrificing statistical accuracy.
        
        Returns:
        --------
        Dict
            Structured dictionary containing the divided datasets (X_train, y_train, etc.)
            and metadata about the process for each model.
        """
        
        # Performs structure and integrity validations.
        self._validate_data(df)
        
        # Discard records that lack values in the target variables,
        # as they do not contribute to supervised learning.
        df_clean = df.dropna(subset=['fare_amount', 'trip_duration_minutes'])
        
        if len(df_clean) < len(df):
            self.logging.info(f"Removed {len(df) - len(df_clean)} rows with missing targets")
        
        # Separate the feature matrix from the target vectors.
        X_all = self._prepare_features(df_clean)
        y_fare = df_clean['fare_amount']
        y_duration = df_clean['trip_duration_minutes']
        
        logging.info(f"Total samples: {len(df_clean)}")
        logging.info(f"Total features available: {len(X_all.columns)}")
        
        # Phase 1: Segregation of the test set.
        # It is isolated first to ensure that it is never seen during training.
        X_temp, X_test, y_fare_temp, y_fare_test, y_duration_temp, y_duration_test = train_test_split(
            X_all, y_fare, y_duration,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Phase 2: Division of the remaining set into Training and Validation.
        # The relative size of the validation set is adjusted with respect to the remainder.
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        X_train, X_val, y_fare_train, y_fare_val, y_duration_train, y_duration_val = train_test_split(
            X_temp, y_fare_temp, y_duration_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        self.logging.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Optimization: Use a representative sample to calculate feature importance.
        # This drastically reduces execution time in algorithms such as Boruta or RFE.
        sample_size = min(sample_for_feature_selection, len(X_train))
        if sample_size < len(X_train):
            sample_idx = np.random.RandomState(self.random_state).choice(
                len(X_train), size=sample_size, replace=False
            )
            X_sample = X_train.iloc[sample_idx]
            y_fare_sample = y_fare_train.iloc[sample_idx]
            y_duration_sample = y_duration_train.iloc[sample_idx]
        else:
            X_sample = X_train
            y_fare_sample = y_fare_train
            y_duration_sample = y_duration_train
        
        # Initialize the specialized selectors for each dependent variable.
        fare_selector = FeatureSelector('fare_amount')
        duration_selector = FeatureSelector('trip_duration_minutes')
        
        # Perform the identification of the most predictive variables for the rate.
        self.logging.info("Selecting features for fare model...")
        fare_features, fare_scores = fare_selector.select_features_for_fare(
            X_sample, y_fare_sample
        )
        
        # Perform identification of the most predictive variables for duration.
        self.logging.info("Selecting features for duration model...")
        duration_features, duration_scores = duration_selector.select_features_for_duration(
            X_sample, y_duration_sample
        )
        
        self.logging.info(f"Selected {len(fare_features)} features for fare model")
        self.logging.info(f"Selected {len(duration_features)} features for duration model")
        
        # Package the subsets filtered by the selected columns for the rate model.
        fare_data = {
            'X_train': X_train[fare_features],
            'X_val': X_val[fare_features],
            'X_test': X_test[fare_features],
            'y_train': y_fare_train,
            'y_val': y_fare_val,
            'y_test': y_fare_test,
            'features': fare_features,
            'feature_scores': fare_scores,
            'selector': fare_selector
        }
        
        # Package the subsets filtered by the selected columns for the duration model.
        duration_data = {
            'X_train': X_train[duration_features],
            'X_val': X_val[duration_features],
            'X_test': X_test[duration_features],
            'y_train': y_duration_train,
            'y_val': y_duration_val,
            'y_test': y_duration_test,
            'features': duration_features,
            'feature_scores': duration_scores,
            'selector': duration_selector
        }
        
        return {
            'fare_model': fare_data,
            'duration_model': duration_data,
            'split_info': {
                'total_samples': len(df_clean),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'train_ratio': len(X_train) / len(df_clean),
                'val_ratio': len(X_val) / len(df_clean),
                'test_ratio': len(X_test) / len(df_clean),
                'feature_selection_sample_size': sample_size
            }
        }
        
    def get_feature_overlap(self, splits_dict: Dict) -> Dict:
        """
        Analyze the intersection and divergence of characteristics between both models.
        Useful for diagnosis and for understanding which variables are universally 
        predictive versus which are specific to a single objective.
        """
        fare_features = set(splits_dict['fare_model']['features'])
        duration_features = set(splits_dict['duration_model']['features'])
        
        overlap = fare_features.intersection(duration_features)
        fare_only = fare_features - duration_features
        duration_only = duration_features - fare_features
        
        return {
            'overlap_features': list(overlap),
            'fare_only_features': list(fare_only),
            'duration_only_features': list(duration_only),
            'overlap_count': len(overlap),
            'fare_only_count': len(fare_only),
            'duration_only_count': len(duration_only)
        }
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
            Class to handle data splitting for both fare and duration models
            
            Parameters:
            test_size: float - Proportion of data to use for testing
            val_size: float - Proportion of data to use for validation
            random_state: int - Random seed for reproducibility
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
        
        # Validate proportions
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data before splitting"""
        required_columns = ['fare_amount', 'trip_duration_minutes']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for excessive missing values in targets
        for col in required_columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.5:
                self.logging.warning(f"High missing values in {col}: {missing_pct:.2%}")
                
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix by removing target variables"""
        exclude_columns = ['fare_amount', 'trip_duration_minutes']
        
        # Remove any columns that shouldn't be features
        additional_exclude = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'total_amount', 'tip_amount', 'tolls_amount'  # These could leak info
        ]
        
        exclude_columns.extend([col for col in additional_exclude if col in df.columns])
        X_all = df.drop(columns=exclude_columns)
        
        return X_all
                
    def split_data_for_both_models(self, df: pd.DataFrame, 
                                 sample_for_feature_selection: int = 10000) -> Dict:
        """
        Split data for both fare_amount and trip_duration_minutes models
        Uses same split to maintain consistency in evaluation
        
        Parameters:
        df: DataFrame with all features and targets
        sample_for_feature_selection: int - Sample size for feature selection (for efficiency)
        """
        
        # Validate input data
        self._validate_data(df)
        
        # Remove any rows with missing targets
        df_clean = df.dropna(subset=['fare_amount', 'trip_duration_minutes'])
        
        if len(df_clean) < len(df):
            self.logging.info(f"Removed {len(df) - len(df_clean)} rows with missing targets")
        
        # Prepare features
        X_all = self._prepare_features(df_clean)
        y_fare = df_clean['fare_amount']
        y_duration = df_clean['trip_duration_minutes']
        
        logging.info(f"Total samples: {len(df_clean)}")
        logging.info(f"Total features available: {len(X_all.columns)}")
        
        # Initial split: (train+val) vs test
        X_temp, X_test, y_fare_temp, y_fare_test, y_duration_temp, y_duration_test = train_test_split(
            X_all, y_fare, y_duration,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Split train+val: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        
        X_train, X_val, y_fare_train, y_fare_val, y_duration_train, y_duration_val = train_test_split(
            X_temp, y_fare_temp, y_duration_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        self.logging.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Feature selection using sample for efficiency
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
        
        # Initialize feature selectors
        fare_selector = FeatureSelector('fare_amount')
        duration_selector = FeatureSelector('trip_duration_minutes')
        
        # Select features for each model
        self.logging.info("Selecting features for fare model...")
        fare_features, fare_scores = fare_selector.select_features_for_fare(
            X_sample, y_fare_sample
        )
        
        self.logging.info("Selecting features for duration model...")
        duration_features, duration_scores = duration_selector.select_features_for_duration(
            X_sample, y_duration_sample
        )
        
        self.logging.info(f"Selected {len(fare_features)} features for fare model")
        self.logging.info(f"Selected {len(duration_features)} features for duration model")
        
        # Create model-specific datasets
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
        """Analyze feature overlap between models"""
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
"""
Module for preprocessing NYC Yellow Taxi Trip Records data.
Based on findings from the EDA conducted in 01_eda.ipynb.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from pathlib import Path
import sys
import os

# Add src to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import get_full_logger
from config.paths import LOGGER_NAME, RAW_DATA, PROCESSED_DATA
from config.settings import PREPROCESSING_PARAMS, LOG_LEVEL

warnings.filterwarnings('ignore')
logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)

class TaxiDataPreprocessor:
    """
    Class for cleaning NYC Yellow Taxi data based on findings from the EDA.
    
    Responsibilities:
    - Validate basic dataset structure
    - Clean temporal fields (invalid dates, anomalous durations)
    - Clean categorical fields (invalid values, outliers)
    - Clean monetary fields (negative values, discrepancies)
    - Validate and correct total_amount
    - Remove extreme outliers
    
    Critical issues identified in EDA:
    - 28.01% records with discrepancies in total_amount
    - Widespread negative values (0.57% fares, 0.58% totals)
    - Invalid VendorID=5 (14 records)
    - Dates out of range May 2022
    - 73,587 trips with 0 passengers (2.05%)
    - 46,438 trips with 0 distance (1.29%)
    - 135 anomalous tips in cash payments
    
    Note: Feature engineering is performed in TaxiFeatureEngineer.
    """
    
    def __init__(self, raw_data_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to the raw data file. If None, uses RAW_DATA from config.
        """
        self.raw_data_path = raw_data_path or RAW_DATA
        self.df = None
        self.original_shape = None
        self.preprocessing_stats = {}
        
        # Definir mappings según diccionario de datos TLC
        self.vendor_mapping = {
            1: "Creative Mobile Technologies LLC",
            2: "Curb Mobility LLC", 
            6: "Myle Technologies Inc",
            7: "Helix"
        }
        
        self.ratecode_mapping = {
            1: "Standard rate", 2: "JFK", 3: "Newark",
            4: "Nassau or Westchester", 5: "Negotiated fare", 
            6: "Group ride", 99: "Null/unknown"
        }
        
        self.payment_mapping = {
            0: "Flex Fare trip", 1: "Credit card", 2: "Cash", 
            3: "No charge", 4: "Dispute", 5: "Unknown", 6: "Voided trip"
        }
        
        # Definir campos monetarios según diccionario TLC
        self.monetary_fields = [
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
            'tolls_amount', 'improvement_surcharge', 'total_amount',
            'congestion_surcharge', 'airport_fee'
        ]
        
        logger.info("TaxiDataPreprocessor initialized")
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data from parquet file."""
        try:
            logger.info(f"Loading data from: {self.raw_data_path}")
            self.df = pd.read_parquet(self.raw_data_path)
            self.original_shape = self.df.shape
            logger.info(f"Data loaded: {self.original_shape[0]:,} rows, {self.original_shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_basic_structure(self) -> dict:
        """
        Validate basic structure and completeness of the dataset.
        
        Returns:
            dict: Validation statistics
        """
        logger.info("=== BASIC STRUCTURE VALIDATION ===")
        
        # Expected columns according to TLC data dictionary
        expected_columns = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type',
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
            'improvement_surcharge', 'total_amount', 'congestion_surcharge',
            'airport_fee'#, 'cbd_congestion_fee'
        ]
        
        actual_columns = set(self.df.columns)
        expected_set = set(expected_columns)
        
        missing_cols = expected_set - actual_columns
        extra_cols = actual_columns - expected_set
        
        validation_stats = {
            'total_records': len(self.df),
            'total_columns': len(actual_columns),
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        validation_stats['missing_values'] = {
            col: {'count': int(count), 'percentage': round(pct, 2)}
            for col, count, pct in zip(missing_values.index, missing_values.values, missing_percentage.values)
            if count > 0
        }
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        validation_stats['duplicates'] = {
            'count': int(duplicates),
            'percentage': round(duplicates/len(self.df)*100, 2)
        }
        
        logger.info(f"Columnas faltantes: {missing_cols}")
        logger.info(f"Columnas adicionales: {extra_cols}")
        logger.info(f"Registros con valores faltantes: {sum(v['count'] for v in validation_stats['missing_values'].values())}")
        logger.info(f"Registros duplicados: {duplicates}")
        
        self.preprocessing_stats['validation'] = validation_stats
        return validation_stats
    
    def clean_datetime_fields(self) -> pd.DataFrame:
        """
        Clean datetime fields based on EDA findings.
        
        Identified issues:
        - Dates out of range May 2022 (data 2003-2022)
        - 3,030 trips with negative/zero duration (0.08%)
        - 48,944 trips < 1 minute (1.36%)
        - 5,084 trips > 3 hours (0.14%)
        """
        logger.info("=== DATETIME FIELDS CLEANING ===")
        
        initial_count = len(self.df)
        
        # Convert to datetime if necessary
        datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        for col in datetime_cols:
            if col not in self.df.columns:
                continue
            
            if not self.df[col].dtype == 'object':
                continue
            
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Filter valid dates using PREPROCESSING_PARAMS[data_year] and PREPROCESSING_PARAMS[data_month]
        if 'tpep_pickup_datetime' in self.df.columns:
            valid_date_mask = (
                (self.df['tpep_pickup_datetime'].dt.year == PREPROCESSING_PARAMS['data_year']) &
                (self.df['tpep_pickup_datetime'].dt.month == PREPROCESSING_PARAMS['data_month'])
            )
            
            invalid_dates = (~valid_date_mask).sum()
            logger.info(f"Records with dates outside {PREPROCESSING_PARAMS['data_year']}-{PREPROCESSING_PARAMS['data_month']}: {invalid_dates:,} ({invalid_dates/len(self.df)*100:.2f}%)")
            
            if invalid_dates > 0:
                self.df = self.df[valid_date_mask].copy()
                logger.info(f"Filtered {invalid_dates:,} records with invalid dates")
        
        # Calculate trip duration
        if all(col in self.df.columns for col in datetime_cols):
            self.df['trip_duration_minutes'] = (
                self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']
            ).dt.total_seconds() / 60
            
            # Filter valid durations (between 1 minute and 3 hours)
            duration_mask = (
                (self.df['trip_duration_minutes'] >= 1) &
                (self.df['trip_duration_minutes'] <= 180)
            )
            
            invalid_duration = (~duration_mask).sum()
            logger.info(f"Records with invalid duration: {invalid_duration:,} ({invalid_duration/len(self.df)*100:.2f}%)")
            
            if invalid_duration > 0:
                self.df = self.df[duration_mask].copy()
                logger.info(f"Filtered {invalid_duration:,} records with invalid duration")
            
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Removed records due to date/duration: {removed_records:,}")
        logger.info(f"Remaining records: {final_count:,}")
        
        self.preprocessing_stats['datetime_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_categorical_fields(self) -> pd.DataFrame:
        """
        Clean categorical fields based on TLC dictionary.
        
        Identified issues:
        - VendorID=5  invalid (14 records)
        - 73,587 trips with 0 passengers (2.05%)
        - Missing values in categorical fields (3.6%)
        """
        logger.info("=== CLEANING CATEGORICAL FIELDS ===")
        
        initial_count = len(self.df)
        
        # Clean VendorID - keep only valid values
        if 'VendorID' in self.df.columns:
            valid_vendors = list(self.vendor_mapping.keys())
            invalid_vendor_mask = ~self.df['VendorID'].isin(valid_vendors)
            invalid_vendors = invalid_vendor_mask.sum()
            
            if invalid_vendors > 0:
                logger.info(f"Invalid VendorIDs: {invalid_vendors:,}")
                self.df = self.df[~invalid_vendor_mask].copy()
        
        # Clean passenger_count - remove 0 passengers and extreme values
        if 'passenger_count' in self.df.columns:
            passenger_mask = (
                (self.df['passenger_count'] >= 1) &
                (self.df['passenger_count'] <= 6) &
                (self.df['passenger_count'].notna())
            )
            
            invalid_passengers = (~passenger_mask).sum()
            logger.info(f"Records with invalid passenger_count: {invalid_passengers:,}")
            
            if invalid_passengers > 0:
                self.df = self.df[passenger_mask].copy()
        
        # Validate RatecodeID
        if 'RatecodeID' in self.df.columns:
            valid_ratecodes = list(self.ratecode_mapping.keys())
            invalid_ratecode_mask = (
                ~self.df['RatecodeID'].isin(valid_ratecodes) |
                self.df['RatecodeID'].isna()
            )
            invalid_ratecodes = invalid_ratecode_mask.sum()
            
            if invalid_ratecodes > 0:
                logger.info(f"Invalid RatecodeIDs: {invalid_ratecodes:,}")
                self.df = self.df[~invalid_ratecode_mask].copy()
        
        # Validate payment_type
        if 'payment_type' in self.df.columns:
            valid_payments = list(self.payment_mapping.keys())
            invalid_payment_mask = ~self.df['payment_type'].isin(valid_payments)
            invalid_payments = invalid_payment_mask.sum()
            
            if invalid_payments > 0:
                logger.info(f"Invalid payment_types: {invalid_payments:,}")
                self.df = self.df[~invalid_payment_mask].copy()
        
        # Clean store_and_fwd_flag
        if 'store_and_fwd_flag' in self.df.columns:
            valid_flags = ['Y', 'N']
            invalid_flag_mask = ~self.df['store_and_fwd_flag'].isin(valid_flags)
            invalid_flags = invalid_flag_mask.sum()
            
            if invalid_flags > 0:
                logger.info(f"Invalid store_and_fwd_flags: {invalid_flags:,}")
                self.df = self.df[~invalid_flag_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Records removed due to categorical fields: {removed_records:,}")
        
        self.preprocessing_stats['categorical_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def clean_monetary_fields(self) -> pd.DataFrame:
        """
        Clean monetary fields based on EDA findings.
        
        Identified Issues:
        - 28.01% rows with discrepancies in total_amount
        - 20,506 negative fares (0.57%)
        - 20,709 negative totals (0.58%)
        - 135 anomalous tips in cash payments
        """
        logger.info("=== CLEANING MONETARY FIELDS ===")
        
        initial_count = len(self.df)
        
        # Remove negative values in fields that should not be negative
        positive_fields = ['fare_amount', 'trip_distance', 'total_amount', 'tip_amount', 'tolls_amount']
        
        for field in positive_fields:
            if not field in self.df.columns:
                continue
            negative_mask = self.df[field] < 0
            negative_count = negative_mask.sum()
            
            if negative_count == 0:
                continue
            
            logger.info(f"Negative values in {field}: {negative_count:,}")
            self.df = self.df[~negative_mask].copy()
        
        # Filter trips with fare_amount = 0 (problematic for calculations)
        if 'fare_amount' in self.df.columns:
            zero_fare_mask = self.df['fare_amount'] == 0
            zero_fares = zero_fare_mask.sum()
            
            if zero_fares > 0:
                logger.info(f"Trips with fare_amount = 0: {zero_fares:,}")
                self.df = self.df[~zero_fare_mask].copy()
        
        # Filter trips with distance 0
        if 'trip_distance' in self.df.columns:
            zero_distance_mask = self.df['trip_distance'] == 0
            zero_distance = zero_distance_mask.sum()
            
            if zero_distance > 0:
                logger.info(f"Trips with distance 0: {zero_distance:,}")
                self.df = self.df[~zero_distance_mask].copy()
        
        # Clean anomalous tips in cash payments
        if 'tip_amount' in self.df.columns and 'payment_type' in self.df.columns:
            cash_tip_mask = (self.df['payment_type'] == 2) & (self.df['tip_amount'] > 0)
            cash_tips = cash_tip_mask.sum()
            
            if cash_tips > 0:
                logger.info(f"Anomalous tips in cash payments: {cash_tips:,}")
                self.df.loc[cash_tip_mask, 'tip_amount'] = 0
        
        # Apply reasonable limits to extreme outliers
        if 'fare_amount' in self.df.columns:
            fare_q95 = self.df['fare_amount'].quantile(0.95)
            high_fare_mask = self.df['fare_amount'] > fare_q95 * 3  # 3x the 95th percentile
            high_fares = high_fare_mask.sum()
            
            if high_fares > 0:
                logger.info(f"Extremely high fares removed: {high_fares:,}")
                self.df = self.df[~high_fare_mask].copy()
        
        final_count = len(self.df)
        removed_records = initial_count - final_count
        
        logger.info(f"Records removed due to monetary fields: {removed_records:,}")
        
        self.preprocessing_stats['monetary_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_records': removed_records,
            'removal_percentage': round(removed_records/initial_count*100, 2)
        }
        
        return self.df
    
    def validate_total_amount(self) -> pd.DataFrame:
        """
        Validate and correct discrepancies in total_amount.
        
        EDA found 28.01% records with discrepancies (average difference $2.50).
        """
        logger.info("=== VALIDATION OF total_amount ===")
        
        if 'total_amount' not in self.df.columns:
            logger.warning("total_amount field not found")
            return self.df
        
        # Calculate expected total
        base_fields = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
        additional_fields = ['congestion_surcharge', 'airport_fee']
        
        # Initialize with base fields
        calculated_total = pd.Series(0, index=self.df.index)
        
        for field in base_fields:
            if field not in self.df.columns:
                continue
            
            calculated_total += self.df[field].fillna(0)
        
        # Add additional fields if they exist
        for field in additional_fields:
            if field not in self.df.columns:
                continue
            
            calculated_total += self.df[field].fillna(0)
        
        # Calculate differences
        diff = abs(self.df['total_amount'] - calculated_total)
        discrepancy_mask = diff > 0.01  # Difference greater than 1 cent
        discrepancies = discrepancy_mask.sum()
        
        logger.info(f"Records with discrepancies in total_amount: {discrepancies:,} ({discrepancies/len(self.df)*100:.2f}%)")
        
        if discrepancies > 0:
            logger.info(f"Average difference: ${diff[discrepancy_mask].mean():.2f}")
            logger.info(f"Maximum difference: ${diff.max():.2f}")
            
            # Correct total_amount with calculated value
            self.df.loc[discrepancy_mask, 'total_amount'] = calculated_total[discrepancy_mask]
            logger.info("total_amount corrected based on sum of components")
        
        self.preprocessing_stats['total_amount_validation'] = {
            'discrepancies_found': int(discrepancies),
            'discrepancy_percentage': round(discrepancies/len(self.df)*100, 2),
            'avg_difference': round(diff[discrepancy_mask].mean(), 2) if discrepancies > 0 else 0,
            'max_difference': round(diff.max(), 2)
        }
        
        return self.df
    
    def remove_outliers(self, method='iqr', factor=1.5) -> pd.DataFrame:
        """
        Remove extreme outliers using the IQR method.
        
        Args:
            method: Method to detect outliers ('iqr')
            factor: Multiplicative factor for IQR limits (default 1.5)
        """
        logger.info(f"=== OUTLIER REMOVAL ({method.upper()}) ===")
        
        initial_count = len(self.df)
        
        # Fields for outlier analysis
        outlier_fields = ['trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'trip_duration_minutes']
        outlier_fields = [field for field in outlier_fields if field in self.df.columns]
        
        total_outliers_removed = 0
        
        for field in outlier_fields:
            if method == 'iqr':
                Q1 = self.df[field].quantile(0.25)
                Q3 = self.df[field].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outlier_mask = (self.df[field] < lower_bound) | (self.df[field] > upper_bound)
                outliers_count = outlier_mask.sum()
                
                if outliers_count > 0:
                    logger.info(f"Outliers in {field}: {outliers_count:,} ({outliers_count/len(self.df)*100:.2f}%)")
                    self.df = self.df[~outlier_mask].copy()
                    total_outliers_removed += outliers_count
        
        final_count = len(self.df)
        
        logger.info(f"Total outliers removed: {total_outliers_removed:,}")
        logger.info(f"Remaining records: {final_count:,}")
        
        self.preprocessing_stats['outlier_removal'] = {
            'method': method,
            'factor': factor,
            'initial_count': initial_count,
            'final_count': final_count,
            'outliers_removed': total_outliers_removed,
            'removal_percentage': round(total_outliers_removed/initial_count*100, 2)
        }
        
        return self.df
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get full preprocessing summary.
        
        Returns:
            dict: Complete preprocessing statistics
        """
        if not self.preprocessing_stats:
            logger.warning("No preprocessing statistics available")
            return {}
        
        summary = {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape if self.df is not None else None,
            'total_removed': self.original_shape[0] - (self.df.shape[0] if self.df is not None else 0),
            'removal_percentage': round((self.original_shape[0] - (self.df.shape[0] if self.df is not None else 0)) / self.original_shape[0] * 100, 2),
            'steps': self.preprocessing_stats
        }
        
        return summary
    
    def preprocess_full_pipeline(self, 
                                remove_outliers: bool = True,
                                outlier_method: str = 'iqr',
                                outlier_factor: float = 1.5) -> pd.DataFrame:
        """
        Execute full data cleaning pipeline.
        
        Args:
            remove_outliers: if to remove extreme outliers
            outlier_method: Method to detect outliers
            outlier_factor: Factor for outlier limits
            
        Returns:
            pd.DataFrame: Clean data (without feature engineering)
        """
        logger.info("=== STARTING DATA CLEANING PIPELINE ===")
        
        # 1. Load data
        self.load_data()
        
        # 2. Validate basic structure
        self.validate_basic_structure()
        
        # 3. Clean datetime fields
        self.clean_datetime_fields()
        
        # 4. Clean categorical fields
        self.clean_categorical_fields()
        
        # 5. Clean monetary fields
        self.clean_monetary_fields()
        
        # 6. Validate total_amount
        self.validate_total_amount()
        
        # 7. Remove outliers if requested
        if remove_outliers:
            self.remove_outliers(method=outlier_method, factor=outlier_factor)
        
        # Final summary
        summary = self.get_preprocessing_summary()
        logger.info("=== DATA CLEANING COMPLETED ===")
        logger.info(f"Original records: {summary['original_shape'][0]:,}")
        logger.info(f"Clean records: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        logger.info("Data ready for feature engineering")
        
        return self.df
    
    def save_processed_data(self, output_path: str = None) -> str:
        """
        Save processed data to a parquet file.
        
        Args:
            output_path: Output path. If None, uses PROCESSED_DATA from config.
            
        Returns:
            str: Path of the saved file
        """
        if self.df is None:
            raise ValueError("No processed data to save. Run the pipeline first.")
        else:
            logger.info("Saving processed data...")
        
        output_path = output_path or PROCESSED_DATA
        
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Processed data saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path


def main():
    """Main function to run data cleaning."""
    try:
        # Initialize preprocessor
        preprocessor = TaxiDataPreprocessor()
        
        # Run cleaning pipeline
        cleaned_data = preprocessor.preprocess_full_pipeline(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_factor=1.5
        )
        
        # Save cleaned data
        output_path = preprocessor.save_processed_data()
        
        # Show summary
        summary = preprocessor.get_preprocessing_summary()
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        print(f"Original records: {summary['original_shape'][0]:,}")
        print(f"Clean records: {summary['final_shape'][0]:,}")
        print(f"Records removed: {summary['total_removed']:,} ({summary['removal_percentage']:.2f}%)")
        print(f"Clean data file: {output_path}")
        print("\n📋 NEXT STEP:")
        print("   Run features.py for feature engineering")
        
        return cleaned_data, summary
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise


if __name__ == "__main__":
    main()
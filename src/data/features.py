"""
Módulo de Ingeniería de Características (Feature Engineering) para NYC Taxi Data.

Este módulo centraliza la transformación de datos crudos limpios en vectores de características
optimizados para el modelado predictivo. Implementa estrategias de transformación temporal,
geoespacial y categórica, asegurando consistencia tanto en el entrenamiento (batch) como
en la inferencia en tiempo real.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import os
import sys

# Configuration of the execution environment and logging
try:
    # Dynamic adjustment of the path to allow execution as an independent script
    if 'src' not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    from src.utils.logging import get_full_logger
    from src.config.paths import LOGGER_NAME, PROCESSED_DATA, FEATURE_DATA
    from src.config.settings import LOG_LEVEL, PREPROCESSING_PARAMS
    
    logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)
except ImportError:
    # Fallback mechanism for environments where the project structure is not complete
    warnings.warn("Using fallback logging/config structure. Ensure project paths are set.")
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def warning(self, msg): pass
    logger = DummyLogger()

warnings.filterwarnings('ignore')


class TaxiFeatureEngineer:
    """
    Orchestrator of the feature engineering process.
    
    Responsible for applying deterministic transformations to the taxi dataset.
    Encapsulates the business logic to derive new predictive variables from
    basic transactional data.
    """
    
    def __init__(self, processed_data_path: str = None):
        """
        Initializes the feature processor.

        Args:
            processed_data_path (str, optional): Path to the preprocessed data file.
            By default, it uses the path defined in the configuration.
        """
        self.processed_data_path = processed_data_path or PROCESSED_DATA
        self.df = None
        self.feature_stats = {}
        
        logger.info("TaxiFeatureEngineer initialized")
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        Retrieves the clean dataset from persistent storage (Parquet).
        Mainly used in the batch training flow.
        """
        try:
            logger.info(f"Loading processed data from: {self.processed_data_path}")
            self.df = pd.read_parquet(self.processed_data_path)
            logger.info(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates variables derived from the temporal component of the trip.
        
        Extracts cyclical patterns (time, day of the week) and categorical patterns (weekend, rush hour)
        essential for capturing the seasonality of demand and traffic.
        """
        self.df = df
        logger.info("Creating temporal features...")
        
        if 'tpep_pickup_datetime' in self.df.columns and not self.df['tpep_pickup_datetime'].empty:
            # Standardization of the datetime data type
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'], errors='coerce')
            
            # Extraction of basic temporary components
            self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            
            # Binary indicator for weekends (Saturday=5, Sunday=6)
            self.df['is_weekend'] = (self.df['pickup_day_of_week'].isin([5, 6])).astype(int)
            
            # Segmentation of the day into operating time slots
            bins_tod = [-1, 6, 12, 18, 24]
            labels_tod = ['Night', 'Morning', 'Afternoon', 'Evening']
            self.df['time_of_day'] = pd.cut(self.df['pickup_hour'], bins=bins_tod, labels=labels_tod, right=False)
            
            if not self.df['time_of_day'].empty:
                self.df['time_of_day'] = self.df['time_of_day'].cat.set_categories(labels_tod)
            
            # Identification of rush hours based on NYC traffic patterns
            morning_rush = (self.df['pickup_hour'].between(7, 9))
            evening_rush = (self.df['pickup_hour'].between(17, 19))
            
            self.df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
            self.df['is_morning_rush'] = morning_rush.astype(int)
            self.df['is_evening_rush'] = evening_rush.astype(int)
        
        # Calculation of travel time for training (Target Calculation)
        if 'trip_duration_minutes' not in self.df.columns and all(col in self.df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
            self.df['tpep_dropoff_datetime'] = pd.to_datetime(self.df['tpep_dropoff_datetime'], errors='coerce')
            self.df['trip_duration_minutes'] = (
                (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 60
            )
        
        # Duration categorization for stratified analysis
        if 'trip_duration_minutes' in self.df.columns:
            bins_duration = [0, 5, 15, 30, 60, float('inf')]
            labels_duration = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['duration_category'] = pd.cut(self.df['trip_duration_minutes'], bins=bins_duration, labels=labels_duration, right=False)
            
            if not self.df['duration_category'].empty:
                 self.df['duration_category'] = self.df['duration_category'].cat.set_categories(labels_duration)

            self.df['is_very_short_trip'] = (self.df['trip_duration_minutes'] <= 5).astype(int)
            self.df['is_long_trip'] = (self.df['trip_duration_minutes'] >= 30).astype(int)
            
        logger.info("Temporal features created")
        return self.df
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the dataset with metrics derived from travel distance.
        Apply logarithmic transformations to normalize the distribution of distances
        and create range categories to capture non-linear behaviors.
        """
        self.df = df
        logger.info("Creating distance features...")
        
        if 'trip_distance' in self.df.columns:
            # Distance segmentation
            bins_distance = [0, 1, 3, 7, 15, float('inf')]
            labels_distance = ['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            self.df['distance_category'] = pd.cut(self.df['trip_distance'], bins=bins_distance, labels=labels_distance, right=False)
            
            self.df['is_short_distance'] = (self.df['trip_distance'] <= 1).astype(int)
            self.df['is_long_distance'] = (self.df['trip_distance'] >= 10).astype(int)
            
            # Logarithmic transformation (Log1p) to reduce distribution bias
            self.df['log_trip_distance'] = np.log1p(self.df['trip_distance'])
            
        logger.info("Distance features created")
        return self.df
    
    def create_fare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived economic metrics.
        Note: These features (especially those using fare_amount) should be handled 
        with care to avoid data leakage if used as predictors of the target.
        """
        self.df = df
        logger.info("Creating fare features...")
        
        # Economic efficiency calculation (Rate per mile)
        if all(col in self.df.columns for col in ['fare_amount', 'trip_distance']) and (self.df['trip_distance'] > 0).any():
            self.df['fare_per_mile'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['fare_amount'] / self.df['trip_distance'],
                0
            )
        
        # Analysis of tipping behavior
        if 'tip_amount' in self.df.columns and 'fare_amount' in self.df.columns and (self.df['fare_amount'] > 0).any():
            self.df['tip_percentage'] = np.where(
                self.df['fare_amount'] > 0,
                (self.df['tip_amount'] / self.df['fare_amount']) * 100,
                0
            )

        # Logarithmic transformation of the target (useful for normalization in regression)
        if 'fare_amount' in self.df.columns:
            self.df['log_fare_amount'] = np.log1p(self.df['fare_amount'])
            
        logger.info("Fare features created")
        return self.df
    
    def create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive speed and operational efficiency metrics.
        These metrics are essential for understanding traffic congestion and flow.
        """
        self.df = df
        logger.info("Creating speed features...")
        
        if all(col in self.df.columns for col in ['trip_distance', 'trip_duration_minutes']) and (self.df['trip_duration_minutes'] > 0).any():
            # Calculation of average trip speed (Miles per hour)
            self.df['avg_speed_mph'] = np.where(
                self.df['trip_duration_minutes'] > 0,
                (self.df['trip_distance'] / (self.df['trip_duration_minutes'] / 60)),
                0
            )
            
            # Semantic categorization of speed
            bins_speed = [0, 5, 15, 25, 40, float('inf')]
            labels_speed = ['Very_Slow', 'Slow', 'Medium', 'Fast', 'Very_Fast']
            self.df['speed_category'] = pd.cut(self.df['avg_speed_mph'], bins=bins_speed, labels=labels_speed, right=False)
            
            # Speed range indicators (heavy traffic vs. open highway)
            self.df['is_slow_trip'] = (self.df['avg_speed_mph'] <= 10).astype(int)
            self.df['is_fast_trip'] = (self.df['avg_speed_mph'] >= 30).astype(int)
            
            # Travel efficiency (inverse of congestion)
            if (self.df['trip_distance'] > 0).any():
                 self.df['trip_efficiency'] = np.where(
                    self.df['trip_distance'] > 0,
                    self.df['avg_speed_mph'] / self.df['trip_distance'],
                    0
                )
            
        logger.info("Speed features created")
        return self.df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes and normalizes nominal categorical variables.
        Ensures that location IDs and rate codes are treated as categories
        and handles the imputation of unknown or null values.
        """
        self.df = df
        logger.info("Creating categorical features...")
        
        # Standard mappings of the TLC data dictionary
        ratecode_mapping = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau_Westchester", 5: "Negotiated", 6: "Group_ride", 99: "Unknown"}
        payment_mapping = {0: "Flex_Fare", 1: "Credit_Card", 2: "Cash", 3: "No_Charge", 4: "Dispute", 5: "Unknown", 6: "Voided"}

        RATECODE_CATEGORIES = list(ratecode_mapping.values())
        PAYMENT_CATEGORIES = list(payment_mapping.values())

        # Standardization of location IDs (PULocationID, DOLocationID)
        if 'PULocationID' in self.df.columns:
            self.df['PULocationID'] = self.df['PULocationID'].astype(str)
        if 'DOLocationID' in self.df.columns:
            self.df['DOLocationID'] = self.df['DOLocationID'].astype(str)

        EXPECTED_LOCATION_CATEGORIES = [str(i) for i in range(1, 266)]
        EXPECTED_LOCATION_CATEGORIES.append('0') # Category for values outside range or unknown
        
        if 'PULocationID' in self.df.columns:
             self.df['PULocationID'] = pd.to_numeric(self.df['PULocationID'], errors='coerce').fillna(0).astype(int).astype(str)
             self.df['PULocationID'] = pd.Categorical(
                self.df['PULocationID'], 
                categories=EXPECTED_LOCATION_CATEGORIES
             )
        if 'DOLocationID' in self.df.columns:
             self.df['DOLocationID'] = pd.to_numeric(self.df['DOLocationID'], errors='coerce').fillna(0).astype(int).astype(str)
             self.df['DOLocationID'] = pd.Categorical(
                self.df['DOLocationID'], 
                categories=EXPECTED_LOCATION_CATEGORIES
             )

        # Vendor Standardization (VendorID)
        EXPECTED_VENDORS = ['CMT', 'VTS', 'TPEV', 'Unknown'] 
        if 'vendor_id' in self.df.columns and self.df['vendor_id'].dtype.name in ['object', 'category']:
            self.df['vendor_id'] = self.df['vendor_id'].astype(str).str.upper().replace('NAN', 'UNKNOWN')
            self.df['vendor_id'] = self.df['vendor_id'].fillna('Unknown')
            self.df['vendor_id'] = pd.Categorical(self.df['vendor_id'], categories=EXPECTED_VENDORS)

        # Store and Forward Flag
        EXPECTED_FWD_FLAGS = ['Y', 'N'] 
        if 'store_and_fwd_flag' in self.df.columns and self.df['store_and_fwd_flag'].dtype.name in ['object', 'category']:
            self.df['store_and_fwd_flag'] = self.df['store_and_fwd_flag'].fillna('N')
            self.df['store_and_fwd_flag'] = pd.Categorical(self.df['store_and_fwd_flag'], categories=EXPECTED_FWD_FLAGS)

        # Rate code mapping (RateCodeID)
        if 'RatecodeID' in self.df.columns:
            self.df['RatecodeID'] = self.df['RatecodeID'].astype(str)
            self.df['RatecodeID'] = pd.to_numeric(self.df['RatecodeID'], errors='coerce').fillna(99).astype(int)
            self.df['ratecode_name'] = self.df['RatecodeID'].map(ratecode_mapping)
            self.df['ratecode_name'] = self.df['ratecode_name'].fillna('Unknown')
            self.df['ratecode_name'] = self.df['ratecode_name'].astype('category').cat.set_categories(RATECODE_CATEGORIES)
            
            self.df['is_standard_rate'] = (self.df['RatecodeID'] == 1).astype(int)
            self.df['is_airport_trip'] = self.df['RatecodeID'].isin([2, 3]).astype(int)
        
        # Payment type mapping
        if 'payment_type' in self.df.columns:
            self.df['payment_type'] = pd.to_numeric(self.df['payment_type'], errors='coerce').fillna(99).astype(int)
            self.df['payment_name'] = self.df['payment_type'].map(payment_mapping)
            self.df['payment_name'] = self.df['payment_name'].fillna('Unknown')
            self.df['payment_name'] = self.df['payment_name'].astype('category').cat.set_categories(PAYMENT_CATEGORIES)
            self.df['is_credit_card'] = (self.df['payment_type'] == 1).astype(int)
        
        if 'store_and_fwd_flag' in self.df.columns:
            EXPECTED_FWD_FLAGS = ['Y', 'N']
            self.df['store_and_fwd_flag'] = pd.Categorical(self.df['store_and_fwd_flag'], categories=EXPECTED_FWD_FLAGS)
            self.df['is_store_forward'] = (self.df['store_and_fwd_flag'] == 'Y').astype(int)
            
        logger.info("Categorical features created")
        return self.df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates topological features based on the geography of the journey.
        """
        self.df = df
        logger.info("Creating location features...")
        
        if all(col in self.df.columns for col in ['PULocationID', 'DOLocationID']):
            # Detection of local/circular trips (same starting point and destination)
            self.df['is_round_trip'] = (
                self.df['PULocationID'] == self.df['DOLocationID']
            ).astype(int)
            
        logger.info("Location features created")
        return self.df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction characteristics to capture combined effects.
        Example: A long trip during rush hour behaves differently than a long trip 
        during off-peak hours.
        """
        self.df = df
        logger.info("Creating interaction features...")
        
        # Temporal Interaction: Weekend + Rush Hour
        if all(col in self.df.columns for col in ['is_weekend', 'is_rush_hour']):
            self.df['weekend_rush'] = (
                self.df['is_weekend'] & self.df['is_rush_hour']
            ).astype(int)
        
        # Spatial-Temporal Interaction: Long distance + Long duration
        if all(col in self.df.columns for col in ['is_long_distance', 'is_long_trip']):
            self.df['long_distance_long_time'] = (
                self.df['is_long_distance'] & self.df['is_long_trip']
            ).astype(int)
        
        logger.info("Interaction features created")
        return self.df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder for creating aggregate statistical features.
        
        Warning: This method is designed for batch training contexts,
        where the entire dataset is accessible. In real-time inference, these metrics
        must be pre-calculated or use default values to avoid latency.
        """
        self.df = df
        logger.warning("Skipping real-time statistical feature creation (requires batch context).")
        return self.df
        
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical variables into numerical representations suitable for ML models.
        Apply ordinal encoding for hierarchical variables and one-hot encoding for nominal 
        variables.
        """
        self.df = df
        logger.info("Applying encoding to categorical variables...")

        # Mappings for ordinal encoding (preserving order of magnitude)
        ordinal_mappings = {
            'distance_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'duration_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'speed_category': {'Very_Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very_Fast': 5},
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                temp_col = self.df[col].astype(object)
                self.df[f'{col}_encoded'] = temp_col.map(mapping).fillna(0).astype(int)
                
        # Nominal variables for One-Hot Encoding
        nominal_columns = [
            'time_of_day', 'ratecode_name', 'payment_name', 
            'PULocationID', 'DOLocationID', 'vendor_id', 'store_and_fwd_flag'
        ]
        
        # Application of One-Hot Encoding and removal of original columns
        for col in nominal_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(object).fillna('Unknown')
                dummies = pd.get_dummies(self.df[col], prefix=col, dummy_na=False)
                self.df = pd.concat([self.df, dummies], axis=1).drop(columns=[col])

        logger.info("Categorical variable encoding completed")
        return self.df
    
    def feature_engineering_pipeline(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Execute the complete feature engineering flow.
        
        Execution sequence:
        1. Load data (if not provided).
        2. Generate temporal, spatial, and rate features.
        3. Creation of interactions and statistical transformations.
        4. Final coding of categorical variables.

        Args:
        df (pd.DataFrame, optional): Initial DataFrame. If None, load from disk.

        Returns:
        pd.DataFrame: Enriched dataset ready for training or inference.
        """
        
        logger.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")
        self.df = df
        
        # 1. Loading base data
        if df is None:
            df = self.load_processed_data()
        
        # 2. Generation of base characteristics
        df = self.create_temporal_features(df)
        df = self.create_distance_features(df)
        df = self.create_fare_features(df)
        df = self.create_speed_features(df)
        
        # 3. Treatment of categorical variables and location
        df = self.create_categorical_features(df)
        df = self.create_location_features(df)
        
        # 4. Complex features and statistics
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        
        # 5. Final encoding
        df = self.encode_categorical_variables(df)
        
        # Generation of execution summary
        summary = self.create_feature_summary()
        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        logger.info(f"Total features: {summary['total_columns']}")
        
        self.feature_stats = summary
        return df
    
    def create_feature_summary(self):
        """
        Generates a basic statistical summary of the created features.
        Useful for validation and recording pipeline metadata.
        """
        return {
            'total_columns': len(self.df.columns) if self.df is not None else 0,
            'numeric_columns': 0, # Placeholder for future implementation of detailed counting
            'categorical_columns': 0,
            'binary_indicators': 0,
            'data_shape': self.df.shape if self.df is not None else (0, 0),
        }
    
    def save_feature_data(self, output_path: str = None) -> str:
        """
        Persists the processed dataset in Parquet format.
        
        Args:
            output_path (str, optional): Destination path. By default, uses FEATURE_DATA.
            
        Returns:
            str: Absolute path of the generated file.
        """
        if self.df is None:
            raise ValueError("No feature data to save.")
        
        output_path = output_path or FEATURE_DATA
        
        # Ensure that the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimized writing with gzip compression
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Feature dataset saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path

# --- Entry point for batch execution ---
def main():
    """
    Main function for executing the pipeline in batch mode.
    Typically invoked by data orchestrators (e.g., Airflow) or build scripts.
    """
    try:
        feature_engineer = TaxiFeatureEngineer()
        
        # Execution of the transformation pipeline
        feature_data = feature_engineer.feature_engineering_pipeline() 
        
        # Persistence of results
        output_path = feature_engineer.save_feature_data()
        
        # Final console report
        summary = feature_engineer.feature_stats
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total features: {summary['total_columns']}")
        print(f"Dataset size: {summary['data_shape']}")
        print(f"Saved file: {output_path}")
        
        return feature_data, summary
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise        

if __name__ == "__main__":
    # Direct execution block for independent tests or batch processes.
    main()
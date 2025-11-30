"""
Módulo de ingeniería de características para NYC Yellow Taxi Trip Records.
Transforma datos limpios en características listas para modelado.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

from utils.logging import get_full_logger
from config.paths import LOGGER_NAME, PROCESSED_DATA, FEATURE_DATA
from config.settings import RANDOM_STATE, LOG_LEVEL

warnings.filterwarnings('ignore')
logger = get_full_logger(name=LOGGER_NAME, log_level=LOG_LEVEL)

class TaxiFeatureEngineer:
    """
    Class for creating machine learning features from clean NYC Taxi data.
    
    Responsibilities:
    - Create temporal variables
    - Generate ratios and derived metrics
    - Categorize continuous variables
    - Create dummy/one-hot encoding variables
    - Normalize/scale features
    """
    
    def __init__(self, processed_data_path: str = None):
        """
        Initialize the feature engineer.
        
        Args:
            processed_data_path: Path to the processed/clean data file
        """
        self.processed_data_path = processed_data_path or PROCESSED_DATA
        self.df = None
        self.feature_stats = {}
        
        logger.info("TaxiFeatureEngineer initialized")
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed/clean data from parquet file."""
        try:
            logger.info(f"Loading processed data from: {self.processed_data_path}")
            self.df = pd.read_parquet(self.processed_data_path)
            logger.info(f"Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create advanced temporal features.
        """
        logger.info("Creating temporal features...")
        
        if 'tpep_pickup_datetime' in self.df.columns:
            # Basic temporal features
            self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['pickup_day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            self.df['pickup_day'] = self.df['tpep_pickup_datetime'].dt.day
            self.df['pickup_month'] = self.df['tpep_pickup_datetime'].dt.month
            
            # Day of week features
            self.df['is_weekend'] = (self.df['pickup_day_of_week'].isin([5, 6])).astype(int)
            self.df['is_monday'] = (self.df['pickup_day_of_week'] == 0).astype(int)
            self.df['is_friday'] = (self.df['pickup_day_of_week'] == 4).astype(int)
            
            # Time of day categories
            self.df['time_of_day'] = pd.cut(
                self.df['pickup_hour'],
                bins=[-1, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            
            # Detailed rush hours
            morning_rush = (self.df['pickup_hour'].between(7, 9))
            evening_rush = (self.df['pickup_hour'].between(17, 19))
            
            self.df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
            self.df['is_morning_rush'] = morning_rush.astype(int)
            self.df['is_evening_rush'] = evening_rush.astype(int)
            
            # More specific time categories
            conditions = [
                self.df['pickup_hour'].between(0, 5),   # Late night
                self.df['pickup_hour'].between(6, 11),  # Morning
                self.df['pickup_hour'].between(12, 16), # Afternoon
                self.df['pickup_hour'].between(17, 20), # Evening
                self.df['pickup_hour'].between(21, 23)  # Night
            ]
            choices = ['Late_Night', 'Morning', 'Afternoon', 'Evening', 'Night']
            self.df['detailed_time_category'] = np.select(conditions, choices, default='Unknown')
        
        if not 'trip_duration_minutes' in self.df.columns:
            if all(col in self.df.columns for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']):
                self.df['trip_duration_minutes'] = (
                    (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 60
                )
        
        # Trip duration if exists
        if 'trip_duration_minutes' in self.df.columns:
            # Categorize duration
            self.df['duration_category'] = pd.cut(
                self.df['trip_duration_minutes'],
                bins=[0, 5, 15, 30, 60, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            # Duration indicators
            self.df['is_very_short_trip'] = (self.df['trip_duration_minutes'] <= 5).astype(int)
            self.df['is_long_trip'] = (self.df['trip_duration_minutes'] >= 30).astype(int)
        
        logger.info("Temporal features created")
        return self.df
    
    def create_distance_features(self) -> pd.DataFrame:
        """
        Create distance-related features.
        """
        logger.info("Creating distance features...")
        
        if 'trip_distance' in self.df.columns:
            # Distance categorization
            self.df['distance_category'] = pd.cut(
                self.df['trip_distance'],
                bins=[0, 1, 3, 7, 15, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            # Distance indicators
            self.df['is_short_distance'] = (self.df['trip_distance'] <= 1).astype(int)
            self.df['is_medium_distance'] = (self.df['trip_distance'].between(1, 5)).astype(int)
            self.df['is_long_distance'] = (self.df['trip_distance'] >= 10).astype(int)
            
            # Logarithmic distance to normalize distribution
            self.df['log_trip_distance'] = np.log1p(self.df['trip_distance'])
        
        logger.info("Distance features created")
        return self.df
    
    def create_fare_features(self) -> pd.DataFrame:
        """
        Create fare and payment-related features.
        """
        logger.info("Creating fare features...")
        
        # Ratios and derived metrics
        if all(col in self.df.columns for col in ['tip_amount', 'fare_amount']):
            # Tip percentage
            self.df['tip_percentage'] = np.where(
                self.df['fare_amount'] > 0,
                (self.df['tip_amount'] / self.df['fare_amount']) * 100,
                0
            )
            
            # Categorize tips
            self.df['tip_category'] = pd.cut(
                self.df['tip_percentage'],
                bins=[-0.1, 0, 10, 20, 30, float('inf')],
                labels=['No_Tip', 'Low_Tip', 'Medium_Tip', 'High_Tip', 'Very_High_Tip']
            )
            
            # Tip indicators
            self.df['has_tip'] = (self.df['tip_amount'] > 0).astype(int)
            self.df['generous_tipper'] = (self.df['tip_percentage'] >= 20).astype(int)
        
        if all(col in self.df.columns for col in ['fare_amount', 'trip_distance']):
            # Fare per mile
            self.df['fare_per_mile'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['fare_amount'] / self.df['trip_distance'],
                0
            )
            
            # Categorize fare per mile
            self.df['fare_per_mile_category'] = pd.cut(
                self.df['fare_per_mile'],
                bins=[0, 3, 6, 10, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        
        if 'total_amount' in self.df.columns:
            # Categorize total amount
            self.df['total_amount_category'] = pd.cut(
                self.df['total_amount'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High', 'Extreme']
            )
            
            # Logarithmic transformation to normalize
            self.df['log_total_amount'] = np.log1p(self.df['total_amount'])
        
        if 'fare_amount' in self.df.columns:
            self.df['log_fare_amount'] = np.log1p(self.df['fare_amount'])
        
        logger.info("Fare features created")
        return self.df
    
    def create_speed_features(self) -> pd.DataFrame:
        """
        Create speed and efficiency features.
        """
        logger.info("Creating speed features...")
        
        if all(col in self.df.columns for col in ['trip_distance', 'trip_duration_minutes']):
            # Average speed
            self.df['avg_speed_mph'] = np.where(
                self.df['trip_duration_minutes'] > 0,
                (self.df['trip_distance'] / (self.df['trip_duration_minutes'] / 60)),
                0
            )
            
            # Categorize speed
            self.df['speed_category'] = pd.cut(
                self.df['avg_speed_mph'],
                bins=[0, 5, 15, 25, 40, float('inf')],
                labels=['Very_Slow', 'Slow', 'Medium', 'Fast', 'Very_Fast']
            )
            
            # Speed indicators
            self.df['is_slow_trip'] = (self.df['avg_speed_mph'] <= 10).astype(int)
            self.df['is_fast_trip'] = (self.df['avg_speed_mph'] >= 30).astype(int)
            
            # Trip efficiency (speed vs distance)
            self.df['trip_efficiency'] = np.where(
                self.df['trip_distance'] > 0,
                self.df['avg_speed_mph'] / self.df['trip_distance'],
                0
            )
        
        logger.info("Speed features created")
        return self.df
    
    def create_categorical_features(self) -> pd.DataFrame:
        """
        Create and enhance categorical features.
        """
        logger.info("Creating categorical features...")
        
        # Mappings for interpretability
        vendor_mapping = {
            1: "Creative_Mobile", 2: "Curb_Mobility", 
            6: "Myle_Technologies", 7: "Helix"
        }
        
        ratecode_mapping = {
            1: "Standard", 2: "JFK", 3: "Newark",
            4: "Nassau_Westchester", 5: "Negotiated", 
            6: "Group_ride", 99: "Unknown"
        }
        
        payment_mapping = {
            0: "Flex_Fare", 1: "Credit_Card", 2: "Cash", 
            3: "No_Charge", 4: "Dispute", 5: "Unknown", 6: "Voided"
        }
        
        # Apply mappings
        if 'VendorID' in self.df.columns:
            self.df['vendor_name'] = self.df['VendorID'].map(vendor_mapping)
            
        if 'RatecodeID' in self.df.columns:
            self.df['ratecode_name'] = self.df['RatecodeID'].map(ratecode_mapping)
            
            # Specific indicators
            self.df['is_airport_trip'] = self.df['RatecodeID'].isin([2, 3]).astype(int)
            self.df['is_jfk_trip'] = (self.df['RatecodeID'] == 2).astype(int)
            self.df['is_newark_trip'] = (self.df['RatecodeID'] == 3).astype(int)
            self.df['is_standard_rate'] = (self.df['RatecodeID'] == 1).astype(int)
        
        if 'payment_type' in self.df.columns:
            self.df['payment_name'] = self.df['payment_type'].map(payment_mapping)
            
            # Payment indicators
            self.df['is_credit_card'] = (self.df['payment_type'] == 1).astype(int)
            self.df['is_cash_payment'] = (self.df['payment_type'] == 2).astype(int)
            self.df['is_no_charge'] = (self.df['payment_type'] == 3).astype(int)
        
        if 'passenger_count' in self.df.columns:
            # Categorize passenger count
            self.df['passenger_category'] = pd.cut(
                self.df['passenger_count'],
                bins=[0, 1, 2, 4, float('inf')],
                labels=['Single', 'Couple', 'Small_Group', 'Large_Group']
            )
            
            self.df['is_single_passenger'] = (self.df['passenger_count'] == 1).astype(int)
            self.df['is_group_trip'] = (self.df['passenger_count'] >= 3).astype(int)
        
        if 'store_and_fwd_flag' in self.df.columns:
            self.df['is_store_forward'] = (self.df['store_and_fwd_flag'] == 'Y').astype(int)
        
        logger.info("Categorical features created")
        return self.df
    
    def create_location_features(self) -> pd.DataFrame:
        """
        Create location-related features.
        """
        logger.info("Creating location features...")
        
        if all(col in self.df.columns for col in ['PULocationID', 'DOLocationID']):
            # Circular trip indicator (same pickup and dropoff)
            self.df['is_round_trip'] = (
                self.df['PULocationID'] == self.df['DOLocationID']
            ).astype(int)
            
            # Create unique route identifier
            self.df['route_id'] = (
                self.df['PULocationID'].astype(str) + '_to_' + 
                self.df['DOLocationID'].astype(str)
            )
            
            # Popularity of locations (frequency)
            pickup_counts = self.df['PULocationID'].value_counts()
            dropoff_counts = self.df['DOLocationID'].value_counts()
            
            self.df['pickup_popularity'] = self.df['PULocationID'].map(pickup_counts)
            self.df['dropoff_popularity'] = self.df['DOLocationID'].map(dropoff_counts)
            
            # Categorize popularity
            self.df['pickup_popularity_category'] = pd.cut(
                self.df['pickup_popularity'],
                bins=[0, 100, 500, 2000, float('inf')],
                labels=['Rare', 'Uncommon', 'Common', 'Very_Common']
            )
        
        logger.info("Location features created")
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between variables.
        """
        logger.info("Creating interaction features...")
        
        # Temporal interactions
        if all(col in self.df.columns for col in ['is_weekend', 'is_rush_hour']):
            self.df['weekend_rush'] = (
                self.df['is_weekend'] & self.df['is_rush_hour']
            ).astype(int)
        
        # Payment and tip interactions
        if all(col in self.df.columns for col in ['is_credit_card', 'has_tip']):
            self.df['credit_card_with_tip'] = (
                self.df['is_credit_card'] & self.df['has_tip']
            ).astype(int)
        
        # Distance and time interactions
        if all(col in self.df.columns for col in ['is_long_distance', 'is_long_trip']):
            self.df['long_distance_long_time'] = (
                self.df['is_long_distance'] & self.df['is_long_trip']
            ).astype(int)
        
        # Airport and time interactions
        if all(col in self.df.columns for col in ['is_airport_trip', 'time_of_day']):
            self.df['airport_morning'] = (
                self.df['is_airport_trip'] & (self.df['time_of_day'] == 'Morning')
            ).astype(int)
            
            self.df['airport_evening'] = (
                self.df['is_airport_trip'] & (self.df['time_of_day'] == 'Evening')
            ).astype(int)
        
        logger.info("Interaction features created")
        return self.df
    
    def create_statistical_features(self) -> pd.DataFrame:
        """
        Create aggregated statistical features.
        """
        logger.info("Creating statistical features...")
        
        # Aggregations by hour of day
        if 'pickup_hour' in self.df.columns:
            hourly_stats = self.df.groupby('pickup_hour').agg({
                'fare_amount': ['mean', 'std'],
                'trip_distance': ['mean', 'std'],
                'tip_percentage': 'mean'
            }).round(2)
            
            # Flatten column names
            hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
            
            # Merge back to main dataframe
            for col in hourly_stats.columns:
                self.df[f'hourly_{col}'] = self.df['pickup_hour'].map(
                    hourly_stats[col]
                )
        
        # Aggregations by day of week
        if 'pickup_day_of_week' in self.df.columns:
            daily_stats = self.df.groupby('pickup_day_of_week').agg({
                'fare_amount': 'mean',
                'trip_distance': 'mean',
                'avg_speed_mph': 'mean'
            }).round(2)
            
            for col in daily_stats.columns:
                self.df[f'daily_avg_{col}'] = self.df['pickup_day_of_week'].map(
                    daily_stats[col]
                )
        
        logger.info("Statistical features created")
        return self.df
    
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Apply encoding to categorical variables for ML.
        """
        logger.info("Applying encoding to categorical variables...")
        
        # Ordinal categorical variables (keep as numeric)
        ordinal_mappings = {
            'distance_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'duration_category': {'Very_Short': 1, 'Short': 2, 'Medium': 3, 'Long': 4, 'Very_Long': 5},
            'speed_category': {'Very_Slow': 1, 'Slow': 2, 'Medium': 3, 'Fast': 4, 'Very_Fast': 5},
            'tip_category': {'No_Tip': 0, 'Low_Tip': 1, 'Medium_Tip': 2, 'High_Tip': 3, 'Very_High_Tip': 4},
            'total_amount_category': {'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4, 'Extreme': 5}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = self.df[col].map(mapping)
        
        # One-hot encoding for nominal categorical variables
        nominal_columns = [
            'time_of_day', 'detailed_time_category', 'vendor_name', 
            'ratecode_name', 'payment_name', 'passenger_category'
        ]
        
        for col in nominal_columns:
            if col not in self.df.columns:
                continue
            # Create dummies
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)
        
        logger.info("Categorical variable encoding completed")
        return self.df
    
    def create_feature_summary(self) -> dict:
        """
        Create summary of generated features.
        """
        if self.df is None:
            return {}
        
        # Categorize column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Binary/indicator columns (only 0s and 1s)
        binary_cols = []
        for col in numeric_cols:
            if set(self.df[col].dropna().unique()).issubset({0, 1}):
                binary_cols.append(col)
        
        summary = {
            'total_columns': len(self.df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'binary_indicators': len(binary_cols),
            'column_types': {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols,
                'binary_indicators': binary_cols
            },
            'data_shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return summary
    
    def feature_engineering_pipeline(self) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        
        Returns:
            pd.DataFrame: Dataset with features for ML
        """
        logger.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")
        
        # 1. Load processed data
        self.load_processed_data()
        
        # 2. Create temporal features
        self.create_temporal_features()
        
        # 3. Create distance features
        self.create_distance_features()
        
        # 4. Create fare features
        self.create_fare_features()
        
        # 5. Create speed features
        self.create_speed_features()
        
        # 6. Create categorical features
        self.create_categorical_features()
        
        # 7. Create location features
        self.create_location_features()
        
        # 8. Create interaction features
        self.create_interaction_features()
        
        # 9. Create statistical features
        self.create_statistical_features()
        
        # 10. Encode categorical variables
        self.encode_categorical_variables()
        
        # Final summary
        summary = self.create_feature_summary()
        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        logger.info(f"Total features: {summary['total_columns']}")
        logger.info(f"Numeric variables: {summary['numeric_columns']}")
        logger.info(f"Categorical variables: {summary['categorical_columns']}")
        logger.info(f"Binary indicators: {summary['binary_indicators']}")
        
        self.feature_stats = summary
        return self.df
    
    def save_feature_data(self, output_path: str = None) -> str:
        """
        Save dataset with features to a parquet file.
        
        Args:
            output_path: Output path. If None, uses FEATURE_DATA from config.
            
        Returns:
            str: Path of the saved file
        """
        if self.df is None:
            raise ValueError("No feature data to save.")
        
        output_path = output_path or FEATURE_DATA
        
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        self.df.to_parquet(output_path, index=False, engine='pyarrow', compression='gzip')
        logger.info(f"Feature dataset saved at: {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
        
        return output_path


def main():
    """Main function to run feature engineering."""
    try:
        # Initialize feature engineer
        feature_engineer = TaxiFeatureEngineer()
        
        # Run full pipeline
        feature_data = feature_engineer.feature_engineering_pipeline()
        
        # Save feature data
        output_path = feature_engineer.save_feature_data()
        
        # Show summary
        summary = feature_engineer.feature_stats
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total features: {summary['total_columns']}")
        print(f"Numeric variables: {summary['numeric_columns']}")
        print(f"Categorical variables: {summary['categorical_columns']}")
        print(f"Binary indicators: {summary['binary_indicators']}")
        print(f"Dataset size: {summary['data_shape']}")
        print(f"Saved file: {output_path}")
        
        return feature_data, summary
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()

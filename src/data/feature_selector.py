import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.selected_features = None
        
    def select_features_for_fare(self, X: pd.DataFrame, y: pd.Series, k=30):
        """
            Feature selection for prediction of fare_amount
            
            Parameters:
            X: pd.DataFrame - DataFrame containing features
            y: pd.Series - Target variable (fare_amount)
            k: int - Number of top features to select
        """
        
        # Features most relevant for fare prediction based on your schema
        fare_relevant_features = [
            # Core trip features
            'trip_distance', 'log_trip_distance', 'fare_per_mile',
            'passenger_count', 'trip_duration_minutes',
            
            # Temporal features
            'pickup_hour', 'pickup_day_of_week', 'is_weekend', 
            'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening',
            
            # Location/Route features
            'is_airport_trip', 'is_jfk_trip', 'is_newark_trip',
            'pickup_popularity', 'dropoff_popularity',
            
            # Rate and payment features
            'is_standard_rate', 'ratecode_name_Standard', 'ratecode_name_Newark',
            'is_credit_card', 'is_cash_payment', 'payment_name_Credit_Card',
            
            # Trip characteristics
            'distance_category_encoded', 'is_short_distance', 
            'is_medium_distance', 'is_long_distance',
            'avg_speed_mph', 'speed_category_encoded',
            
            # Additional features
            'extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge',
            'airport_fee', 'is_store_forward',
            
            # Aggregated features
            'hourly_fare_amount_mean', 'hourly_fare_amount_std',
            'daily_avg_fare_amount'
        ]
        
        # Filter features that exist in the dataset
        available_features = [f for f in fare_relevant_features if f in X.columns]
        
        # Statistical feature selection
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        # Feature importance with RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        # Combine both methods
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        # Normalize scores and create combined score
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        # Select top K features
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores
    
    def select_features_for_duration(self, X: pd.DataFrame, y: pd.Series, k=25):
        """Feature selection specific for predicting trip_duration_minutes
        
            Parameters:
            X: pd.DataFrame - DataFrame containing features
            y: pd.Series - Target variable (trip_duration_minutes)
            k: int - Number of top features to select
        """
        
        # Features most relevant for duration prediction
        duration_relevant_features = [
            # Core trip features
            'trip_distance', 'log_trip_distance', 'avg_speed_mph',
            'passenger_count', 'fare_amount',
            
            # Temporal features - clave para duration
            'pickup_hour', 'pickup_day_of_week', 'is_weekend',
            'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening',
            'detailed_time_category_Morning', 'detailed_time_category_Evening',
            'detailed_time_category_Late_Night', 'detailed_time_category_Night',
            
            # Speed and efficiency features
            'speed_category_encoded', 'is_slow_trip', 'is_fast_trip',
            'trip_efficiency',
            
            # Location features
            'is_airport_trip', 'is_jfk_trip', 'is_newark_trip',
            'pickup_popularity', 'dropoff_popularity',
            
            # Distance categories
            'distance_category_encoded', 'is_short_distance',
            'is_medium_distance', 'is_long_distance',
            'is_very_short_trip', 'is_long_trip',
            
            # Vendor and operational
            'VendorID', 'is_store_forward',
            'ratecode_name_Standard', 'ratecode_name_Newark',
            
            # Interaction features
            'weekend_rush', 'long_distance_long_time',
            'airport_morning', 'airport_evening',
            
            # Aggregated features
            'hourly_trip_distance_mean', 'daily_avg_avg_speed_mph',
            'is_round_trip'
        ]
        
        # Filter features that exist in the dataset
        available_features = [f for f in duration_relevant_features if f in X.columns]
        
        # Similar process as for fare
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores

    def get_selected_features(self):
        """Returns the selected features"""
        return self.selected_features
    
    def transform(self, X: pd.DataFrame):
        """
            Applies feature selection to a DataFrame
            Parameters:
            X: pd.DataFrame - DataFrame to transform
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Run select_features first.")
        
        return X[self.selected_features]
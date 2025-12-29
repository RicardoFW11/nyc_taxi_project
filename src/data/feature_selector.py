import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    def __init__(self, target_name: str):
        """
        Class responsible for identifying and selecting the optimal subset of predictor variables.
        It implements a hybrid strategy that combines univariate statistical filtering methods
        with feature importance derived from tree-based models (Random Forest).

        Parameters:
        -----------
        target_name : str
            Name of the target variable for which the selection will be optimized.
        """
        self.target_name = target_name
        self.selected_features = None
        
    def select_features_for_fare(self, X: pd.DataFrame, y: pd.Series, k=30):
        """
        Executes the feature selection process optimized for fare prediction (Fare Amount).
        
        Applies a strict filter on candidate variables to eliminate any feature
        that constitutes ‘Data Leakage’ (information available only post-trip). Subsequently,
        ranks and selects the ‘k’ best variables based on a combined score.

        Parameters:
        -----------
        X : pd.DataFrame
            Array of candidate features.
        y : pd.Series
            Vector of the target variable (fare amount).
        k : int
            Maximum number of features to keep in the final model.
            
        Returns:
        --------
        tuple
            - List of names of selected features.
            - DataFrame with details of the importance scores for each evaluated variable.
        """
        
        # Explicit definition of candidate variables.
        # Variables containing future information (leakage)
        # or direct components of the final rate (tolls, surcharges, taxes) have been audited and excluded.
        fare_relevant_features = [
            # --- VALID CHARACTERISTICS (Ex-ante information) ---

            # Basic trip metrics
            'trip_distance',        # Estimated distance of the optimal route.
            'log_trip_distance',    # Logarithmic transformation to handle asymmetry.
            'passenger_count',      # Declared by the user at the beginning.

            # Temporal Characteristics (Derived from the Start Timestamp)
            'pickup_hour', 
            'pickup_day_of_week', 
            'is_weekend', 
            'is_rush_hour', 
            'is_morning_rush', 
            'is_evening_rush',
            'time_of_day_Morning', 
            'time_of_day_Afternoon', 
            'time_of_day_Evening',

            # Geospatial Features (Origin and Destination)
            'is_airport_trip', 
            'is_jfk_trip', 
            'is_newark_trip',
            'pickup_popularity',    # Historical frequency in the area.
            'dropoff_popularity',

            # Fare and Payment Information (Known when configuring the trip)
            'is_standard_rate', 
            'ratecode_name_Standard', 
            'ratecode_name_Newark',
            'is_credit_card', 
            'is_cash_payment', 
            'payment_name_Credit_Card',

            # Travel Categorization (Pre-trip Estimates)
            'distance_category_encoded', 
            'is_short_distance', 
            'is_medium_distance', 
            'is_long_distance',
            
            # Historical Aggregations (Statistical context, not from the current trip)
            'hourly_fare_amount_mean', 
            'hourly_fare_amount_std',
            'daily_avg_fare_amount'

            # --- VARIABLES EXCLUDED DUE TO DATA LEAKAGE ---
            # ‘fare_per_mile’: Contains the implicit target variable.
            # ‘trip_duration_minutes’: Information only known at the end.
            # ‘avg_speed_mph’: Requires knowledge of the actual duration.
            # ‘extra’, ‘mta_tax’, ‘improvement_surcharge’, etc.: Post-trip billing components.
        ]

        # Intersection between theoretical characteristics and those available in the dataset
        available_features = [f for f in fare_relevant_features if f in X.columns]
        
        # Method 1: Univariate Statistical Selection (F-test for regression)
        # Captures direct linear relationships between each feature and the target.
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        # Method 2: Model-based Importance (Random Forest)
        # Captures nonlinear relationships and complex interactions between variables.
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        # Consolidation of Results
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        # Min-Max normalization and consensus score calculation
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        # Final selection based on combined score
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores
    
    def select_features_for_duration(self, X: pd.DataFrame, y: pd.Series, k=25):
        """
        Executes the feature selection process optimized for predicting trip duration.
        
        The logic differs from the fare model, prioritizing variables that affect traffic flow
        and operating speed, while maintaining strict exclusions to prevent information leakage.

        Parameters:
        -----------
        X : pd.DataFrame
            Array of candidate features.
        y : pd.Series
            Vector of the target variable (duration in minutes).
        k : int
            Maximum number of features to retain.
        """
        
        # Definition of audited candidate variables for the time model.
        duration_relevant_features = [
            # Basic metrics (Physical distance and passenger volume)
            'trip_distance', 
            'log_trip_distance',
            'passenger_count',
            
            # Temporary Variables (Critical for modeling congestion)
            'pickup_hour', 'pickup_day_of_week', 'is_weekend',
            'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'time_of_day_Morning', 'time_of_day_Afternoon', 'time_of_day_Evening',
            'detailed_time_category_Morning', 'detailed_time_category_Evening',
            'detailed_time_category_Late_Night', 'detailed_time_category_Night',
            
            # Geospatial Variables (High-density areas or airports)
            'is_airport_trip', 'is_jfk_trip', 'is_newark_trip',
            'pickup_popularity', 'dropoff_popularity',
            
            # Distance Categorization
            'distance_category_encoded', 'is_short_distance',
            'is_medium_distance', 'is_long_distance',
            'is_very_short_trip', 'is_long_trip',
            
            # Operative Information
            'VendorID', 'is_store_forward',
            'ratecode_name_Standard', 'ratecode_name_Newark',
            
            # Historical Interactions and Aggregations
            'weekend_rush', 
            'airport_morning', 'airport_evening',
            'hourly_trip_distance_mean', 
            'daily_avg_avg_speed_mph', # Valid only if it represents the historical average for the day, not for the current trip.
            'is_round_trip'

            # --- VARIABLES EXCLUDED DUE TO DATA LEAKAGE ---
            # ‘fare_amount’: Highly correlated but unknown at the exact start (if there is a meter).
            # ‘avg_speed_mph’: Directly derived from distance/real time.
            # ‘speed_category_encoded’, ‘is_slow_trip’: Derived from actual speed.
            # ‘long_distance_long_time’: Interaction that reveals duration.
        ]

        # Filtering features available in the current dataset
        available_features = [f for f in duration_relevant_features if f in X.columns]
        
        # Application of the hybrid methodology (Statistics + Random Forest)
        selector_stats = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
        _ = selector_stats.fit_transform(X[available_features], y)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[available_features], y)
        
        # Score Consolidation and Normalization
        feature_scores = pd.DataFrame({
            'feature': available_features,
            'rf_importance': rf.feature_importances_,
            'statistical_score': selector_stats.scores_
        })
        
        feature_scores['rf_norm'] = feature_scores['rf_importance'] / feature_scores['rf_importance'].max()
        feature_scores['stat_norm'] = feature_scores['statistical_score'] / feature_scores['statistical_score'].max()
        feature_scores['combined_score'] = (feature_scores['rf_norm'] + feature_scores['stat_norm']) / 2
        
        # Selection of Top-K
        top_features = feature_scores.nlargest(k, 'combined_score')['feature'].tolist()
        
        self.selected_features = top_features
        return top_features, feature_scores

    def get_selected_features(self):
        """Devuelve la lista de características seleccionadas en la última ejecución."""
        return self.selected_features
    
    def transform(self, X: pd.DataFrame):
        """
        Reduces the input DataFrame, retaining only the previously selected columns.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Original DataFrame with all features.
            
        Returns:
        --------
        pd.DataFrame
        Filtered DataFrame containing only the relevant features.

        Raises:
        -------
        ValueError
        If no previous selection process has been executed.
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Run select_features first.")
        
        return X[self.selected_features]
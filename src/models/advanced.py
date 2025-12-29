"""
Advanced Regression Model Implementation Module.

This module defines the concrete classes for the XGBoost and Random Forest algorithms,
inheriting from the abstract base class. It encapsulates the configuration, training,
prediction, and evaluation logic specific to each model architecture.
"""

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics

class XGBoostModel(BaseModel):
    def __init__(self, output_path:str, target: str = 'fare_amount',
                 n_estimators: int = 100,
                 max_depth: int = 8,
                 learning_rate: float = 0.05,
                 subsample: float = 0.85,
                 colsample_bytree: float = 0.85,
                 colsample_bylevel: float = 0.8,
                 min_child_weight: int = 3,
                 gamma: float = 0.1,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 tree_method: str = 'hist',
                 grow_policy: str = 'depthwise',
                 max_leaves: int = 0,
                 **kwargs):
        """
        Initializes the Gradient Boosting model (XGBoost).
        
        Configures the hyperparameters of the XGBoost regressor to optimize performance
        in tabular regression tasks. The default values are set to
        balance generalization ability and training time.

        Parameters:
        -----------
        output_path : str
            Base path for model artifact persistence.
        target : str
            Name of the target variable to be predicted.
        n_estimators : int
            Number of decision trees (boosting rounds).
        max_depth : int
            Maximum depth of each tree, controlling the complexity of the model.
        learning_rate : float
            Learning rate (eta) for weight updating.
        subsample: float
        Fraction of samples used to train each tree (bagging).
        colsample_bytree: float
        Fraction of features used per tree.
        tree_method: str
        Tree construction algorithm (‘hist’ optimized for speed).
        kwargs: dict
        Additional parameters passed directly to the XGBRegressor constructor.
        """
        super().__init__('xgboost', target, output_path, model_type='advanced')
        
        # Detailed hyperparameter configuration
        xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'tree_method': tree_method,
            'objective': 'reg:squarederror', # Loss function for regression
            'eval_metric': 'rmse',           # Evaluation metric during training
            'n_jobs': -1,                    # Parallelization using all available cores
            'random_state': RANDOM_STATE,    # Reproducibility
            'verbosity': 0,
        }
        
        # Specific settings for histogram-based methods
        if tree_method in ['hist', 'gpu_hist']:
            xgb_params['grow_policy'] = grow_policy
            if max_leaves > 0:
                xgb_params['max_leaves'] = max_leaves
        
        xgb_params.update(kwargs)
        
        self.model = XGBRegressor(**xgb_params)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Specific preprocessing for XGBoost before training/prediction.
        Currently passes data directly, but allows future logic to be injected
        (e.g., sparse matrix handling).
        """
        features = data.copy()
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the XGBoost model with the provided data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Target variable vector.
        """
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        Apply a rectification (ReLU) to ensure that no negative values occur,
        as rates and durations must be strictly non-negative.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        
        # Output correction: max(0, prediction)
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model's performance on an independent dataset.
        
        Returns:
        --------
        dict
            Dictionary with standard metrics (R2, MAE, RMSE) and model-specific metadata
            (number of estimators used, best iteration).
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        # Enrichment with internal XGBoost metrics
        self.metrics.update({
            'n_estimators_used': self.model.get_booster().num_boosted_rounds(),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extracts and organizes feature importance according to the model.
        Calculates multiple types of importance (gain, weight, cover) for robust analysis.

        Returns:
        --------
        pd.DataFrame
        DataFrame sorted by information gain (gain_importance).
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        feature_names = self.model.get_booster().feature_names
        
        # Extracting native importance metrics from XGBoost
        gain_importance = self.model.get_booster().get_score(importance_type='gain')
        weight_importance = self.model.get_booster().get_score(importance_type='weight')
        cover_importance = self.model.get_booster().get_score(importance_type='cover')
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'gain_importance': [gain_importance.get(f, 0) for f in feature_names],
            'weight_importance': [weight_importance.get(f, 0) for f in feature_names],
            'cover_importance': [cover_importance.get(f, 0) for f in feature_names]
        }).sort_values('gain_importance', ascending=False)
        
        return importance_df

class RandomForestModel(BaseModel):
    def __init__(self, output_path:str, target: str = 'fare_amount',
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 criterion: str = 'squared_error',
                 min_impurity_decrease: float = 0.001,
                 min_weight_fraction_leaf: float = 0.01,
                 max_leaf_nodes: int = None,
                 **kwargs):
        
        """
        Initializes the Random Forest Regressor model.
        
        Configures a robust bagging ensemble, ideal for establishing performance baselines
        and analyzing feature importance without the risk of excessive overfitting.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int
            Maximum depth of the trees.
        min_samples_split : int
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int
            Minimum number of samples required in a leaf node.
        max_features : str
            Number of features to consider when searching for the best split (‘sqrt’ is standard).
        bootstrap : bool
            Whether bootstrap samples are used to construct trees.
        oob_score : bool
            Whether out-of-bag samples are used to estimate generalization accuracy.
        """
        
        super().__init__('random_forest', target, output_path, model_type='advanced')
        
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'criterion': criterion,
            'min_impurity_decrease': min_impurity_decrease,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'warm_start': False,
            'verbose': 0
        }
        
        if max_leaf_nodes is not None:
            rf_params['max_leaf_nodes'] = max_leaf_nodes
            
        rf_params.update(kwargs)
        
        self.model = RandomForestRegressor(**rf_params)
        self.oob_score_ = None
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Specific preprocessing for Random Forest."""
        features = data.copy()
        return features
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Random Forest model.
        Capture the Out-of-Bag (OOB) score if enabled, providing an estimate of internal 
        cross-validation error without requiring a separate set.
        """
        # Defensive verification of OOB configuration
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
            self.model.oob_score = True
        
        X_prepared = self._prepare_features(X_train)
        self.model.fit(X_prepared, y_train)
        
        # Storage of the post-training OOB score
        if 'oob_score' in self.model.get_params() and self.model.oob_score:
             self.oob_score_ = self.model.oob_score_

        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediction with Random Forest.
        Applies negative value rectification (ReLU) to the output.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)
        
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model and returns performance metrics.
        Includes the OOB score as an additional metric for overfitting diagnosis.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        self.metrics = calculate_metrics(y_test.values, y_pred)
        
        self.metrics.update({
            'oob_score': self.oob_score_,
            'n_trees': self.model.n_estimators
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate the importance of characteristics based on the mean decrease in impurity (MDI).
        
        In addition to the mean, calculate the standard deviation of importance among all trees
        in the forest, allowing you to evaluate the stability of the importance of each variable.
        """
        if not self.is_trained:
            raise ValueError("The model must be trained first")
        
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            # Generation of generic names if they are not available in the model object
            n_features = len(self.model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Overall average importance
        importance_scores = self.model.feature_importances_
        
        # Analysis of significant variability among individual trees
        tree_importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        importance_std = np.std(tree_importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_scores,
            'importance_std': importance_std,
            'importance_cv': importance_std / (importance_scores + 1e-10) # Coeficiente de variación
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
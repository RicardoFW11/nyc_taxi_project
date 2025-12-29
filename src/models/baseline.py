"""
Baseline Models Module for Regression.

This module implements classic supervised learning algorithms that serve as a
benchmark for evaluating the performance of more complex models.
It includes Linear Regression (to capture simple linear relationships) and Decision Trees
(to capture basic non-linearities without the computational cost of complex ensembles).
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics
import pandas as pd
import numpy as np

class LinearRegressionModel(BaseModel):
    def __init__(self, output_path:str, 
                 target: str = 'fare_amount',
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 positive: bool = False,
                 **kwargs
                 ):
        """
        Initialize the Linear Regression Model (Ordinary Least Squares).
        
        This model is mainly used as a baseline to establish the lower limit
        of expected performance. Its simplicity and high interpretability make it ideal for
        detecting whether the complexity added by other models justifies their computational cost.
        
        Parameters:
        -----------
        output_path : str
            Directory where the serialized model will be persisted.
        target : str
            Dependent variable to be predicted.
        fit_intercept : bool
            Whether to calculate the bias (intercept) of the model.
        positive : bool
            Whether to force the coefficients to be positive (useful in pricing contexts).
        kwargs : dict
            Additional arguments for the sklearn LinearRegression class.
        """
        
        super().__init__('linear_regression', target, output_path, model_type='baseline')
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            n_jobs=-1 #  Uses all available CPU cores for matrix operations
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the coefficients of the linear model to the training data.
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates linear predictions based on the learned coefficients.
        Applies a ReLU activation function (max(0, x)) to the output to ensure
        that no physically or economically impossible negative values are generated.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Calculates standard performance metrics by comparing predictions against actual values.
        Stores the results internally for later serialization.
        """
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Returns the current hyperparameter settings for the estimator.."""
        return self.model.get_params(deep=deep)
        
class DecisionTreeModel(BaseModel):
    def __init__(self, output_path:str, target: str = 'fare_amount',
                 criterion: str = 'squared_error',
                 splitter: str = 'best',
                 max_depth: int = 10,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 max_leaf_nodes: int = 5,
                 min_impurity_decrease: float = 0.001,
                 max_features: str = 'sqrt',
                 **kwargs):
        """
        Initializes the Decision Tree Model (CART).
        
        Configured as a nonlinear base model capable of capturing simple interactions
        between variables. Default hyperparameters are restricted (e.g., max_depth=10)
        to prevent overfitting, a common problem in individual trees.
        
        Parameters:
        -----------
        output_path : str
            Model storage path.
        target : str
            Target variable.
        criterion : str
            Loss function to measure the quality of a split (‘squared_error’ for MSE).
        max_depth : int
            Maximum depth of the tree. Limits the complexity of the model.
        min_samples_split : int
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int
            Minimum number of samples required in a leaf node (smoothing).
        """
        # Initialization of the parent class
        super().__init__('decision_tree', target, output_path, model_type='baseline')
        
        self.model = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=RANDOM_STATE, # Ensures deterministic reproducibility
            **kwargs
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Build the decision tree from the training set.
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Infer target values for new observations.
        Apply negative value correction to the output.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Run the model evaluation using the standardized set of metrics.
        """
        predictions = self.predict(X)
        self.metrics = calculate_metrics(y, predictions)
        return self.metrics
    
    def get_params(self, deep=True):
        """Restores the tree configuration parameters."""
        return self.model.get_params(deep=deep)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics
class LinearRegressionModel(BaseModel):
    def __init__(self, output_path:str, 
                 target: str = 'fare_amount',
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 positive: bool = False,
                 **kwargs
                 ):
        """
            Initialize the Linear Regression Model
            Args:
                output_path (str): Path where the model will be saved
                target (str): Target variable
                fit_intercept (bool): Whether to calculate the intercept for this model
                copy_X (bool): Whether to copy X before fitting
                positive (bool): Whether to restrict coefficients to be positive
        """
        
        super().__init__('linear_regression', target, output_path, model_type='baseline')
        self.model = LinearRegression(fit_intercept=fit_intercept,
                                      copy_X=copy_X,
                                      positive=positive,
                                      n_jobs=-1)
    
    def fit(self, X, y):
        """Train the linear regression model"""
        self.model.fit(X, y)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        return calculate_metrics(y, predictions)
    
    def get_params(self, deep=True):
        """Return model parameters"""
        return self.model.get_params(deep=deep)
        
class DecisionTreeModel(BaseModel):
    def __init__(self, output_path:str,target: str = 'fare_amount',
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
            Initialize the Decision Tree Model
            Args:
                output_path (str): Path where the model will be saved
                target (str): Target variable
                criterion (str): The function to measure the quality of a split 
                splitter (str): The strategy used to choose the split at each node
                max_depth (int): The maximum depth of the tree
                min_samples_split (int): The minimum number of samples required to split an internal node
                min_samples_leaf (int): The minimum number of samples required to be at a leaf node
                max_leaf_nodes (int): Grow a tree with max_leaf_nodes in best-first fashion
                min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value
                max_features (str): The number of features to consider when looking for the best split
        """
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
            random_state=RANDOM_STATE,
            **kwargs
        )
    
    def fit(self, X, y):
        """Train the decision tree model"""
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        return calculate_metrics(y, predictions)
    
    def get_params(self, deep=True):
        """Return model parameters"""
        return self.model.get_params(deep=deep)
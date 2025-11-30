from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.models.base_model import BaseModel
from src.config.settings import RANDOM_STATE
from src.evaluation.metrics import calculate_metrics
class LinearRegressionModel(BaseModel):
    def __init__(self, model_path:str, target: str = 'fare_amount', random_state: int = RANDOM_STATE):
        super().__init__('linear_regression', target, model_path)
        self.model = LinearRegression()
    
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
    
    def get_params(self):
        """Return model parameters"""
        return self.model.get_params()
        
class DecisionTreeModel(BaseModel):
    def __init__(self, model_path:str,target: str = 'fare_amount', max_depth: int = 10, random_state: int = RANDOM_STATE):
        super().__init__('decision_tree', target, model_path)
        self.model = DecisionTreeRegressor(
            max_depth=max_depth, 
            random_state=random_state,
            min_samples_split=20,  
            min_samples_leaf=10
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
    
    def get_params(self):
        """Return model parameters"""
        return self.model.get_params()
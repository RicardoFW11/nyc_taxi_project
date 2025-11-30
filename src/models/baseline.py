from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self, target: str = 'fare_amount'):
        super().__init__('linear_regression', target)
        self.model = LinearRegression()
        
class DecisionTreeModel(BaseModel):
    def __init__(self, target: str = 'fare_amount'):
        super().__init__('decision_tree', target)
        self.model = DecisionTreeRegressor(max_depth=10, random_state=42)
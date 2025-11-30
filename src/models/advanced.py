from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, target: str = 'fare_amount'):
        super().__init__('xgboost', target)
        self.model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
import pandas as pd
import time

class ModelComparison:
    def __init__(self, models):
        self.models = models
        
    def compare_models(self, X_train, X_test, y_train, y_test):
        results = []
        for model in self.models:
            # Train and evaluate each model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            
            results.append({
                'model': model.model_name,
                'training_time': training_time,
                **metrics
            })
        return pd.DataFrame(results)
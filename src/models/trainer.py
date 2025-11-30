import argparse
import joblib
from pathlib import Path
from .baseline_models import LinearRegressionModel, DecisionTreeModel
from .advanced_models import XGBoostModel, RandomForestModel
from ..evaluation.model_comparison import ModelComparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['linear', 'tree', 'xgboost', 'rf', 'all'])
    parser.add_argument('--target', choices=['fare_amount', 'trip_duration_minutes'])
    
    args = parser.parse_args()
    
    # Load your processed feature data
    data = pd.read_parquet('data/processed/features_engineered.parquet')
    X = data.drop(['fare_amount', 'trip_duration_minutes'], axis=1)
    y = data[args.target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if args.model == 'all':
        # Compare all models
        models = [
            LinearRegressionModel(args.target),
            DecisionTreeModel(args.target), 
            XGBoostModel(args.target),
            RandomForestModel(args.target)
        ]
        comparison = ModelComparison(models)
        results = comparison.compare_models(X_train, X_test, y_train, y_test)
        print(results)
        results.to_csv(f'model_comparison_{args.target}.csv')
    else:
        # Train single model
        model_dict = {
            'linear': LinearRegressionModel(args.target),
            'tree': DecisionTreeModel(args.target),
            'xgboost': XGBoostModel(args.target)
        }
        model = model_dict[args.model]
        model.fit(X_train, y_train)
        
        # Save model
        Path('models').mkdir(exist_ok=True)
        joblib.dump(model, f'models/{args.model}_{args.target}.pkl')

if __name__ == "__main__":
    main()
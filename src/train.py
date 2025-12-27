"""Main training pipeline"""
import yaml
import warnings
from src.utils import load_data, preprocess_data, create_submission
from src.models import train_lightgbm, train_xgboost, train_catboost, create_ensemble

warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("HEALTH INSURANCE CLAIMS PREDICTION")
    print("Zerve AI Datathon")
    print("="*70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    train, test = load_data(config['data']['train_path'], config['data']['test_path'])
    
    # Preprocess
    X_train, X_test, y_train, test_ids = preprocess_data(train, test, config)
    
    # Train models
    predictions = {}
    
    lgb_oof, lgb_test, lgb_gini = train_lightgbm(X_train, y_train, X_test, config)
    predictions['lgb'] = {'oof': lgb_oof, 'test': lgb_test, 'gini': lgb_gini}
    
    xgb_oof, xgb_test, xgb_gini = train_xgboost(X_train, y_train, X_test, config)
    predictions['xgb'] = {'oof': xgb_oof, 'test': xgb_test, 'gini': xgb_gini}
    
    cat_oof, cat_test, cat_gini = train_catboost(X_train, y_train, X_test, config)
    predictions['cat'] = {'oof': cat_oof, 'test': cat_test, 'gini': cat_gini}
    
    # Ensemble
    ensemble_preds, ensemble_gini = create_ensemble(predictions, y_train)
    
    # Create submission
    create_submission(test_ids, ensemble_preds, config['data']['submission_path'])
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"LightGBM: {lgb_gini:.6f}")
    print(f"XGBoost:  {xgb_gini:.6f}")
    print(f"CatBoost: {cat_gini:.6f}")
    print(f"Ensemble: {ensemble_gini:.6f} ⭐")
    print("="*70)

if __name__ == "__main__":
    main()
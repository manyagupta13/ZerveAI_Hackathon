"""Model training functions"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from .utils import gini_normalized

# ============================================================================
# TRAIN MODELS
# ============================================================================

def train_lightgbm(X_train, y_train, X_test, config):
    """Train LightGBM with CV"""
    print("\n🚀 Training LightGBM...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'verbose': -1,
        **config['lightgbm']
    }
    
    return train_with_cv(X_train, y_train, X_test, 'lgb', params, config)

def train_xgboost(X_train, y_train, X_test, config):
    """Train XGBoost with CV"""
    print("\n🚀 Training XGBoost...")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 10000,
        'n_jobs': -1,
        **config['xgboost']
    }
    
    return train_with_cv(X_train, y_train, X_test, 'xgb', params, config)

def train_catboost(X_train, y_train, X_test, config):
    """Train CatBoost with CV"""
    print("\n🚀 Training CatBoost...")
    
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 10000,
        'early_stopping_rounds': 100,
        'verbose': 100,
        **config['catboost']
    }
    
    return train_with_cv(X_train, y_train, X_test, 'cat', params, config)

# ============================================================================
# CV TRAINING
# ============================================================================

def train_with_cv(X_train, y_train, X_test, model_type, params, config):
    """Generic CV training"""
    n_folds = config['n_folds']
    random_state = config['random_state']
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val)
            
            model = lgb.train(
                params, train_data,
                num_boost_round=10000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
            )
            
            oof_preds[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / n_folds
            
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
            
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds
            
        elif model_type == 'cat':
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        fold_gini = gini_normalized(y_val, oof_preds[val_idx])
        print(f"  Fold Gini: {fold_gini:.6f}")
    
    overall_gini = gini_normalized(y_train, oof_preds)
    print(f"\n  ✅ Overall Gini: {overall_gini:.6f}")
    
    return oof_preds, test_preds, overall_gini

# ============================================================================
# ENSEMBLE
# ============================================================================

def create_ensemble(predictions_dict, y_train):
    """Create weighted ensemble"""
    print("\n🎯 Creating Ensemble...")
    
    best_gini = 0
    best_weights = None
    
    for w1 in np.arange(0.2, 0.5, 0.1):
        for w2 in np.arange(0.2, 0.5, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0.2 or w3 > 0.5:
                continue
            
            ensemble_oof = (w1 * predictions_dict['lgb']['oof'] + 
                           w2 * predictions_dict['xgb']['oof'] + 
                           w3 * predictions_dict['cat']['oof'])
            
            gini_score = gini_normalized(y_train, ensemble_oof)
            
            if gini_score > best_gini:
                best_gini = gini_score
                best_weights = (w1, w2, w3)
    
    print(f"  Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"  ✅ Ensemble Gini: {best_gini:.6f}")
    
    ensemble_test = (best_weights[0] * predictions_dict['lgb']['test'] + 
                     best_weights[1] * predictions_dict['xgb']['test'] + 
                     best_weights[2] * predictions_dict['cat']['test'])
    
    return ensemble_test, best_gini
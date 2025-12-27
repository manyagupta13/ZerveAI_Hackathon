"""Utilities for data processing and evaluation"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# METRICS
# ============================================================================

def gini(actual, pred):
    """Calculate Gini coefficient"""
    assert len(actual) == len(pred)
    all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)

def gini_normalized(actual, pred):
    """Calculate Normalized Gini coefficient"""
    return gini(actual, pred) / gini(actual, actual)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(train_path, test_path):
    """Load train and test data"""
    print("📂 Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")
    print(f"  Target: {train['target'].value_counts(normalize=True).to_dict()}")
    
    return train, test

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df, numeric_features, binary_features, categorical_features):
    """Create engineered features"""
    df = df.copy()
    
    # Numeric aggregations
    df['numeric_sum'] = df[numeric_features].sum(axis=1)
    df['numeric_mean'] = df[numeric_features].mean(axis=1)
    df['numeric_std'] = df[numeric_features].std(axis=1)
    df['numeric_range'] = df[numeric_features].max(axis=1) - df[numeric_features].min(axis=1)
    
    # Binary aggregations
    df['binary_sum'] = df[binary_features].sum(axis=1)
    
    # Missing indicators
    df['missing_count'] = df.isnull().sum(axis=1)
    
    # Interactions
    for i, f1 in enumerate(numeric_features[:5]):
        for f2 in numeric_features[i+1:6]:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    
    df.fillna(0, inplace=True)
    return df

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_data(train, test, config):
    """Preprocess train and test data"""
    print("\n🔧 Preprocessing...")
    
    binary_features = config['features']['binary']
    categorical_features = config['features']['categorical']
    numeric_features = config['features']['numeric']
    
    # Separate target
    y_train = train['target']
    X_train = train.drop(['target', 'id'], axis=1, errors='ignore')
    X_test = test.drop(['id'], axis=1, errors='ignore')
    test_ids = test['id'] if 'id' in test.columns else None
    
    # Handle missing values
    for col in numeric_features:
        median = X_train[col].median()
        X_train[col].fillna(median, inplace=True)
        X_test[col].fillna(median, inplace=True)
    
    for col in binary_features:
        X_train[col].fillna(0, inplace=True)
        X_test[col].fillna(0, inplace=True)
    
    for col in categorical_features:
        X_train[col].fillna('missing', inplace=True)
        X_test[col].fillna('missing', inplace=True)
    
    # Label encode categoricals
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Feature engineering
    X_train = create_features(X_train, numeric_features, binary_features, categorical_features)
    X_test = create_features(X_test, numeric_features, binary_features, categorical_features)
    
    print(f"  Final shapes: {X_train.shape}, {X_test.shape}")
    
    return X_train, X_test, y_train, test_ids

# ============================================================================
# SUBMISSION
# ============================================================================

def create_submission(test_ids, predictions, filepath):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_ids,
        'target': np.clip(predictions, 0, 1)
    })
    submission.to_csv(filepath, index=False)
    print(f"\n✅ Submission saved: {filepath}")
    print(submission['target'].describe())
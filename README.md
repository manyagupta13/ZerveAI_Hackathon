# 🏥 Health Insurance Claims Prediction

> **Zerve AI Datathon 2025** - Machine Learning solution for predicting health insurance claims using ensemble methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: clean](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)

## 🎯 Problem Statement

Health insurance companies need to accurately identify high-risk customers who are likely to file significant claims. This project develops a machine learning model to predict the probability of a customer filing a health insurance claim using 50 anonymized engineered features.

**Challenge:** Imbalanced binary classification with mixed data types (binary, categorical, numeric features)

## 📊 Results

Our ensemble model achieved strong performance on the Normalized Gini Coefficient metric:

| Model | OOF Gini Score | 
|-------|----------------|
| LightGBM | 0.279636 |
| XGBoost | 0.280097 |
| CatBoost | 0.279584 |
| **Ensemble** | **0.282137** ⭐ |

The ensemble uses optimized weighted averaging of all three models with 5-fold stratified cross-validation.

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/zerve-ai-datathon.git
cd zerve-ai-datathon
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your data**
Place your CSV files in the `data/` folder:
- `training_data.csv` (with target column)
- `test_data.csv` (without target column)

### Training & Prediction

Run the complete pipeline:
```bash
python train.py
```

This will:
- ✅ Load and preprocess data
- ✅ Engineer features
- ✅ Train LightGBM, XGBoost, and CatBoost models
- ✅ Create optimized ensemble
- ✅ Generate submission file (`data/k_zerveai_datathon.csv`)

## 📁 Project Structure

```
zerve-ai-datathon/
├── README.md                    # You are here
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration & hyperparameters
├── train.py                     # Main training pipeline
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory
│   ├── training_data.csv        # Training dataset
│   ├── test_data.csv            # Test dataset
│   └── submission.csv   # Submission file (generated)
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── utils.py                 # Data processing & metrics
│   └── models.py                # Model training functions
│
├── notebooks/                   # Jupyter notebooks
│   └── eda.ipynb               # Exploratory data analysis
│
└── docs/                        # Documentation
    └── methodology.md           # Detailed methodology
```

## 🧠 Methodology

### 1. **Data Preprocessing**
- **Missing Values**: Median imputation for numeric, mode for categorical, zero for binary
- **Encoding**: Label encoding for categorical features
- **Feature Types**: 17 binary, 14 categorical, 19 numeric features

### 2. **Feature Engineering**
- Statistical aggregations (sum, mean, std, range) for numeric features
- Binary feature counts
- Missing value indicators
- Interaction features (multiplication and division of top numeric features)

### 3. **Model Training**
Three gradient boosting models trained with 5-fold stratified cross-validation:

#### LightGBM
- Fast, efficient gradient boosting
- Handles categorical features natively
- Best for: Speed and memory efficiency

#### XGBoost
- Robust regularization
- Excellent performance on structured data
- Best for: Complex patterns

#### CatBoost
- Symmetric tree structure
- Built-in categorical handling
- Best for: Reducing overfitting

### 4. **Ensemble Strategy**
- **Method**: Weighted average of model predictions
- **Optimization**: Grid search over weight space [0.2, 0.5]
- **Final Weights**: Automatically determined to maximize Gini coefficient

### 5. **Evaluation Metric**
**Normalized Gini Coefficient**: Measures ranking quality of predicted probabilities
```
Normalized Gini = Gini(model) / Gini(perfect model)
```
- Range: 0 (random) to 1 (perfect)
- Ideal for imbalanced datasets
- Industry standard for insurance risk modeling

## ⚙️ Configuration

Modify `config.yaml` to customize:

```yaml
# Cross-validation settings
random_state: 42
n_folds: 5

# Model hyperparameters
lightgbm:
  learning_rate: 0.01
  num_leaves: 31
  max_depth: 6
  scale_pos_weight: 3  # Handle class imbalance

xgboost:
  learning_rate: 0.01
  max_depth: 6
  scale_pos_weight: 3

catboost:
  learning_rate: 0.01
  depth: 6
  scale_pos_weight: 3
```

## 📈 Model Performance Details

### Cross-Validation Strategy
- **Type**: Stratified K-Fold (k=5)
- **Stratification**: Maintains class distribution across folds
- **Validation**: Out-of-fold (OOF) predictions for unbiased evaluation

### Handling Class Imbalance
- `scale_pos_weight=3`: Increases weight of minority class
- Stratified sampling: Ensures balanced folds
- Gini metric: Robust to imbalance

## 🔧 Advanced Usage

### Custom Feature Engineering
Edit `src/utils.py` → `create_features()` function:
```python
def create_features(df, numeric_features, binary_features, categorical_features):
    # Add your custom features here
    df['your_feature'] = ...
    return df
```

### Hyperparameter Tuning
Modify `config.yaml` or use grid search in `train.py`

### Using Individual Models
```python
from src.models import train_lightgbm
from src.utils import load_data, preprocess_data

# Load and preprocess
train, test = load_data('data/training_data.csv', 'data/test_data.csv')
X_train, X_test, y_train, test_ids = preprocess_data(train, test, config)

# Train single model
oof_preds, test_preds, gini = train_lightgbm(X_train, y_train, X_test, config)
```

## 📊 Feature Importance Analysis

Top features by importance (averaged across models):
1. `numeric_mean` - Average of all numeric features
2. `feature_13` - Original numeric feature
3. `binary_sum` - Count of active binary features
4. `feature_1_x_feature_2` - Interaction feature
5. `missing_count` - Missing value indicator

## 🧪 Testing

Run tests to verify functionality:
```bash
python -m pytest tests/
```

## 📝 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
pyyaml>=6.0
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- **Zerve AI** and **IIT Bombay** for organizing the datathon
- Gradient boosting libraries: LightGBM, XGBoost, CatBoost
- The open-source ML community

---

⭐ **If you found this helpful, please star the repository!**

*Built with ❤️ for the Zerve AI Datathon 2025*

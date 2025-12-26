# Health Insurance Claims Prediction Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end machine learning pipeline for predicting health insurance claims using ensemble methods. Built for the Zerve AI Datathon, optimized for Normalized Gini Coefficient.

## 🎯 Project Overview

This project implements a robust ML pipeline that:
- Processes 50 features (binary, categorical, and numeric)
- Engineers advanced features for improved predictions
- Trains three gradient boosting models (LightGBM, XGBoost, CatBoost)
- Creates an optimized weighted ensemble
- Achieves competitive performance through stratified cross-validation

## 📊 Results

| Model | OOF Gini Score |
|-------|----------------|
| LightGBM | 0.279636 |
| XGBoost | 0.280097 |
| CatBoost | 0.279584 |
| **Ensemble** | **0.282137** ⭐ |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```python
python main.py
```

This will:
1. Load and preprocess the data
2. Train all three models with 5-fold CV
3. Create an ensemble
4. Generate `k_zerveai_datathon.csv` submission file

## 📁 Project Structure
```
├── main.py                 
├── requirements.txt        
├── README.md              
├── docs/
│   ├── METHODOLOGY.md    
│   ├── FEATURES.md        
│   └── API.md             
├── data/
│   ├── train.csv          
│   └── test.csv           
└── notebooks/
    └── EDA.ipynb         
```

## 🔧 Configuration

Key parameters can be modified in the `CONFIG` dictionary:
```python
CONFIG = {
    'n_folds': 5,              # Number of CV folds
    'random_state': 42,        # Reproducibility seed
    'early_stopping_rounds': 100,
    'verbose': 100
}
```

## 📈 Feature Engineering

The pipeline creates several feature groups:

- **Statistical Features**: sum, mean, std, min, max, range of numeric features
- **Binary Aggregations**: sum and mean of binary features
- **Missing Indicators**: count of missing values by feature type
- **Interaction Features**: multiplication and division of top numeric pairs

See [FEATURES.md](docs/FEATURES.md) for detailed documentation.

## 🤖 Models

### LightGBM
- Objective: binary classification
- Learning rate: 0.01
- Max depth: 6
- Scale pos weight: 3 (handles imbalance)

### XGBoost
- Objective: binary:logistic
- Learning rate: 0.01
- Max depth: 6
- Scale pos weight: 3

### CatBoost
- Loss function: Logloss
- Learning rate: 0.01
- Depth: 6
- Scale pos weight: 3

## 🎲 Ensemble Strategy

Weighted ensemble using grid search to find optimal weights that maximize OOF Gini score:
- Weights range: 0.2 - 0.5 for each model
- Optimization metric: Normalized Gini Coefficient

## 📊 Evaluation Metric

**Normalized Gini Coefficient**:
```
Normalized Gini = Gini(actual, predicted) / Gini(actual, actual)
```

This metric measures the model's ability to rank predictions, crucial for insurance risk assessment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Zerve AI and IIT Bombay for hosting the datathon

Project Link: [https://github.com/yourusername/insurance-claims-prediction](https://github.com/yourusername/insurance-claims-prediction)

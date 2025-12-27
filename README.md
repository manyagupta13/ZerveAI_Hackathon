# Health Insurance Claims Prediction 🏥

Zerve AI Datathon - Predicting health insurance claims using ML

## Results
- **Ensemble Gini: 0.282137** ⭐
- LightGBM: 0.279636
- XGBoost: 0.280097  
- CatBoost: 0.279584

## Quick Start
```bash
pip install -r requirements.txt
python train.py
```

## Structure
- `train.py` - Main pipeline
- `src/utils.py` - Preprocessing & metrics
- `src/models.py` - Model training
- `config.yaml` - Hyperparameters
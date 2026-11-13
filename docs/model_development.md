# Model Development

## Models Compared
1. Logistic Regression
2. Random Forest
3. XGBoost

## Training
Training: src/models/trainer.py
Model selection: src/model_selection/selector.py

## Evaluation Metrics
- Precision
- Recall
- F1-score
- Average Precision (PR-AUC)
- ROC-AUC

## XGBoost Validation Results
- Average Precision: 0.9204
- Precision: 0.9444
- Recall: 0.8500
- F1-score: 0.8947

## Selected Model
XGBoost was selected as the champion model.
Registered model: Financial_Fraud_Detection_Model
Champion alias: champion

## Experiment Tracking
MLflow tracks experiments, model versions and evaluation artifacts.

## Important
The dataset is highly imbalanced. Precision, recall and PR-AUC are important evaluation metrics.

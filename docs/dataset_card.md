# Dataset Documentation

## Dataset Information

- Dataset: PaySim Synthetic Financial Transactions
- Source: https://www.kaggle.com/datasets/ealaxi/paysim1
- Original rows: 6,362,620
- Sample rows: 200,000
- Target column: isFraud
- Random seed: 42

## Sampling Methodology

A reproducible simple random sample of 200,000 rows was selected
without replacement using pandas DataFrame.sample().

- Random state: 42
- Stratification: Not applied
- Artificial balancing: Not applied

## Sample Class Distribution

| Class | Meaning | Transactions | Percentage |
|---|---|---:|---:|
| 0 | Legitimate | 199,732 | 99.866% |
| 1 | Fraudulent | 268 | 0.134% |

## Data Quality

| Check | Result |
|---|---|
| Rows | 200,000 |
| Columns | 11 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Both classes present | Yes |

## Dataset Columns

step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

## Important Considerations

- PaySim is synthetic data, not actual bank transaction data.
- The dataset is highly imbalanced.
- Accuracy alone is insufficient for fraud detection.
- We will evaluate precision, recall, F1-score, PR-AUC and ROC-AUC.
- Any resampling will be restricted to training data.
- Validation and test sets will retain their original distributions.

## Storage Locations

- Original: data/raw/
- Sample: data/samples/paysim_sample_200k.csv
- Validated: data/validated/
- Processed: data/processed/

## Reproducibility

sample_df = df.sample(n=200_000, random_state=42)

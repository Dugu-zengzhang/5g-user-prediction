# 5G User Prediction

Artificial Intelligence Group Project — Predict whether a user is a **5G user** based on user profile and communication data.

## Overview

This project aims to build a binary classification model to predict potential 5G users for telecom operators. Using user-side information such as billing data, traffic usage, activity patterns, package types, and regional information, the model helps operators with targeted marketing and user profiling.

## Dataset

The dataset contains **60 fields**:

| Field Group | Description | Count |
|-------------|-------------|-------|
| `id` | Sample identifier | 1 |
| `cat_0 ~ cat_19` | Categorical features | 20 |
| `num_0 ~ num_37` | Numerical features | 38 |
| `target` | Ground truth (0: Non-5G, 1: 5G User) | 1 |

## Algorithms

Two ensemble learning algorithms are implemented:

1. **Random Forest (Bagging)** - `RandomForestClassifier` from scikit-learn
2. **XGBoost (Boosting)** - `XGBClassifier` from XGBoost

## Evaluation Metric

**AUC** (Area Under the ROC Curve) — higher is better.

## Project Structure

```
├── 5G_User_Prediction.ipynb   # Jupyter notebook with full pipeline
├── Random_Forest.py           # Standalone training script
├── train.csv                  # Training dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Pipeline

1. **Data Loading & EDA**
2. **Feature Engineering** — One-hot encoding for categorical features, statistical aggregations for numerical features
3. **Feature Selection** — SelectKBest with f_classif (top 100 features)
4. **Data Preprocessing** — StandardScaler, SMOTE for class imbalance
5. **Model Training** — Random Forest & XGBoost
6. **Evaluation** — AUC score, classification report, ROC curve
7. **Feature Importance Analysis** — Top contributing features visualized

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
Run the notebook cell by cell:

```bash
jupyter notebook 5G_User_Prediction.ipynb
```

### Python Script
Execute the standalone training script:

```bash
python Random_Forest.py
```

## Results

| Model | AUC | Training Time |
|-------|-----|---------------|
| Random Forest | **0.8838** | 395.60s |
| XGBoost | **0.8598** | 5.83s |

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | num_37 | 0.0645 |
| 2 | cat_11 | 0.0485 |
| 3 | num_3 | 0.0485 |
| 4 | num_25 | 0.0467 |
| 5 | num_18 | 0.0465 |
| 6 | cat_10 | 0.0380 |
| 7 | cat_0_2 | 0.0314 |
| 8 | cat_0_5 | 0.0310 |
| 9 | num_27 | 0.0290 |
| 10 | cat_12 | 0.0282 |

## Contributors

Software Engineering Department, Team project for Artificial Intelligence course.

## License

This project is for educational purposes only.

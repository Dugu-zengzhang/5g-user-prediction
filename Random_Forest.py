"""
5G User Prediction - Random Forest & XGBoost
=============================================
Course: Artificial Intelligence Group Project
Description: Predict whether a user is a 5G user based on user profile and 
             communication data using ensemble learning methods.
Evaluation Metric: AUC (Area Under the ROC Curve)
"""

import pandas as pd
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

print("=" * 60)
print("  5G User Prediction - Random Forest & XGBoost")
print("=" * 60)

# ============================================================
# Step 1: Load Data
# ============================================================
print("\n[Step 1/9] Loading data...")
start_time = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
train_csv_path = os.path.join(script_dir, 'train.csv')
data = pd.read_csv(train_csv_path)

print(f"  Data shape: {data.shape}")
print(f"  Target distribution:\n{data['target'].value_counts()}")
print(f"  Positive ratio: {data['target'].mean():.4f}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 2: Exploratory Data Analysis (EDA)
# ============================================================
print("\n[Step 2/9] Exploratory Data Analysis...")
start_time = time.time()

# Identify categorical and numerical features
cat_cols = [col for col in data.columns if col.startswith('cat_')]
num_cols = [col for col in data.columns if col.startswith('num_')]
print(f"  Categorical features: {len(cat_cols)}")
print(f"  Numerical features: {len(num_cols)}")

# Check for missing values
missing_vals = data.isnull().sum().sum()
print(f"  Total missing values: {missing_vals}")

# Check unique values per categorical feature
for col in cat_cols[:5]:
    print(f"  {col}: {data[col].nunique()} unique values")

print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 3: Feature Engineering - One-Hot Encoding
# ============================================================
print("\n[Step 3/9] Encoding categorical features...")
start_time = time.time()

# One-hot encode ALL categorical features
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(f"  Features after one-hot encoding: {data.shape[1]}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 4: Feature Engineering - Numerical Aggregations
# ============================================================
print("\n[Step 4/9] Creating numerical aggregation features...")
start_time = time.time()

# Create statistical summaries from numerical features
data['num_mean'] = data[[c for c in data.columns if c.startswith('num_')]].mean(axis=1)
data['num_std'] = data[[c for c in data.columns if c.startswith('num_')]].std(axis=1)
data['num_max'] = data[[c for c in data.columns if c.startswith('num_')]].max(axis=1)
data['num_min'] = data[[c for c in data.columns if c.startswith('num_')]].min(axis=1)

print(f"  Features after aggregation: {data.shape[1]}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 5: Split Features and Target
# ============================================================
print("\n[Step 5/9] Splitting features and target...")
start_time = time.time()

X = data.drop(columns=['id', 'target'])
y = data['target']

print(f"  Feature matrix: {X.shape}")
print(f"  Target vector: {y.shape}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 6: Feature Selection (Top 100 features)
# ============================================================
print("\n[Step 6/9] Selecting top features...")
start_time = time.time()

# Sample data for feature selection if dataset is large
n_samples = min(50000, len(X))
X_sample = X.sample(n=n_samples, random_state=42)
y_sample = y.loc[X_sample.index]

selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
selector.fit(X_sample.fillna(0), y_sample)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
selected_features = [X.columns[i] for i in selected_indices]
X_selected = X.iloc[:, selected_indices]

print(f"  Selected {len(selected_features)} features out of {X.shape[1]}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 7: Preprocessing - Scale & Handle Imbalance
# ============================================================
print("\n[Step 7/9] Scaling and handling class imbalance...")
start_time = time.time()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected.fillna(0))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Train - Positive ratio: {y_train.mean():.4f}")

# Apply SMOTE to handle class imbalance
print("  Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  After SMOTE - Positive ratio: {y_train_res.mean():.4f}")
print(f"  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Step 8: Model Training
# ============================================================
print("\n[Step 8/9] Training models...")

# --- Random Forest ---
print("\n  --- Random Forest ---")
print("  Params: n_estimators=100, max_depth=None, class_weight='balanced'")
start_time = time.time()

model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model_rf.fit(X_train_res, y_train_res)
rf_time = time.time() - start_time
print(f"  Training completed in {rf_time:.2f}s")

# --- XGBoost ---
print("\n  --- XGBoost ---")
print("  Params: learning_rate=0.1, max_depth=6, scale_pos_weight=calculated")
start_time = time.time()

# Calculate ratio of negative to positive samples
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    verbosity=0,
    use_label_encoder=False
)
model_xgb.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_time = time.time() - start_time
print(f"  Training completed in {xgb_time:.2f}s")

# ============================================================
# Step 9: Evaluation
# ============================================================
print("\n[Step 9/9] Evaluating models...")
start_time = time.time()

# --- Random Forest Evaluation ---
y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]
y_pred_rf = model_rf.predict(X_test)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

# --- XGBoost Evaluation ---
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = model_xgb.predict(X_test)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\n  Time: {time.time() - start_time:.2f}s")

# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Model':<20} {'AUC':<10} {'Training Time':<15}")
print(f"  {'-'*45}")
print(f"  {'Random Forest':<20} {auc_rf:<10.4f} {rf_time:<15.2f}s")
print(f"  {'XGBoost':<20} {auc_xgb:<10.4f} {xgb_time:<15.2f}s")

print(f"\n  Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=['Non-5G', '5G User']))

print(f"\n  Classification Report - XGBoost:")
print(classification_report(y_test, y_pred_xgb, target_names=['Non-5G', '5G User']))

# ============================================================
# Feature Importance Analysis
# ============================================================
print("\n" + "=" * 60)
print("  FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# XGBoost Feature Importance
if hasattr(model_xgb, 'feature_importances_'):
    importances = model_xgb.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    print("\n  Top 15 Features (XGBoost):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':<10}")
    print(f"  {'-'*51}")
    for i, idx in enumerate(indices):
        feat_name = selected_features[idx][:35] if idx < len(selected_features) else f"feature_{idx}"
        print(f"  {i+1:<6} {feat_name:<35} {importances[idx]:<10.4f}")

# Random Forest Feature Importance
if hasattr(model_rf, 'feature_importances_'):
    rf_importances = model_rf.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:15]
    
    print("\n  Top 15 Features (Random Forest):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':<10}")
    print(f"  {'-'*51}")
    for i, idx in enumerate(rf_indices):
        feat_name = selected_features[idx][:35] if idx < len(selected_features) else f"feature_{idx}"
        print(f"  {i+1:<6} {feat_name:<35} {rf_importances[idx]:<10.4f}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)

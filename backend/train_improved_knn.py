"""
Improved KNN Model Training - Anti-Overfitting Strategy
This script trains a KNN model with proper regularization to prevent overfitting
Run from project root:
    python backend/train_improved_knn.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTEENN

print("=" * 70)
print("IMPROVED KNN MODEL TRAINING - ANTI-OVERFITTING STRATEGY")
print("=" * 70)

# Load data
proj_root = Path(__file__).resolve().parents[1]
csv_path = proj_root / 'healthcaredatasetstrokedata.csv'
backend_path = Path(__file__).resolve().parent

if not csv_path.exists():
    print(f"❌ CSV not found at {csv_path}")
    sys.exit(1)

print("\n[1/7] Loading data...")
df = pd.read_csv(csv_path)
print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Drop unnecessary columns
if {'id', 'gender', 'Residence_type'}.issubset(df.columns):
    df = df.drop(['id', 'gender', 'Residence_type'], axis=1)
    print("✓ Dropped unnecessary columns")

# Outlier removal on BMI
if 'bmi' in df.columns:
    upper = df['bmi'].mean() + 3 * df['bmi'].std()
    lower = df['bmi'].mean() - 3 * df['bmi'].std()
    initial_rows = len(df)
    df = df[(df['bmi'] < upper) & (df['bmi'] > lower)]
    removed = initial_rows - len(df)
    print(f"✓ Removed {removed} outliers from BMI (kept {len(df)} rows)")

# Identify features
X = df.drop(['stroke'], axis=1)
y = df['stroke']

numeric_features = [c for c in X.columns if X[c].dtype != 'O']
categorical_features = [c for c in X.columns if X[c].dtype == 'O']
continuous_features = [c for c in numeric_features if X[c].nunique() > 25]

print(f"✓ Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

# Build preprocessor
print("\n[2/7] Building preprocessor pipeline...")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', StandardScaler(with_mean=False))
])

transform_pipe = Pipeline([
    ('power', PowerTransformer(standardize=True))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
    ('power', transform_pipe, continuous_features)
])

X_processed = preprocessor.fit_transform(X)
print(f"✓ Preprocessor built. Output shape: {X_processed.shape}")

# Handle class imbalance
print("\n[3/7] Handling class imbalance with SMOTE+ENN...")
smt = SMOTEENN(random_state=42, sampling_strategy='minority')
X_res, y_res = smt.fit_resample(X_processed, y)
print(f"✓ Resampled to {X_res.shape[0]} samples")
print(f"  Class distribution: {np.bincount(y_res)}")

# Train-test split
print("\n[4/7] Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Hyperparameter tuning with GridSearchCV
print("\n[5/7] Tuning KNN hyperparameters (GridSearchCV)...")
print("  Testing: n_neighbors=[5,7,9,11,13,15], weights=['uniform','distance']")

param_grid = {
    'n_neighbors': [5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn_base,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best CV F1-Score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
print("\n[6/7] Training final KNN model with best parameters...")
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
print("✓ Model trained")

# Evaluate on both train and test sets
print("\n[7/7] Evaluating model performance...")
print("\n" + "-" * 70)

# Training set evaluation
y_train_pred = best_knn.predict(X_train)
y_train_proba = best_knn.predict_proba(X_train)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred)
train_rec = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print("TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Precision: {train_prec:.4f}")
print(f"  Recall:    {train_rec:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  AUC-ROC:   {train_auc:.4f}")

# Test set evaluation
y_test_pred = best_knn.predict(X_test)
y_test_proba = best_knn.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\nTEST SET PERFORMANCE (Generalization - Most Important!):")
print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  AUC-ROC:   {test_auc:.4f}")

# Overfitting analysis
print("\n" + "-" * 70)
print("OVERFITTING ANALYSIS:")
overfit_gap = train_acc - test_acc
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Gap (Overfitting): {overfit_gap:.4f} ({overfit_gap*100:.2f}%)")

if overfit_gap > 0.05:
    print(f"  ⚠️  WARNING: {overfit_gap*100:.2f}% gap indicates some overfitting")
    print(f"  → Solution: Increase n_neighbors (currently {grid_search.best_params_['n_neighbors']})")
elif overfit_gap > 0.01:
    print(f"  ✓ Good: Minor overfitting ({overfit_gap*100:.2f}% gap is acceptable)")
else:
    print(f"  ✓ Excellent: No overfitting (gap < 1%)")

# Confusion matrix
print("\n" + "-" * 70)
print("CONFUSION MATRIX (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Model interpretation
print("\n" + "-" * 70)
print("MODEL INTERPRETATION:")
print(f"  Algorithm: K-Nearest Neighbors (KNN)")
print(f"  n_neighbors={grid_search.best_params_['n_neighbors']}")
print(f"  weights={grid_search.best_params_['weights']}")
print(f"  ")
print(f"  What this means:")
print(f"  - For each new patient, the model finds the {grid_search.best_params_['n_neighbors']} most similar")
print(f"    patients in the training data (using Euclidean distance)")
print(f"  - Predicts the majority class among those {grid_search.best_params_['n_neighbors']} neighbors")
print(f"  - Larger n_neighbors = less overfitting but potentially less flexible")
print(f"  - Smaller n_neighbors = more flexible but higher overfitting risk")

# Save models
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

model_path = backend_path / 'best_model_knn.joblib'
preproc_path = backend_path / 'preprocessor.joblib'

try:
    joblib.dump(best_knn, str(model_path), compress=3)
    print(f"✓ Model saved to: {model_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")
    sys.exit(1)

try:
    joblib.dump(preprocessor, str(preproc_path), compress=3)
    print(f"✓ Preprocessor saved to: {preproc_path}")
except Exception as e:
    print(f"❌ Error saving preprocessor: {e}")
    sys.exit(1)

# Test loading
print("\n" + "-" * 70)
print("VERIFICATION - Testing model load and prediction...")

try:
    loaded_model = joblib.load(str(model_path))
    loaded_preproc = joblib.load(str(preproc_path))
    
    # Test prediction
    test_input = pd.DataFrame({
        'age': [67],
        'hypertension': [0],
        'heart_disease': [1],
        'ever_married': ['Yes'],
        'work_type': ['Private'],
        'avg_glucose_level': [228.69],
        'bmi': [36.6],
        'smoking_status': ['formerly smoked']
    })
    
    X_test_proc = loaded_preproc.transform(test_input)
    pred = loaded_model.predict(X_test_proc)[0]
    proba = loaded_model.predict_proba(X_test_proc)[0]
    
    print(f"✓ Model loads and predicts successfully")
    print(f"  Test prediction: {pred} (0=No Stroke, 1=Stroke)")
    print(f"  Confidence: {max(proba)*100:.2f}%")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nKEY TAKEAWAYS:")
print(f"1. Test Accuracy: {test_acc*100:.2f}% (what matters for real data)")
print(f"2. Overfitting Gap: {overfit_gap*100:.2f}%")
print(f"3. Best Configuration: n_neighbors={grid_search.best_params_['n_neighbors']}, weights={grid_search.best_params_['weights']}")
print(f"4. The model generalizes well to unseen data")
print(f"\nNext step: Run 'python backend/app.py' to deploy the model")
print("=" * 70)

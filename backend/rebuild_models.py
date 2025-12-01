"""
Rebuild model files script
This script regenerates the model and preprocessor files with the current scikit-learn version
Run this from the project root directory where the notebook and CSV are located
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import from main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("REBUILDING MODEL FILES")
print("=" * 60)

# Try to load the CSV and check if we need to retrain
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'healthcaredatasetstrokedata.csv')

if not os.path.exists(csv_path):
    print(f"\n❌ CSV file not found at: {csv_path}")
    print("\nPlease run this script from the project root directory where:")
    print("  - healthcaredatasetstrokedata.csv is located")
    print("  - Heart_Stroke_ML_Project.ipynb is located")
    sys.exit(1)

print(f"\n✓ Found CSV file: {csv_path}")

# Load and preprocess data (using the same pipeline as notebook)
print("\nLoading and preprocessing data...")

df = pd.read_csv(csv_path)
print(f"  Dataset shape: {df.shape}")

# Remove unnecessary columns
df.drop(['id', 'gender', 'Residence_type'], inplace=True, axis=1)

# Identify features
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
continuous_features = [feature for feature in numeric_features if len(df[feature].unique()) > 25]

print(f"  Numeric features: {len(numeric_features)}")
print(f"  Categorical features: {len(categorical_features)}")
print(f"  Continuous features: {len(continuous_features)}")

# Outlier removal for BMI
def outlier_removal(column, df):
    upper_limit = df[column].mean() + 3 * df[column].std()
    lower_limit = df[column].mean() - 3 * df[column].std()
    df = df[(df[column] < upper_limit) & (df[column] > lower_limit)]
    return df

print("\nRemoving outliers...")
df = outlier_removal('bmi', df)
print(f"  Dataset shape after outlier removal: {df.shape}")

# Split X and y
X = df.drop(['stroke'], axis=1)
y = df['stroke']

# Create preprocessor with the same pipeline
print("\nBuilding preprocessor pipeline...")

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
])

transform_pipe = Pipeline(steps=[
    ('transformer', PowerTransformer(standardize=True))
])

preprocessor = ColumnTransformer([
    ("numeric_Pipeline", numeric_pipeline, numeric_features[:-1]),  # Exclude 'stroke'
    ("Categorical_Pipeline", categorical_pipeline, categorical_features),
    ("Power_Transformation", transform_pipe, continuous_features)
])

print("✓ Preprocessor built successfully")

# Preprocess data
print("\nPreprocessing data...")
X_processed = preprocessor.fit_transform(X)
print(f"  Processed shape: {X_processed.shape}")

# Apply SMOTE to handle imbalance
print("\nHandling class imbalance with SMOTE...")
from imblearn.combine import SMOTEENN

smt = SMOTEENN(random_state=42, sampling_strategy='minority')
X_res, y_res = smt.fit_resample(X_processed, y)
print(f"  Resampled shape: {X_res.shape}")
print(f"  Class distribution: {np.bincount(y_res)}")

# Train the KNN model
print("\nTraining KNN model...")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Using best parameters found in notebook
best_model_knn = KNeighborsClassifier(
    algorithm='auto',
    n_neighbors=3,
    weights='uniform'
)

best_model_knn.fit(X_train, y_train)
print("✓ Model trained successfully")

# Evaluate
y_pred = best_model_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save models
backend_path = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

model_path = os.path.join(backend_path, 'best_model_knn.joblib')
preprocessor_path = os.path.join(backend_path, 'preprocessor.joblib')

try:
    joblib.dump(best_model_knn, model_path, compress=3)
    print(f"\n✓ Model saved: {model_path}")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")
    sys.exit(1)

try:
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    print(f"✓ Preprocessor saved: {preprocessor_path}")
except Exception as e:
    print(f"❌ Error saving preprocessor: {e}")
    sys.exit(1)

# Verify files
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model file exists: {size_mb:.2f} MB")
else:
    print("❌ Model file not found!")

if os.path.exists(preprocessor_path):
    size_mb = os.path.getsize(preprocessor_path) / (1024 * 1024)
    print(f"✓ Preprocessor file exists: {size_mb:.2f} MB")
else:
    print("❌ Preprocessor file not found!")

# Try to load them back
print("\nTesting model loading...")
try:
    loaded_model = joblib.load(model_path)
    print("✓ Model loads successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

try:
    loaded_preprocessor = joblib.load(preprocessor_path)
    print("✓ Preprocessor loads successfully")
except Exception as e:
    print(f"❌ Error loading preprocessor: {e}")
    sys.exit(1)

# Test a prediction
print("\nTesting prediction...")
try:
    test_data = pd.DataFrame({
        'age': [67],
        'hypertension': [0],
        'heart_disease': [1],
        'ever_married': ['Yes'],
        'work_type': ['Private'],
        'avg_glucose_level': [228.69],
        'bmi': [36.6],
        'smoking_status': ['formerly smoked']
    })
    
    X_test_proc = loaded_preprocessor.transform(test_data)
    pred = loaded_model.predict(X_test_proc)
    proba = loaded_model.predict_proba(X_test_proc)
    
    print(f"✓ Test prediction: {pred[0]}")
    print(f"  Probabilities: No Stroke={proba[0][0]:.2%}, Stroke={proba[0][1]:.2%}")
except Exception as e:
    print(f"❌ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nYou can now run: python app.py")

# ============================================
# HealthGuard AI – Model Training Script
# ============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================
# 1. LOAD DATASETS
# ============================================

diabetes = pd.read_csv("datasets/diabetes.csv")
heart = pd.read_csv("datasets/heart.csv")
kidney = pd.read_csv("datasets/kidney.csv")

print("✅ Datasets loaded successfully")

# ============================================
# 2. DIABETES DATA PREPROCESSING
# ============================================

diabetes.fillna(diabetes.mean(numeric_only=True), inplace=True)

X_diabetes = diabetes.drop("Outcome", axis=1)
y_diabetes = diabetes["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

diabetes_model = LogisticRegression(max_iter=1000)
diabetes_model.fit(X_train, y_train)

diabetes_acc = accuracy_score(y_test, diabetes_model.predict(X_test))
print("🩸 Diabetes Accuracy:", diabetes_acc)

joblib.dump(diabetes_model, "models/diabetes_model.pkl")

# ============================================
# 3. HEART DISEASE PREPROCESSING (FIXED)
# ============================================

# Target conversion
heart["target"] = heart["num"].apply(lambda x: 1 if x > 0 else 0)

# Drop unused columns
heart.drop(columns=["id", "dataset", "num"], inplace=True)

# One-hot encoding
heart = pd.get_dummies(heart, drop_first=True)

# Handle missing values
heart.fillna(heart.mean(), inplace=True)

X_heart = heart.drop("target", axis=1)
y_heart = heart["target"]

# SAVE FEATURE NAMES
joblib.dump(X_heart.columns, "models/heart_features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

heart_model = LogisticRegression(max_iter=2000)
heart_model.fit(X_train, y_train)

heart_acc = accuracy_score(y_test, heart_model.predict(X_test))
print("❤️ Heart Disease Accuracy:", heart_acc)

joblib.dump(heart_model, "models/heart_model.pkl")

# ============================================
# 4. KIDNEY DISEASE PREPROCESSING (FIXED)
# ============================================

# Clean text
for col in kidney.columns:
    kidney[col] = kidney[col].astype(str).str.strip().str.lower()

# Replace missing indicators
kidney.replace(["?", "nan", "none", ""], np.nan, inplace=True)

# Convert target
kidney["classification"] = kidney["classification"].apply(
    lambda x: 1 if x == "ckd" else 0
)

# Drop ID column
kidney.drop(columns=["id"], inplace=True)

# One-hot encoding
kidney = pd.get_dummies(kidney, drop_first=True)

# Convert to numeric
kidney = kidney.apply(pd.to_numeric, errors="coerce")

# Fill missing values
kidney.fillna(kidney.mean(), inplace=True)

X_kidney = kidney.drop("classification", axis=1)
y_kidney = kidney["classification"]

# SAVE FEATURE NAMES
joblib.dump(X_kidney.columns, "models/kidney_features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_kidney, y_kidney, test_size=0.2, random_state=42
)

kidney_model = LogisticRegression(max_iter=2000)
kidney_model.fit(X_train, y_train)

kidney_acc = accuracy_score(y_test, kidney_model.predict(X_test))
print("🧪 Kidney Disease Accuracy:", kidney_acc)

joblib.dump(kidney_model, "models/kidney_model.pkl")

# ============================================
print("\n✅ ALL MODELS TRAINED & SAVED SUCCESSFULLY")

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Data" / "phone_addiction.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

# Features to use
FEATURES = [
    "Daily_Usage_Hours",
    "Sleep_Hours",
    "Anxiety_Level",
    "Academic_Performance",
    "Age",
    "Gender"
]

# Load dataset
df = pd.read_csv(DATA_PATH)

# Create target classes
df["Addiction_Level_Class"] = pd.cut(
    df["Addiction_Level"],
    bins=[-0.001, 4, 7, 10],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

df.dropna(subset=["Addiction_Level_Class"], inplace=True)

X = df[FEATURES].copy()
y = df["Addiction_Level_Class"]

# Detect numeric & categorical
numeric_features = ["Daily_Usage_Hours", "Sleep_Hours", "Anxiety_Level", "Academic_Performance", "Age"]
categorical_features = ["Gender"]

# Pipelines
numeric_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transform, numeric_features),
    ("cat", categorical_transform, categorical_features)
])

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

pipelines = {name: Pipeline([("preprocess", preprocess), ("clf", model)])
             for name, model in models.items()}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}

for name, pipe in pipelines.items():
    s = cross_val_score(pipe, X, y, scoring="f1_weighted", cv=cv, n_jobs=-1)
    scores[name] = np.mean(s)

best_model_name = max(scores, key=scores.get)
best_pipeline = pipelines[best_model_name]

# Train/test split for final report
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print("\n==== MODEL PERFORMANCE ====")
print(f"Selected model: {best_model_name}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Weighted:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")
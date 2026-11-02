import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define base directory
BASE_DIR = "/home/s2026aj05525/Downloads"

# Create model directory inside BASE_DIR
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# 1. Load Dataset from BASE_DIR
dataset_path = os.path.join(BASE_DIR, "heart.csv")
df = pd.read_csv(dataset_path)

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 2. Train-Test Split (80-20, Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Export test_data.csv for Streamlit UI upload into BASE_DIR
test_export = X_test.copy()
test_export["HeartDisease"] = y_test
test_export_path = os.path.join(BASE_DIR, "test_data.csv")
test_export.to_csv(test_export_path, index=False)
print(f"Saved test_data.csv successfully to {test_export_path}.")

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
    ]
)

# 5. Model Definitions (Using CalibratedClassifierCV for SVC to avoid the future warning)
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=42), 
        ensemble=False
    )
}

results = []

for name, clf in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Save Model using standard Python pickle (.pkl) inside model_dir
    filename = name.lower().replace(" ", "_") + ".pkl"
    file_path = os.path.join(model_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })

results_df = pd.DataFrame(results)
print("\nEvaluation Results Table:")
print(results_df.to_string(index=False))

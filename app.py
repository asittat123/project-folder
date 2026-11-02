import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Heart Disease Classifier Dashboard", layout="wide")

st.title("Heart Disease Classification & Model Evaluation")
st.markdown("BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2 Dashboard")

st.sidebar.header("User Controls")

# Feature a: CSV Upload Option
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (test_data.csv)", 
    type=["csv"]
)

# Feature b: Model Selection Dropdown (.pkl files)
model_mapping = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model = st.sidebar.selectbox("Choose Classification Model", list(model_mapping.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(test_df.head(), use_container_width=True)
    
    target_column = None
    for candidate in ["HeartDisease", "target", "class", "output"]:
        if candidate in test_df.columns:
            target_column = candidate
            break
            
    if target_column is None:
        target_column = st.sidebar.selectbox("Select Target Column", test_df.columns)
        
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].astype(int)
    
    # Load Model Pipeline with standard pickle
    model_path = model_mapping[selected_model]
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.subheader(f"Evaluation Metrics: {selected_model}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        
        c_left, c_right = st.columns([1, 1])
        
        with c_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
            
        with c_right:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            
    else:
        st.error(f"Saved model binary not found at `{model_path}`.")
else:
    st.info("Please upload `test_data.csv` via the sidebar to evaluate the models.")

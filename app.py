import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load all trained models
@st.cache_resource
def load_models():
    models = {}
    model_names = [
        "Logistic_Regression", "Decision_Tree", "KNN", 
        "Naive_Bayes", "Random_Forest", "XGBoost"
    ]
    for name in model_names:
        models[name] = joblib.load(f"model/{name}.pkl")
    return models

models = load_models()

st.title("Wine Quality Classifier")
st.markdown("Upload test CSV (must have same 12 features + optional 'good_quality' column)")

# File upload
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Features (drop target if present)
    feature_cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "sulfur_ratio"
    ]
    
    missing_cols = [col for col in feature_cols if col not in test_df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}. Expected: {feature_cols}")
    else:
        X_test = test_df[feature_cols]
    
    # Model selection
    model_name = st.selectbox("Select model:", list(models.keys()))
    model = models[model_name]
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    st.subheader(f"Predictions ({model_name})")
    st.dataframe(pd.DataFrame({
        "Probability_good": np.round(y_proba, 3),
        "Prediction": y_pred
    }))
    
    # Metrics if target column exists
    if "good_quality" in test_df.columns:
        y_true = test_df["good_quality"]
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": 0.0  # Simplified; use roc_auc_score(y_true, y_proba) for full
        }
        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    
    else:
        st.info("Add 'good_quality' column to CSV for metrics + confusion matrix")


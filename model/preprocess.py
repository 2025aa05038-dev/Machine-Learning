import pandas as pd
import os

data_dir = "data/"

raw_path = os.path.join(data_dir, "winequality-red.csv")
df = pd.read_csv(raw_path, sep=";")

df.head(), df.shape

df["good_quality"] = (df["quality"] >= 7).astype(int)
df["good_quality"].value_counts(normalize=True)

df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)

feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "sulfur_ratio"
]
target_col = "good_quality"

df_processed = df[feature_cols + [target_col]]

df_processed.head(), df_processed.shape

processed_path = "winequality_red_processed.csv"
df_processed.to_csv(processed_path, index=False)
processed_path

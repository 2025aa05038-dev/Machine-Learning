# Wine Quality Classification Deployment

## Problem Statement
Predict whether red wine is "good quality" (quality ≥ 7) or "not good" using 12 physicochemical features. Binary classification problem deployed as interactive web app.

## Dataset Description
- **Source**: UCI Wine Quality (Red) dataset [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)[web:17]
- **Instances**: 1,599 red wine samples
- **Features**: 12 numeric features (11 original physicochemical properties + 1 engineered `sulfur_ratio`)
- **Target**: `good_quality` (1 = good ≥7, 0 = not good)
- **Feature list**: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, sulfur_ratio

## Models Performance [6 marks]

| ML Model | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.8906 | 0.898 | 0.6818 | 0.3488 | 0.4615 | 0.4361 |
| Decision Tree | 0.8906 | 0.7895 | 0.5833 | 0.6512 | 0.6154 | 0.5530 |
| kNN | 0.8844 | 0.8241 | 0.6000 | 0.4186 | 0.4932 | 0.4391 |
| Naive Bayes | 0.8500 | 0.8556 | 0.4615 | 0.6977 | 0.5556 | 0.4843 |
| Random Forest (Ensemble) | **0.9312** | **0.9405** | **0.9200** | 0.5349 | **0.6765** | **0.6706** |
| XGBoost (Ensemble) | **0.9438** | 0.9312 | 0.8378 | **0.7209** | **0.7750** | **0.7458** |

## Model Observations [3 marks]

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Strong baseline with good precision but lower recall due to class imbalance |
| Decision Tree | Matches Logistic accuracy but worse AUC; prone to overfitting on small dataset |
| kNN | Solid performance but sensitive to scaling and distance metrics |
| Naive Bayes | Best recall but sacrifices precision; independence assumption helps with imbalance |
| **Random Forest (Ensemble)** | Excellent overall performance; bagging reduces variance significantly |
| **XGBoost (Ensemble)** | **Best model**: highest accuracy + MCC + F1; gradient boosting excels on tabular data |

**Live Demo**: [Your Streamlit URL here]
**Dataset**: UCI Wine Quality (Red) [web:17]
**Trained on**: BITS Pilani Virtual Lab

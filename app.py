import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- PART 1: LOAD THE DATA ---
data = pd.read_csv('kidney_disease.csv')
data = data.drop('id', axis=1, errors='ignore')
data = data.dropna()

target_col = data.columns[-1]
data[target_col] = data[target_col].astype(str).str.replace('\t', '').str.strip()

# --- PART 2: PREPARE THE DATA ---
encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = encoder.fit_transform(data[column])

X = data.drop(target_col, axis=1) 
y = data[target_col]              

# THE FIX: Force all clues to be standard decimals (floats) and answers to be integers
X = X.astype(float)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data prepped! Training the AI brain now...")

# --- PART 3: TRAIN AI & GENERATE GRAPH ---
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

print("Training complete! Generating the Explainable AI graph...")

# THE FIX: Use TreeExplainer, which is specifically designed for XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
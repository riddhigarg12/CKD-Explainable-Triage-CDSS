import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="CKD Triage CDSS", layout="wide")
st.title("🏥 Explainable AI: Chronic Kidney Disease Triage")
st.markdown("A Clinical Decision Support System analyzing routine EHR data.")

# --- 2. LOAD & TRAIN MODEL ---
@st.cache_resource
def build_model():
    data = pd.read_csv('kidney_dataset.csv')
    data = data.drop('id', axis=1, errors='ignore')
    data = data.dropna()
    
    target_col = data.columns[-1]
    data[target_col] = data[target_col].astype(str).str.replace('\t', '').str.strip()
    
    encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = encoder.fit_transform(data[column])
            
    X = data.drop(target_col, axis=1).astype(float)
    y = data[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return model, explainer, shap_values, X_test, y_test

model, explainer, shap_values, X_test, y_test = build_model()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Dashboard Mode:", ["Browse Patient Cohort", "Manual Clinical Entry"])

# --- 4. MODE 1: BROWSE PATIENT COHORT ---
if app_mode == "Browse Patient Cohort":
    st.sidebar.header("Patient Triage Selector")
    st.sidebar.write("Select a patient from the test cohort to analyze their risk.")
    
    patient_index = st.sidebar.slider("Select Patient ID", 0, len(X_test)-1, 0)
    patient_data = X_test.iloc[[patient_index]]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Vitals (Cohort Data)")
        st.dataframe(patient_data.T, use_container_width=True)
        
        prediction = model.predict(patient_data)[0]
        risk_prob = model.predict_proba(patient_data)[0][1] * 100
        
        st.subheader("AI Triage Alert")
        if prediction == 1:
            st.error(f"⚠ **HIGH RISK DETECTED** ({risk_prob:.1f}% Confidence)")
        else:
            st.success(f"✅ **NO RISK DETECTED** ({100 - risk_prob:.1f}% Confidence)")

    with col2:
        st.subheader("Population Model Explainability")
        st.write("This SHAP graph explains which vitals drive the AI's decision across the general patient cohort.")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)

# --- 5. MODE 2: MANUAL CLINICAL ENTRY ---
elif app_mode == "Manual Clinical Entry":
    st.sidebar.header("Manual Patient Entry")
    
    # Enforce strict biological limits
    patient_sex = st.sidebar.radio("Patient Sex (Sets Clinical Limits)", ["Male", "Female"])
    if patient_sex == "Male":
        creat_min, creat_max, creat_def = 0.7, 1.3, 1.0 #
    else:
        creat_min, creat_max, creat_def = 0.6, 1.1, 0.8 #
    
    with st.sidebar.form("patient_form"):
        st.write("Enter patient lab results:")
        user_data = {}
        
        feature_limits = {
            'Creatinine': {'min': creat_min, 'max': creat_max, 'val': creat_def, 'unit': 'mg/dL', 'step': 0.1},
            'BUN': {'min': 7.0, 'max': 120.0, 'val': 30.0, 'unit': 'mg/dL', 'step': 1.0},
            'GFR': {'min': 5.0, 'max': 120.0, 'val': 73.0, 'unit': 'mL/min/1.73m²', 'step': 1.0},
            'Urine_Output': {'min': 400.0, 'max': 2500.0, 'val': 1600.0, 'unit': 'mL/day', 'step': 50.0},
            'Age': {'min': 18.0, 'max': 90.0, 'val': 50.0, 'unit': 'Years', 'step': 1.0},
            'Protein_in_Urine': {'min': 0.0, 'max': 3000.0, 'val': 500.0, 'unit': 'mg/dL', 'step': 10.0},
            'Water_Intake': {'min': 1.0, 'max': 4.0, 'val': 2.5, 'unit': 'Liters/day', 'step': 0.1}
        }

        for col in X_test.columns:
            if col in feature_limits:
                limits = feature_limits[col]
                user_data[col] = st.number_input(
                    f"{col} ({limits['unit']})", 
                    min_value=float(limits['min']), 
                    max_value=float(limits['max']), 
                    value=float(limits['val']), 
                    step=float(limits['step'])
                )
            elif col in ['Diabetes', 'Hypertension']:
                user_data[col] = st.selectbox(f"{col} (0=No, 1=Yes)", options=[0.0, 1.0])
            elif col == 'Medication':
                user_data[col] = st.selectbox(f"{col} (0=None, 1=ACE, 2=ARB, 3=Diuretic)", options=[0.0, 1.0, 2.0, 3.0])
            else:
                mean_val = float(X_test[col].mean())
                user_data[col] = st.number_input(f"{col}", value=mean_val)
        
        submitted = st.form_submit_button("Analyze Risk")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Vitals (Manual Input)")
        patient_df = pd.DataFrame([user_data])
        st.dataframe(patient_df.T, use_container_width=True)
        
        if submitted:
            patient_df = patient_df[X_test.columns]
            prediction = model.predict(patient_df)[0]
            risk_prob = model.predict_proba(patient_df)[0][1] * 100
            
            st.subheader("AI Triage Alert")
            if prediction == 1:
                st.error(f"⚠ **HIGH RISK DETECTED** ({risk_prob:.1f}% Confidence)")
            else:
                st.success(f"✅ **NO RISK DETECTED** ({100 - risk_prob:.1f}% Confidence)")
        else:
            st.info("👈 Enter patient vitals in the sidebar and click 'Analyze Risk'.")

    with col2:
        st.subheader("Population Model Explainability")
        st.write("This SHAP graph explains which vitals drive the AI's decision across the general patient population.")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
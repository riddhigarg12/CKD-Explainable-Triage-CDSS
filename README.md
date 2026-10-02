# 🏥 Explainable AI in Nephrology: CKD Triage CDSS

**Author:** Riddhi Garg  
**Institution:** VIT Bhopal University  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)

## 📌 Project Overview
The lack of explainability in artificial intelligence remains a primary barrier to clinical adoption. This project is a **Clinical Decision Support System (CDSS)** engineered for the early triage of Chronic Kidney Disease (CKD). Moving beyond standard "black box" algorithms, this system integrates advanced predictive modeling (XGBoost) with a fully interactive web dashboard (Streamlit) designed to build clinical trust through mathematical transparency (SHAP).

## 🚀 Key Features & Innovations

| Feature | Description | Clinical Impact |
| :--- | :--- | :--- |
| **Explainable Triage (XAI)** | Integrates `shap.TreeExplainer` to map the exact mathematical weight of every clinical feature used by the model. | Generates live visual plots alongside every prediction, proving exactly *why* a patient was flagged. |
| **Dynamic Cohort Browser** | A slider-based interface to analyze patients from a 20% holdout test dataset. | Evaluates how the AI handles a diverse array of existing patient profiles on unseen data. |
| **Sex-Specific Bounding** | Manual clinical entry dynamically shifts acceptable input thresholds (e.g., Creatinine) based on biological sex. | Mimics true clinical environments and tests AI robustness against real-world inputs. |
| **Dataset Agnostic Engine** | Automatically detects prediction columns and cleans messy categorical data dynamically. | Prevents standard key errors and allows rapid swapping of updated EHR datasets. |

## 🛠️ Technology Stack
*   **Frontend Interface:** Streamlit (Python-based interactive dashboard)
*   **Predictive Engine:** XGBoost Classifier (Optimized for non-linear biological data)
*   **Explainability (XAI):** SHAP (SHapley Additive exPlanations)
*   **Data Processing:** Pandas, Scikit-Learn (LabelEncoding, Train-Test Split)
*   **Visualization:** Matplotlib

## 📊 System Architecture

1.  **Data Ingestion:** Loads the Kidney Function Health Dataset, dropping nulls and strictly casting input features to raw decimals to bypass XAI categorical bugs.
2.  **Training Phase:** 80% split is fed to the XGBoost Classifier to map complex physiological deterioration patterns.
3.  **Inference & UI:** Clinicians input patient vitals (or select from the test cohort).
4.  **Triage Output:** The CDSS outputs a real-time risk alert (High Risk / No Risk) paired with a live SHAP summary plot highlighting microscopic data trends.

## 💻 Installation & Usage

To run this CDSS on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/riddhigarg12/CKD-Explainable-Triage-CDSS.git](https://github.com/riddhigarg12/CKD-Explainable-Triage-CDSS.git)
cd CKD-Explainable-Triage-CDSS

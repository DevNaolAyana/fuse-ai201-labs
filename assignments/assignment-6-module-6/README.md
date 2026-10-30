# Detecting Fraudulent Credit Card Transactions Using Machine Learning
**YEEAP AI-201 | Module 5: AI Project Implementation | Group 16**

## Project Overview
This project delivers a complete, working machine-learning prototype designed to detect fraudulent credit-card transactions. Following the approved Module 4 project proposal, it develops and compares a **weighted Logistic Regression baseline** with an advanced **XGBoost Classifier** trained with **SMOTE class balancing** and **StandardScaler** feature scaling.

The prototype is designed for decision-support: it flags high-risk transactions with probability scores to prioritize human investigator review, mitigating financial losses while preserving customer trust.

---

## Submission Contents
- `Group16_Fraud_Detection_Project_Report.docx` — Written implementation report covering all 8 stages.
- `Group16_Fraud_Detection_Implementation.ipynb` — Fully executed Jupyter notebook with code, outputs, and visualizations across all 8 stages.
- `app/` — Working Streamlit interactive web application (`app/app.py`).
- `models/` — Serialized model binaries:
  - `fraud_detector_model.pkl` — Trained XGBoost classifier.
  - `feature_scaler.pkl` — Fitted StandardScaler for `Time` and `Amount`.
- `data/` — Dataset folder containing `creditcard.csv` and source documentation.
- `requirements.txt` — Python package dependencies.
- `supporting_files/` — Final submission checklist and documentation guidelines.

---

## Dataset Description
- **Source:** Credit Card Fraud Detection dataset (Worldline & Machine Learning Group at ULB, Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Volume:** 284,807 transactions recorded over two days, with 492 fraud cases (0.172% prevalence).
- **Cleaned Data:** 1,081 duplicate transactions removed, yielding 283,726 unique transactions (473 fraud cases).
- **Features:** 30 numerical input features:
  - `Time`: Elapsed seconds since the initial transaction.
  - `V1`–`V28`: Anonymized principal component analysis (PCA) features.
  - `Amount`: Transaction amount in USD.
- **Target:** `Class` (`0` = Legitimate, `1` = Fraudulent).

---

## Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd assignments/assignment-5-module-5
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Running the Application & Notebook

### To run the app:
Launch the interactive Streamlit browser prototype:
```bash
streamlit run app/app.py
```
- Open your browser at the displayed URL (typically `http://localhost:8501`).
- Adjust the **Transaction Amount** and **Time** using the inputs/sliders.
- Test quick presets from the sidebar ("Sample Legitimate Transaction" or "Sample Known Fraudulent Transaction") or enter custom PCA values under the expander.
- Click **"Predict Transaction Status"** to view real-time classification, fraud probability score, and recommended analyst action.
- Use the **Batch Transaction Evaluation** section to upload a CSV file and download annotated risk predictions.

### To view and run the notebook:
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook Group16_Fraud_Detection_Implementation.ipynb
```
All 17 cells are pre-executed with complete outputs and embedded visualizations. To re-run from scratch, select **Kernel -> Restart Kernel and Run All Cells**.

---

## Held-Out Test Results (80/20 Stratified Split)

The final test dataset comprises **56,746 transactions**, including **95 true fraud cases**:

| Metric | Proposal Target | Logistic Regression (Baseline) | XGBoost (Final Improved) | Outcome Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Fraud)** | $\ge$ 0.80 | 0.8737 | **0.8421** | **Target Met: 84.21%** (Catches 80 of 95 frauds) |
| **Precision (Fraud)** | $\ge$ 0.50 | 0.0530 | **0.2676** | **5x Improvement** over baseline |
| **F1-Score (Fraud)** | $\ge$ 0.60 | 0.1000 | **0.4061** | **4x Improvement** over baseline |
| **PR-AUC** | > Baseline | 0.6769 | **0.7870** | **Target Met: +0.1102** gain |
| **ROC-AUC** | High (> 0.95) | 0.9619 | **0.9751** | **Target Met: 0.9751** |
| **True Positives (TP)** | Maximize | 83 | **80** | 80 fraudulent transactions caught |
| **False Positives (FP)**| Minimize | 1,482 | **219** | **Reduced false alarms by 1,263** (85.2% reduction) |
| **False Negatives (FN)**| Minimize | 12 | **15** | Only 15 fraud cases missed |

---

## Responsible AI & Limitations
1. **Decision Support Prototype:** The model produces an uncalibrated ranking risk score. It is designed to assist human fraud analysts, not to automatically cancel cards or decline payments.
2. **Concept Drift:** Payment behaviors and fraud vectors shift dynamically over time; real-world production models require continuous monitoring, threshold tuning, and periodic retraining.
3. **Privacy:** The public dataset anonymizes customer attributes into PCA components (`V1`–`V28`). Commercial deployments must strictly observe PII privacy regulations (e.g. GDPR, local banking laws).

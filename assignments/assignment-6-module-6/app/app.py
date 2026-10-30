import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Set Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector | Group 16",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve model and scaler paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detector_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "feature_scaler.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/fraud_detector_model.pkl")
    SCALER_PATH = Path("models/feature_scaler.pkl")

# Cached resource loaders
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, err = load_artifacts()

# App Header
st.title("🛡️ Credit Card Fraud Detection Prototype")
st.markdown("""
**YEEAP AI-201 | Module 5: AI Project Implementation | Group 16**

This interactive prototype utilizes an optimized **XGBoost Classifier** trained with **SMOTE class balancing** and **StandardScaler** to detect potentially fraudulent transactions.
""")

st.divider()

if err:
    st.error(f"❌ Error loading model artifacts: {err}. Please ensure model files exist in the `models/` directory.")
    st.stop()

# Preset samples for demonstration and testing
PRESET_LEGITIMATE = {
    'Time': 3600.0,
    'Amount': 45.50,
    'V1': -0.42, 'V2': 0.12, 'V3': 1.25, 'V4': -0.31, 'V5': 0.20,
    'V6': -0.15, 'V7': 0.35, 'V8': 0.05, 'V9': 0.18, 'V10': -0.10,
    'V11': -0.45, 'V12': 0.25, 'V13': 0.10, 'V14': 0.30, 'V15': 0.50,
    'V16': 0.12, 'V17': -0.10, 'V18': 0.05, 'V19': -0.05, 'V20': 0.02,
    'V21': -0.08, 'V22': -0.15, 'V23': 0.04, 'V24': 0.10, 'V25': -0.20,
    'V26': 0.15, 'V27': 0.01, 'V28': 0.02
}

PRESET_FRAUDULENT = {
    'Time': 406.0,
    'Amount': 239.93,
    'V1': -2.31, 'V2': 1.95, 'V3': -1.61, 'V4': 3.99, 'V5': -0.52,
    'V6': -1.43, 'V7': -2.54, 'V8': 1.39, 'V9': -2.77, 'V10': -2.77,
    'V11': 3.20, 'V12': -2.90, 'V13': -0.60, 'V14': -4.29, 'V15': 0.39,
    'V16': -1.14, 'V17': -2.83, 'V18': -0.02, 'V19': 0.42, 'V20': 0.13,
    'V21': 0.52, 'V22': -0.04, 'V23': -0.47, 'V24': 0.32, 'V25': 0.04,
    'V26': 0.18, 'V27': 0.26, 'V28': -0.14
}

# Sidebar presets
st.sidebar.header("🎯 Quick Test Presets")
preset_choice = st.sidebar.radio(
    "Load sample transaction profile:",
    ("Custom Inputs", "Sample Legitimate Transaction", "Sample Known Fraudulent Transaction")
)

active_preset = {}
if preset_choice == "Sample Legitimate Transaction":
    active_preset = PRESET_LEGITIMATE
elif preset_choice == "Sample Known Fraudulent Transaction":
    active_preset = PRESET_FRAUDULENT

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Metrics on Held-out Test Set:**
- **Recall (Fraud):** 84.21%
- **PR-AUC:** 0.7870
- **ROC-AUC:** 0.9751
- **Baseline:** Logistic Regression
- **Algorithm:** XGBoost (100 Trees, Depth 5)
""")

# Main form layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💳 Primary Transaction Inputs")
    
    default_amount = float(active_preset.get('Amount', 100.00))
    default_time = float(active_preset.get('Time', 7200.0))

    amount_input = st.number_input(
        "Transaction Amount ($):",
        min_value=0.0,
        max_value=100000.0,
        value=default_amount,
        step=1.0,
        format="%.2f",
        help="Monetary transaction amount in US Dollars."
    )
    
    amount_slider = st.slider(
        "Amount slider quick adjustment:",
        min_value=0.0,
        max_value=2000.0,
        value=min(amount_input, 2000.0),
        step=5.0
    )
    # Sync with slider if changed
    if amount_slider != min(amount_input, 2000.0):
        amount_input = amount_slider

    time_input = st.number_input(
        "Transaction Time (seconds from first transaction):",
        min_value=0.0,
        max_value=200000.0,
        value=default_time,
        step=60.0,
        help="Elapsed seconds recorded in the payment stream."
    )

with col2:
    st.subheader("🔬 PCA Latent Features (V1 – V28)")
    st.markdown("Features V1–V28 represent anonymized PCA principal components.")
    
    with st.expander("Expand to view/edit PCA Features V1–V28", expanded=False):
        pca_inputs = {}
        pca_cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            feat_name = f"V{i}"
            def_val = float(active_preset.get(feat_name, 0.0))
            with pca_cols[col_idx]:
                pca_inputs[feat_name] = st.number_input(
                    feat_name,
                    value=def_val,
                    step=0.1,
                    format="%.2f",
                    key=f"pca_{feat_name}"
                )

st.divider()

# Predict Button
predict_btn = st.button("🔍 Predict Transaction Status", type="primary", use_container_width=True)

if predict_btn:
    # 1. Scale Time and Amount using fitted StandardScaler
    time_amt_df = pd.DataFrame([[time_input, amount_input]], columns=['Time', 'Amount'])
    time_amt_scaled = scaler.transform(time_amt_df)
    scaled_time = time_amt_scaled[0, 0]
    scaled_amount = time_amt_scaled[0, 1]

    # 2. Assemble 30-feature vector in exact training order
    feature_dict = {'Time': scaled_time}
    for i in range(1, 29):
        feat_name = f"V{i}"
        val = pca_inputs.get(feat_name, active_preset.get(feat_name, 0.0))
        feature_dict[feat_name] = val
    feature_dict['Amount'] = scaled_amount

    input_df = pd.DataFrame([feature_dict])

    # 3. Model Inference
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    fraud_prob = float(probabilities[1])
    legit_prob = float(probabilities[0])

    st.subheader("📊 Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ Result: Fraudulent")
            st.markdown(f"**Fraud Probability:** `{fraud_prob * 100:.2f}%`")
            st.warning("**Recommended Action:** Flag for high-priority investigation by fraud review analyst.")
        else:
            st.success("### ✅ Result: Legitimate")
            st.markdown(f"**Legitimate Confidence:** `{legit_prob * 100:.2f}%` (Fraud Probability: `{fraud_prob * 100:.2f}%`)")
            st.info("**Recommended Action:** Standard processing approved. Transaction within normal risk parameters.")

    with res_col2:
        st.markdown("**Risk Score Meter:**")
        st.progress(fraud_prob)
        st.caption(f"Calculated Fraud Risk Score: {fraud_prob:.4f} / 1.0000")

    # Display Input Summary
    with st.expander("Inspect Processed Model Input Vector", expanded=False):
        st.dataframe(input_df.style.format("{:.4f}"))

st.divider()

# Batch CSV Upload Section
st.subheader("📂 Batch Transaction Evaluation (CSV Upload)")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing transactions (must contain columns: Time, V1-V28, Amount)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded batch containing {len(batch_df):,} transactions.")
        
        req_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [c for c in req_cols if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            # Process batch
            batch_features = batch_df[req_cols].copy()
            batch_features[['Time', 'Amount']] = scaler.transform(batch_features[['Time', 'Amount']])
            
            batch_preds = model.predict(batch_features)
            batch_probs = model.predict_proba(batch_features)[:, 1]
            
            results_df = batch_df.copy()
            results_df['Fraud_Risk_Score'] = np.round(batch_probs, 4)
            results_df['Prediction'] = np.where(batch_preds == 1, 'Fraudulent', 'Legitimate')
            
            fraud_count = int((batch_preds == 1).sum())
            st.info(f"Analysis complete: Flagged **{fraud_count}** out of **{len(results_df)}** transactions as potentially fraudulent.")
            
            st.dataframe(results_df[['Time', 'Amount', 'Prediction', 'Fraud_Risk_Score']].head(20))
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Annotated Predictions CSV",
                data=csv_data,
                file_name="fraud_predictions_results.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")

st.markdown("---")
st.markdown("""
<small style='color: gray;'>
<strong>Responsible AI & Decision-Support Notice:</strong> This application is a decision-support prototype. 
Predictions and risk scores are intended to prioritize transactions for human analyst review and must not be used to automatically decline, cancel, or block transactions without human verification.
</small>
""", unsafe_allow_html=True)

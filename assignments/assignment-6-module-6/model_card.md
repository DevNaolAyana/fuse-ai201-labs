# Model Card: Credit Card Fraud Detection Classifier (XGBoost)

## 1. Model Overview
- **Model Name:** Credit Card Fraud Detection Classifier
- **Model Version:** 1.0.0 (Module 6 Deployment Edition)
- **Model Type:** Extreme Gradient Boosting (`XGBClassifier`) with SMOTE class balancing
- **Release Date:** October 2026
- **Developer / Team:** Group 16 — YEEAP AI-201 Applied Machine Learning
- **Primary Objective:** Provide automated, low-latency, high-precision decision support for identifying fraudulent credit card transactions in stream-oriented card processing pipelines.

---

## 2. Intended Use
### Primary Use Cases
- **Tier-1 Real-Time Risk Scoring:** Generating probabilistic fraud risk scores ($[0.0, 1.0]$) for credit card transactions during authorization.
- **Investigator Queue Prioritization:** Triaging flagged transactions into human analyst review queues based on predicted risk.
- **Dynamic Step-Up Authentication:** Triggering two-factor authentication (3D Secure, OTP challenge) for medium-to-high risk transactions rather than immediate hard declines.

### Out-of-Scope & Prohibited Uses
- **Autonomous Account Freezing / Card Cancellation:** The model is not authorized to unilaterally cancel cardholder accounts or decline credit lines without human-in-the-loop review.
- **Unadapted Payment Rails:** Applying the model to non-card payment ecosystems (e.g., automated clearing house [ACH], wire transfers, peer-to-peer crypto transactions) without domain adaptation.
- **Credit Scoring / Underwriting:** The model must never be repurposed to evaluate creditworthiness, loan qualification, or interest rates.

---

## 3. Training Data
- **Dataset Source:** European Cardholders Credit Card Fraud Dataset (Worldline & Machine Learning Group at ULB, distributed via Kaggle).
- **Timeframe:** 2 days of real-world European cardholder transaction activity (September 2013).
- **Data Volume:** 
  - Raw records: 284,807 transactions.
  - Cleaned unique records: 283,726 transactions (1,081 duplicate rows identified and removed).
  - Fraud prevalence: 473 fraud cases (0.167% minority class prevalence).
- **Input Features (30 total):**
  - `Time`: Elapsed seconds between the transaction and the initial transaction in the dataset.
  - `V1`–`V28`: 28 orthogonal continuous features resulting from a Principal Component Analysis (PCA) transformation applied for confidentiality.
  - `Amount`: Monetary transaction amount in USD/EUR.
- **Target Label:** `Class` (0 = Legitimate transaction, 1 = Fraudulent transaction).
- **Data Preprocessing & Split Strategy:**
  - **Stratified Partition:** 80/20 train/test split preserving class imbalance ratios (Train: 226,980 records; Test: 56,746 records).
  - **Feature Standardization:** `StandardScaler` fitted exclusively on training `Time` and `Amount` to prevent data leakage, then transformed across both train and test splits.
  - **Class Balancing:** Synthetic Minority Over-sampling Technique (SMOTE, `k_neighbors=5`, `random_state=42`) applied strictly to the training fold, balancing fraud cases to a 1:1 synthetic ratio.

---

## 4. Evaluation Results

Performance was evaluated on the independent held-out test set comprising **56,746 transactions**, including **95 true fraud cases**:

### Key Performance Metrics
| Metric | Baseline (Logistic Regression) | Final Model (XGBoost) | Production Benchmark |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Primary Metric)** | 0.6769 | **0.7870** | High Performance (+0.1101 gain) |
| **Recall (Fraud Detection Rate)** | 87.37% | **84.21%** | Target Met (> 80.0%) |
| **False Positive Rate (FPR)** | 2.62% | **0.39%** | **Sub-1% Operational Standard** |
| **Precision** | 5.30% | **26.76%** | **5.0x Improvement** |
| **F1-Score** | 0.1000 | **0.4061** | **4.0x Improvement** |
| **ROC-AUC** | 0.9619 | **0.9751** | Strong Discrimination |

### Confusion Matrix on Held-Out Test Set (56,746 Transactions)
- **True Positives (TP):** **80** (Fraudulent transactions correctly intercepted)
- **False Positives (FP):** **219** (Legitimate transactions flagged for review)
- **True Negatives (TN):** **56,432** (Legitimate transactions approved smoothly)
- **False Negatives (FN):** **15** (Fraud cases missed by classifier)

### Operational Impact Summary
By moving from the Logistic Regression baseline to the tuned XGBoost model, **false alarms dropped from 1,482 to 219** (an **85.2% reduction in false positives**), while maintaining an **84.21% fraud capture rate** and achieving a False Positive Rate of **0.39%**. This directly protects operational investigation bandwidth and drastically reduces customer friction.

---

## 5. Limitations and Known Failure Modes
1. **Concept Drift & Emerging Fraud Typologies:**
   The training data captures a snapshot of payment fraud from 2013. Attack vectors evolve (e.g., synthetic identity fraud, new online checkout exploits); continuous drift monitoring (PSI / KS drift detection) and retraining are required.
2. **PCA Interpretability Constraints:**
   Because `V1`–`V28` are anonymized linear projections, domain experts cannot directly trace flags to intuitive retail features like Merchant Category Code (MCC), geographic velocity, or device fingerprinting without proprietary de-anonymization mappings.
3. **Low-Value Stealth Transactions (False Negatives):**
   15 fraud cases were missed in the test split. Analysis indicates these transactions closely emulate everyday cardholder micro-purchases and lack extreme PCA deviations.
4. **Extreme Amount Outliers (False Positives):**
   Occasional high-value legitimate purchases ($> \$2,000$) may generate elevated risk scores due to standard deviation skew, requiring secondary contextual checks.

---

## 6. Ethical Considerations & Responsible AI
- **Fairness & Non-Discrimination:**
  The dataset excludes demographic identifiers (e.g., gender, ethnicity, age, zip code). Features are strictly transactional and mathematically orthogonalized, mitigating direct proxy discrimination.
- **Mitigating Customer Harm:**
  False positives can cause embarrassment, card blocks, or disruption for legitimate customers. To uphold customer trust, high-risk scores must prompt non-intrusive step-up verification (e.g. mobile push notification / SMS challenge) rather than hard service termination.
- **Transparency & Accountability:**
  The model is deployed as a transparent REST API with traceable inputs, reproducible preprocessing pipelines, version-controlled artifacts, and MLflow experiment logging.
- **Data Protection & Compliance:**
  In commercial deployment, raw transaction feeds must comply with PCI-DSS, GDPR, and relevant banking secrecy statutes. All model communications must use TLS encryption with strict authentication controls.

---

## 7. Model Details (XGBoost)
- **Framework:** XGBoost (`xgboost>=1.7.0`)
- **Estimator Class:** `XGBClassifier`
- **Objective Function:** `binary:logistic`
- **Evaluation Metric:** `aucpr` (Area under Precision-Recall curve)
- **Hyperparameters:**
  - `n_estimators`: `100`
  - `max_depth`: `5`
  - `learning_rate`: `0.1`
  - `subsample`: `1.0`
  - `colsample_bytree`: `1.0`
  - `random_state`: `42`
  - `tree_method`: `auto`
- **Input Dimensions:** 30 floating-point features (`Time`, `V1`–`V28`, `Amount`).
- **Output:** Continuous probability $P(\text{Class}=1 \mid X) \in [0.0, 1.0]$ and binary class indicator.
- **Inference Latency:** $< 5\text{ms}$ per single transaction vector.

# Module 6.9: Governance Plan & Audit Trail

## 1. Subgroup Evaluation

- **Overall Performance:** XGBoost model achieved **84.21% Recall** (catching 80 out of 95 frauds).
- **Subgroup Analysis (Transaction Amount):**
  - **Amount < $100:** Recall is ~82%. (Plenty of data available).
  - **Amount > $1000:** Recall drops to ~65%. (Fraudsters often test small amounts first, making high-value fraud harder to catch).
- **Limitations Note:** We could not analyze performance by geographical region or customer gender because the Kaggle ULB dataset is entirely anonymized via PCA (V1-V28), meaning no demographic data is available.

## 2. Human Oversight

- **Decision Type:** The model is a _decision-support tool_, not an automated enforcement tool.
- **Approval:** A human fraud analyst must review any transaction flagged by the model before the customer's account is frozen.
- **High-Stakes Review:** Any flagged transaction over $5,000 requires approval from a Senior Analyst before action is taken.

## 3. Explanation on Request

- If a customer disputes a frozen transaction, the Fraud Analyst provides a plain-language explanation using SHAP (e.g., _"The transaction was flagged because its spending pattern (V14) deviated significantly from your normal historical baseline"_).
- Provided within 2 business days.

## 4. Bias Monitoring

- **Frequency:** Quarterly.
- **Trigger:** If Recall for a specific subgroup (e.g., low-value vs high-value transactions) drops more than 10 percentage points below the overall average, the ML Engineer must initiate a review of the training data and retrain.

## 5. Incident Response & Rollback

- **Notification:** Anyone who sees harmful behaviour (e.g., sudden spike in false positives) notifies the ML Engineer via the Slack/PagerDuty channel.
- **Authority:** The ML Engineer has authority to switch off the service.
- **Rollback:** The ML Engineer moves the MLflow `champion` alias back to the previous version and restarts the FastAPI service. Affected users are emailed within 24 hours, and an incident report is filed within 3 days.

## 6. Model Retirement

- A model version is retired when a replacement model has been in service for 30 days without incident.
- Users receive 2 weeks' notice, and the old model artifact + model card are archived in the MLflow registry for auditing.

## 7. Audit Trail

Here is how a prediction is traced end-to-end:

1. The prediction log (`predictions.db`) records the `request_id`, `timestamp`, and `model_version` (e.g., v1.0.0).
2. Using the `model_version`, we open the MLflow registry and find the linked **Run ID**.
3. The Run ID links to the exact **Git commit** (code) and the dataset version (Kaggle ULB) used for training.
4. This allows us to determine exactly why a decision was made and verify the model was used within its intended scope.

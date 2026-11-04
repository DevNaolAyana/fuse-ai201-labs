# Module 6.8: Retraining Workflow & Policy

## 1. Trigger Strategy

We use a **Hybrid Trigger Strategy**:

- **Scheduled:** A standard retrain every 3 months (quarterly).
- **Drift-Based:** Trigger an immediate retrain if the 6.7 Drift Detection script flags a KS Statistic p-value < 0.05.
- **Performance-Based:** Trigger a retrain if production PR-AUC falls below 0.75.

## 2. Acceptance Rules & Promotion Gates

Before a candidate model can even be considered for release, it must pass three automatic gates evaluated on the _same validation dataset_ as the current champion:

1.  **Absolute Floor:** The candidate must achieve an AUC of at least **0.75**.
2.  **Regression Limit:** The candidate may not be worse than the current champion by more than **0.02** (AUC).
3.  **Critical Class Gate:** Because missing a fraud is very costly, the candidate must maintain a Fraud Recall of at least **0.80**.

## 3. Human Approval

Automation does not mean automatic deployment. If a candidate passes the three gates, it is moved to "Staging" and marked as _APPROVED for Staging_. A named human (the ML Engineer) must then manually review the MLflow logs and approve promotion to Production.

## 4. Rollback Plan

- **Fallback:** The previous production model (`fraud_detector_model_v1.pkl`) is kept in the MLflow registry.
- **Trigger:** If the new model produces >5% false positives in the first 24 hours, or if latency doubles.
- **Procedure:** Change the environment variable `MODEL_VERSION` back to v1 and restart the FastAPI service. Time to rollback: < 2 minutes.

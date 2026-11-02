# Module 6.6: Monitoring Policy, Privacy, & Interpretation

## 1. Privacy & Data Statement

- **What we log:** Request ID, timestamp, input features (Time, Amount, PCA), fraud probability, decision, model version, and latency.
- **What we DO NOT log:** Personal Identifiable Information (PII). The Kaggle dataset uses PCA-anonymized features, so no names or card numbers are ever logged.
- **Protection:** The `logs/` directory is added to `.gitignore` and `.dockerignore`. Real production data is never committed to GitHub or baked into the Docker image.
- **Simulated Data:** The generated history is explicitly marked with `is_simulated = 1` in the database. The Streamlit dashboard also displays a prominent warning banner: _"This dashboard contains SIMULATED data for demonstration purposes."_

## 2. Alert Rules

| Indicator   | Condition                          | Level  | Owner          | Action                                                     |
| :---------- | :--------------------------------- | :----- | :------------- | :--------------------------------------------------------- |
| Error Rate  | > 2% of requests fail (status 500) | High   | Developer      | Check API logs; rollback to previous version if necessary. |
| Latency     | Avg latency > 70ms (2x baseline)   | Medium | Developer      | Check server load or network issues.                       |
| Fraud Rate  | Shifts by > 10% in 7 days          | Medium | Data Scientist | Investigate if fraudsters changed tactics (concept drift). |
| Logging Gap | No records for 24 hours            | Info   | Developer      | Check if the FastAPI service crashed.                      |

## 3. Interpretation Note

Looking at the dashboard, we can observe a deliberate spike in the average fraud probability over the last 6 days of the simulated data. If this were real, my first action would be to investigate the incoming data distributions (drift) to see if fraudulent actors have changed their behaviour (concept drift), or if the model needs retraining on more recent data. The average latency is well within acceptable bounds (35ms).

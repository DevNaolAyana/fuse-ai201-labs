# Capstone Retrospective

**Author:** Naol Melaku Ayana
**Project:** Credit Card Fraud Detection

## What Worked Better Than Expected

The XGBoost model combined with SMOTE performed exceptionally well. Despite the extreme class imbalance (only 0.17% fraud), the model achieved an 84.2% recall rate, meaning it successfully caught 80 out of 95 fraud cases. The Streamlit frontend integrated seamlessly with the FastAPI backend, creating a fully functional decoupled system.

## What Was Harder Than Expected

Handling the data leakage during preprocessing was a major technical challenge. In initial iterations, scaling the `Time` and `Amount` features before the train/test split caused data leakage, leading to artificially high metrics. Learning to fit the scaler only on the training data and transform the test set was a crucial lesson in building robust ML pipelines.

## What I Would Do Differently

If I started again, I would implement the MLflow experiment tracking and Docker containerization from Day 1, rather than waiting until Module 6. Tracking hyperparameters and model versions from the very first baseline model would have made comparing the Logistic Regression and XGBoost models much easier. I would also push harder to deploy the API to a cloud provider like Render to make the app publicly accessible.

## Lessons Learned

Building an AI model is only 20% of the work. The remaining 80% is MLOps: API development, monitoring, drift detection, and governance. A model is useless if it cannot be deployed, monitored, or explained to stakeholders.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score

# Acceptance Rules
MIN_AUC = 0.75          # Candidate must reach at least this level
MAX_REGRESSION = 0.02   # Candidate can be at most 0.02 worse than champion
MIN_RECALL = 0.80       # Candidate must maintain high recall for the fraud class

def validate_data(df):
    print("--- Step 1: Validating Data ---")
    if df.isnull().sum().sum() > 0:
        raise ValueError("Data contains missing values.")
    if len(df) < 200000:
        raise ValueError("Not enough training rows.")
    print("Data validation passed.")
    return True

def retrain_and_evaluate():
    print(f"\n--- Starting Retraining Pipeline ---")
    
    # 1. Load and Validate
    df = pd.read_csv("data/creditcard.csv")
    validate_data(df)
    df = df.drop_duplicates()
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # 2. Fair Split (Same for both models)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Train Candidate (XGBoost)
    print("Training Candidate (XGBoost)...")
    candidate = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='aucpr', random_state=42)
    candidate.fit(X_train, y_train)
    
    # 4. Load Champion (Logistic Regression baseline)
    print("Evaluating Champion (Logistic Regression)...")
    champion = LogisticRegression(max_iter=1000, random_state=42)
    champion.fit(X_train, y_train)
    
    # 5. Evaluate on same validation data
    cand_prob = candidate.predict_proba(X_val)[:, 1]
    cand_auc = roc_auc_score(y_val, cand_prob)
    cand_recall = recall_score(y_val, (cand_prob >= 0.5).astype(int))
    
    champ_prob = champion.predict_proba(X_val)[:, 1]
    champ_auc = roc_auc_score(y_val, champ_prob)
    
    # 6. Apply Gates
    print(f"\n--- Evaluation Results ---")
    print(f"Candidate AUC: {cand_auc:.4f} | Recall: {cand_recall:.4f}")
    print(f"Champion AUC:  {champ_auc:.4f}")
    
    reasons = []
    if cand_auc < MIN_AUC:
        reasons.append(f"Below minimum AUC ({MIN_AUC})")
    if cand_auc < champ_auc - MAX_REGRESSION:
        reasons.append("Worse than champion")
    if cand_recall < MIN_RECALL:
        reasons.append(f"Recall below minimum ({MIN_RECALL})")

    # 7. Final Decision
    if reasons:
        print(f"\n❌ DECISION: REJECTED")
        for r in reasons:
            print(f"  - {r}")
    else:
        print(f"\n✅ DECISION: APPROVED for Staging (Human approval required before Production)")

if __name__ == "__main__":
    retrain_and_evaluate()
    
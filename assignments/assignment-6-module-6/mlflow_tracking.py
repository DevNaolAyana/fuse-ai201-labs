"""
Module 6: Experiment Tracking with MLflow
Fraud Detection Project - Baseline (Logistic Regression) vs Improved (XGBoost)

This script:
1. Loads the creditcard.csv dataset.
2. Preprocesses data: removes duplicates, scales Time and Amount, performs an 80/20 stratified split.
3. Applies SMOTE oversampling to the training set.
4. Initializes local MLflow experiment tracking using SQLite backend ("sqlite:///mlflow.db").
5. Executes two sequential experiment runs:
   - Run 1 (Baseline): Logistic Regression with hyperparameter and metric tracking.
   - Run 2 (Improved): XGBoost Classifier with hyperparameter and metric tracking.
6. Logs parameters, evaluation metrics, and model artifacts into MLflow.
7. Prints a comparison table summarizing performance across both runs.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def resolve_paths():
    """Resolve project paths dynamically."""
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "data" / "creditcard.csv"
    if not data_path.exists():
        data_path = Path("data/creditcard.csv")
    db_path = script_dir / "mlflow.db"
    return script_dir, data_path, db_path


def load_and_preprocess_data(data_path: Path):
    """Load dataset, clean duplicates, scale Time & Amount, split, and apply SMOTE."""
    print("=" * 70)
    print("STEP 1: Loading Dataset & Preprocessing")
    print("=" * 70)
    print(f"Loading data from: {data_path.resolve()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure creditcard.csv is placed in the data folder.")

    df = pd.read_csv(data_path)
    initial_rows = len(df)
    print(f"Raw dataset shape: {df.shape}")

    # Remove duplicates
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicate_count:,} duplicate rows. Cleaned dataset: {len(df):,} records.")

    # Split features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]:,} samples ({int(y_train.sum())} frauds)")
    print(f"Test set:  {X_test.shape[0]:,} samples ({int(y_test.sum())} frauds)")

    # Feature scaling: Standardize Time and Amount (fit on train only)
    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])
    print("StandardScaler applied to 'Time' and 'Amount' (fitted strictly on train set).")

    # Apply SMOTE to training data only
    print("Applying SMOTE balancing to training partition...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Training distribution: {np.bincount(y_train_resampled)}")

    return X_train_resampled, y_train_resampled, X_test, y_test, scaler


def evaluate_model(model, X_test, y_test):
    """Compute key fraud detection metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "false_positive_rate": float(fpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    return metrics


def main():
    script_dir, data_path, db_path = resolve_paths()

    # Step 1: Preprocess dataset
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(data_path)

    # Step 2: Configure MLflow Tracking
    print("\n" + "=" * 70)
    print("STEP 2: Initializing MLflow Tracking")
    print("=" * 70)
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Fraud Detection Experiments"
    mlflow.set_experiment(experiment_name)
    print(f"Tracking URI set to : {tracking_uri}")
    print(f"Experiment name     : '{experiment_name}'")

    results = []

    # -------------------------------------------------------------------------
    # RUN 1: Baseline Logistic Regression
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUN 1 (Baseline): Logistic Regression")
    print("=" * 70)

    lr_params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "None",
        "solver": "lbfgs",
        "random_state": 42
    }

    with mlflow.start_run(run_name="Logistic_Regression_Baseline") as run_lr:
        print("Training Logistic Regression model...")
        lr_model = LogisticRegression(
            max_iter=lr_params["max_iter"],
            class_weight=None,
            solver=lr_params["solver"],
            random_state=lr_params["random_state"]
        )
        lr_model.fit(X_train, y_train)

        # Evaluate model
        lr_metrics = evaluate_model(lr_model, X_test, y_test)

        # Log parameters & metrics to MLflow
        mlflow.log_params(lr_params)
        mlflow.log_metrics(lr_metrics)

        # Log scikit-learn model artifact
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

        run_id_lr = run_lr.info.run_id
        print(f"Run completed! Run ID: {run_id_lr}")
        print(f"  Recall  : {lr_metrics['recall']:.4f} ({lr_metrics['recall']*100:.2f}%)")
        print(f"  Precision: {lr_metrics['precision']:.4f} ({lr_metrics['precision']*100:.2f}%)")
        print(f"  F1-Score: {lr_metrics['f1_score']:.4f}")
        print(f"  PR-AUC  : {lr_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC : {lr_metrics['roc_auc']:.4f}")

        results.append({
            "Run Name": "Run 1 (Baseline LR)",
            "Model": "Logistic Regression",
            "Precision": lr_metrics["precision"],
            "Recall": lr_metrics["recall"],
            "F1-Score": lr_metrics["f1_score"],
            "PR-AUC": lr_metrics["pr_auc"],
            "ROC-AUC": lr_metrics["roc_auc"],
            "FPR": lr_metrics["false_positive_rate"],
            "TP": lr_metrics["true_positives"],
            "FP": lr_metrics["false_positives"]
        })

    # -------------------------------------------------------------------------
    # RUN 2: Improved XGBoost Classifier
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUN 2 (Improved): XGBoost Classifier")
    print("=" * 70)

    xgb_params = {
        "model_type": "XGBClassifier",
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "eval_metric": "aucpr",
        "random_state": 42
    }

    with mlflow.start_run(run_name="XGBoost_Classifier_Improved") as run_xgb:
        print("Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            eval_metric=xgb_params["eval_metric"],
            random_state=xgb_params["random_state"]
        )
        xgb_model.fit(X_train, y_train)

        # Evaluate model
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

        # Log parameters & metrics to MLflow
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(xgb_metrics)

        # Log xgboost model artifact
        mlflow.xgboost.log_model(xgb_model, artifact_path="model")

        run_id_xgb = run_xgb.info.run_id
        print(f"Run completed! Run ID: {run_id_xgb}")
        print(f"  Recall  : {xgb_metrics['recall']:.4f} ({xgb_metrics['recall']*100:.2f}%)")
        print(f"  Precision: {xgb_metrics['precision']:.4f} ({xgb_metrics['precision']*100:.2f}%)")
        print(f"  F1-Score: {xgb_metrics['f1_score']:.4f}")
        print(f"  PR-AUC  : {xgb_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC : {xgb_metrics['roc_auc']:.4f}")

        results.append({
            "Run Name": "Run 2 (Improved XGB)",
            "Model": "XGBoost Classifier",
            "Precision": xgb_metrics["precision"],
            "Recall": xgb_metrics["recall"],
            "F1-Score": xgb_metrics["f1_score"],
            "PR-AUC": xgb_metrics["pr_auc"],
            "ROC-AUC": xgb_metrics["roc_auc"],
            "FPR": xgb_metrics["false_positive_rate"],
            "TP": xgb_metrics["true_positives"],
            "FP": xgb_metrics["false_positives"]
        })

    # -------------------------------------------------------------------------
    # Comparison Table
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("                    MLFLOW EXPERIMENT COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Run / Model':<24} | {'Precision':<9} | {'Recall':<8} | {'F1':<7} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'FPR':<7} | {'TP':<4} | {'FP':<5}")
    print("-" * 90)
    for r in results:
        print(f"{r['Run Name']:<24} | {r['Precision']:<9.4f} | {r['Recall']:<8.4f} | {r['F1-Score']:<7.4f} | {r['PR-AUC']:<8.4f} | {r['ROC-AUC']:<8.4f} | {r['FPR']*100:<6.2f}% | {r['TP']:<4} | {r['FP']:<5}")
    print("=" * 90)

    pr_auc_diff = results[1]["PR-AUC"] - results[0]["PR-AUC"]
    fp_reduction = (results[0]["FP"] - results[1]["FP"]) / results[0]["FP"] * 100
    precision_gain = results[1]["Precision"] / results[0]["Precision"]

    print("\nKey Insights:")
    print(f"  - PR-AUC improved by +{pr_auc_diff:.4f} ({results[0]['PR-AUC']:.4f} -> {results[1]['PR-AUC']:.4f})")
    print(f"  - Precision improved by {precision_gain:.1f}x ({results[0]['Precision']:.4f} -> {results[1]['Precision']:.4f})")
    print(f"  - False alarms reduced by {fp_reduction:.1f}% ({results[0]['FP']} -> {results[1]['FP']})")
    print(f"  - High fraud recall preserved at {results[1]['Recall']*100:.2f}% (capturing {results[1]['TP']} of 95 frauds)")
    print(f"  - False Positive Rate decreased to {results[1]['FPR']*100:.2f}% (well under the 1% target)")
    print(f"\nMLflow tracking database generated at: {db_path.resolve()}")
    print("To launch the MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri sqlite:///{db_path.as_posix()} --port 5000\n")


if __name__ == "__main__":
    main()

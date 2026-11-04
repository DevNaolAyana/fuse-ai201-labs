import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def run_drift_check():
    print("--- Running Drift Detection ---")
    
    # 1. Load the simulated data from our monitoring DB
    conn = sqlite3.connect("logs/predictions.db")
    df = pd.read_sql("SELECT * FROM predictions WHERE is_simulated = 1", conn)
    conn.close()

    if len(df) < 100:
        print("Not enough data to run drift detection.")
        return

    # 2. Create our Reference (baseline) and Current (recent) datasets
    # We will split the data: older data is Reference, newer data is Current
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    split_idx = int(len(df) * 0.7)
    reference = df.iloc[:split_idx]
    current = df.iloc[split_idx:]

    print(f"Reference size: {len(reference)} rows")
    print(f"Current size:   {len(current)} rows")

    # 3. Check Prediction Drift (Comparing the fraud probabilities)
    # We use the Kolmogorov-Smirnov test for continuous variables
    ks_stat, p_value = stats.ks_2samp(reference['fraud_prob'], current['fraud_prob'])
    
    print(f"\nPrediction Drift (Fraud Probability):")
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  P-value:      {p_value:.4f}")
    
    # 4. Interpret the results
    if p_value < 0.05:
        print("  🚨 ALERT: Significant drift detected! The distribution of fraud probabilities has changed.")
        print("  Action: Investigate if fraudsters have changed tactics or if there is a data pipeline issue.")
    else:
        print("  ✅ No significant drift detected.")

if __name__ == "__main__":
    run_drift_check()
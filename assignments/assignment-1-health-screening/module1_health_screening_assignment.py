# ============================================================
# Module 1: Python Fundamentals for AI
# Practical Assignment: Community Health Screening Data Analysis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# TASK 1: Create the Dataset
# ============================================================

# Create dataset using a list of dictionaries
patients = [
    {"Patient_ID": 1, "Age": 45, "Gender": "Male", "Height": 175, "Weight": 80, 
     "Systolic_BP": 130, "Diastolic_BP": 85, "Blood_Sugar": 110, "Physical_Activity": 3, "Smoking_Status": "No"},
    
    {"Patient_ID": 2, "Age": 52, "Gender": "Female", "Height": 160, "Weight": 65, 
     "Systolic_BP": 140, "Diastolic_BP": 90, "Blood_Sugar": 125, "Physical_Activity": 2, "Smoking_Status": "Yes"},
    
    {"Patient_ID": 3, "Age": 38, "Gender": "Male", "Height": 182, "Weight": 95, 
     "Systolic_BP": 145, "Diastolic_BP": 95, "Blood_Sugar": 130, "Physical_Activity": 1, "Smoking_Status": "Yes"},
    
    {"Patient_ID": 4, "Age": 60, "Gender": "Female", "Height": 155, "Weight": 70, 
     "Systolic_BP": 150, "Diastolic_BP": 95, "Blood_Sugar": 140, "Physical_Activity": 1, "Smoking_Status": "No"},
    
    {"Patient_ID": 5, "Age": 29, "Gender": "Male", "Height": 170, "Weight": 68, 
     "Systolic_BP": 120, "Diastolic_BP": 75, "Blood_Sugar": 95, "Physical_Activity": 5, "Smoking_Status": "No"},
    
    {"Patient_ID": 6, "Age": 41, "Gender": "Female", "Height": 165, "Weight": 72, 
     "Systolic_BP": 135, "Diastolic_BP": 85, "Blood_Sugar": 115, "Physical_Activity": 2, "Smoking_Status": "No"},
    
    {"Patient_ID": 7, "Age": 55, "Gender": "Male", "Height": 178, "Weight": 88, 
     "Systolic_BP": 160, "Diastolic_BP": 100, "Blood_Sugar": 150, "Physical_Activity": 0, "Smoking_Status": "Yes"},
    
    {"Patient_ID": 8, "Age": 33, "Gender": "Female", "Height": 162, "Weight": 58, 
     "Systolic_BP": 115, "Diastolic_BP": 70, "Blood_Sugar": 85, "Physical_Activity": 4, "Smoking_Status": "No"},
    
    {"Patient_ID": 9, "Age": 47, "Gender": "Male", "Height": 180, "Weight": 92, 
     "Systolic_BP": 148, "Diastolic_BP": 92, "Blood_Sugar": 135, "Physical_Activity": 1, "Smoking_Status": "Yes"},
    
    {"Patient_ID": 10, "Age": 35, "Gender": "Female", "Height": 158, "Weight": 62, 
     "Systolic_BP": 122, "Diastolic_BP": 78, "Blood_Sugar": 100, "Physical_Activity": 3, "Smoking_Status": "No"},
    
    {"Patient_ID": 11, "Age": 65, "Gender": "Male", "Height": 172, "Weight": 78, 
     "Systolic_BP": 155, "Diastolic_BP": 98, "Blood_Sugar": 160, "Physical_Activity": 0, "Smoking_Status": "Yes"},
    
    {"Patient_ID": 12, "Age": 28, "Gender": "Female", "Height": 168, "Weight": 55, 
     "Systolic_BP": 110, "Diastolic_BP": 68, "Blood_Sugar": 90, "Physical_Activity": 6, "Smoking_Status": "No"}
]

# Convert to DataFrame
df = pd.DataFrame(patients)

# Display first 5 records
print("=" * 60)
print("TASK 1: Dataset Created")
print("=" * 60)
print("First 5 records:")
print(df.head())
print("\nDataset shape:", df.shape)


# ============================================================
# TASK 2: Data Cleaning
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: Data Cleaning")
print("=" * 60)

# Introduce missing values (2 missing values)
df.loc[0, "Physical_Activity"] = None  # Patient 1 missing physical activity
df.loc[5, "Blood_Sugar"] = None        # Patient 6 missing blood sugar

print("Missing values before cleaning:")
print(df.isnull().sum())

# Check for duplicates and remove if any
df = df.drop_duplicates()

# Fill missing numerical values with median
numeric_cols = ["Physical_Activity", "Blood_Sugar"]
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Verify no missing values remain
print("\nAll missing values handled:", df.isnull().sum().sum() == 0)


# ============================================================
# TASK 3: Create Health Indicators
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: Health Indicators")
print("=" * 60)

# Function to calculate BMI
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# Function to determine BMI category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Function to determine Blood Pressure Category
def get_bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return "Stage 1 Hypertension"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    else:
        return "Unknown"

# Function to determine Overall Health Risk
def get_health_risk(bmi, bp_category, blood_sugar, smoking_status):
    risk_score = 0
    
    # BMI risk
    if bmi >= 30:
        risk_score += 3
    elif bmi >= 25:
        risk_score += 2
    elif bmi < 18.5:
        risk_score += 1
    
    # Blood pressure risk
    if bp_category == "Stage 2 Hypertension":
        risk_score += 3
    elif bp_category == "Stage 1 Hypertension":
        risk_score += 2
    elif bp_category == "Elevated":
        risk_score += 1
    
    # Blood sugar risk
    if blood_sugar > 140:
        risk_score += 3
    elif blood_sugar > 120:
        risk_score += 2
    elif blood_sugar > 100:
        risk_score += 1
    
    # Smoking risk
    if smoking_status == "Yes":
        risk_score += 2
    
    # Determine risk level
    if risk_score >= 8:
        return "High"
    elif risk_score >= 5:
        return "Moderate"
    else:
        return "Low"

# Calculate BMI and categories
df["BMI"] = df.apply(lambda row: calculate_bmi(row["Weight"], row["Height"]), axis=1)
df["BMI_Category"] = df["BMI"].apply(get_bmi_category)
df["BP_Category"] = df.apply(lambda row: get_bp_category(row["Systolic_BP"], row["Diastolic_BP"]), axis=1)
df["Health_Risk"] = df.apply(
    lambda row: get_health_risk(row["BMI"], row["BP_Category"], row["Blood_Sugar"], row["Smoking_Status"]),
    axis=1
)

print("Health indicators added successfully!")
print(df[["Patient_ID", "BMI", "BMI_Category", "BP_Category", "Health_Risk"]].head(10))


# ============================================================
# TASK 4: NumPy Analysis
# ============================================================

print("\n" + "=" * 60)
print("TASK 4: NumPy Analysis")
print("=" * 60)

# Convert columns to NumPy arrays
bmi_array = np.array(df["BMI"])
blood_sugar_array = np.array(df["Blood_Sugar"])

print("BMI Statistics:")
print(f"Mean: {np.mean(bmi_array):.2f}")
print(f"Minimum: {np.min(bmi_array):.2f}")
print(f"Maximum: {np.max(bmi_array):.2f}")
print(f"Standard Deviation: {np.std(bmi_array):.2f}")

print("\nBlood Sugar Statistics:")
print(f"Mean: {np.mean(blood_sugar_array):.2f}")
print(f"Minimum: {np.min(blood_sugar_array):.2f}")
print(f"Maximum: {np.max(blood_sugar_array):.2f}")
print(f"Standard Deviation: {np.std(blood_sugar_array):.2f}")


# ============================================================
# TASK 5: Data Analysis Using Pandas
# ============================================================

print("\n" + "=" * 60)
print("TASK 5: Pandas Data Analysis")
print("=" * 60)

# Average BMI and blood sugar by gender
print("\n1. Average BMI and Blood Sugar by Gender:")
gender_stats = df.groupby("Gender")[["BMI", "Blood_Sugar"]].mean()
print(gender_stats)

# Number of patients by BMI category
print("\n2. Number of patients by BMI Category:")
print(df["BMI_Category"].value_counts())

# Number of patients by Health Risk
print("\n3. Number of patients by Health Risk:")
print(df["Health_Risk"].value_counts())

# Average age by gender
print("\n4. Average age by Gender:")
print(df.groupby("Gender")["Age"].mean())

# Percentage of smokers
smoker_percentage = (df["Smoking_Status"] == "Yes").mean() * 100
print(f"\n5. Percentage of Smokers: {smoker_percentage:.1f}%")


# ============================================================
# TASK 6: Data Visualization
# ============================================================

print("\n" + "=" * 60)
print("TASK 6: Data Visualization")
print("=" * 60)
print("Generating plots...")

# Set Seaborn style
sns.set_style("whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. BMI Histogram
sns.histplot(df["BMI"], bins=8, kde=True, ax=axes[0, 0], color="blue")
axes[0, 0].set_title("BMI Distribution", fontsize=14, fontweight="bold")
axes[0, 0].set_xlabel("BMI")
axes[0, 0].set_ylabel("Count")

# 2. Health Risk Bar Chart
risk_counts = df["Health_Risk"].value_counts()
sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Health Risk Distribution", fontsize=14, fontweight="bold")
axes[0, 1].set_xlabel("Health Risk")
axes[0, 1].set_ylabel("Number of Patients")

# 3. Age vs Blood Sugar Scatter Plot
sns.scatterplot(x=df["Age"], y=df["Blood_Sugar"], hue=df["Health_Risk"], size=df["Health_Risk"], sizes=(50, 150), ax=axes[1, 0])
axes[1, 0].set_title("Age vs Blood Sugar (by Health Risk)", fontsize=14, fontweight="bold")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Blood Sugar (mg/dL)")

# 4. Correlation Heatmap
numeric_df = df[["Age", "Height", "Weight", "Systolic_BP", "Diastolic_BP", "Blood_Sugar", "BMI"]]
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("health_screening_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'health_screening_analysis.png'")
print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
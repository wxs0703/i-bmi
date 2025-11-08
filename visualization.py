# team bmi

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded dataset directory
DATA_DIR = r"C:\Users\willy\OneDrive\Desktop\ShouldHavePutInDocuments\CMU Classes\10701\project\i-bmi"
CSV_FILE = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")

# Load data
df = pd.read_csv(CSV_FILE)

# Columns grouped by type
numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
binary_cols = ["hypertension", "heart_disease", "diabetes"]
categorical_cols = ["gender", "smoking_history"]

# Units/labels for each numeric variable
numeric_labels = {
    "age": "Age (years)",
    "bmi": "Body Mass Index (BMI)",
    "HbA1c_level": "HbA1c Level (%)",
    "blood_glucose_level": "Blood Glucose Level (a.u.)"
}

# Create figure for numeric variable histograms (mixed rules)
fig1 = plt.figure(figsize=(10, 8))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)

    if col == "age":
        bins = 30
    elif col == "bmi":
        bins = 30
    elif col == "HbA1c_level":
        bin_width = 0.5
        bins = np.arange(df[col].min(), df[col].max() + bin_width, bin_width)
    elif col == "blood_glucose_level":
        bin_width = 25
        bins = np.arange(df[col].min(), df[col].max() + bin_width, bin_width)
    else:
        bins = 30  

    plt.hist(df[col], bins=bins, edgecolor='black')
    plt.title(col)
    plt.xlabel(numeric_labels[col])
    plt.ylabel("Count")

plt.tight_layout()
fig1_path = os.path.join(DATA_DIR, "numeric_distributions.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)




# Create figure for binary variable counts

fig2 = plt.figure(figsize=(10, 6))
for i, col in enumerate(binary_cols, 1):
    plt.subplot(1, 3, i)
    df[col].value_counts().plot(kind='bar', rot=0)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
fig2_path = os.path.join(DATA_DIR, "binary_distributions.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)





# Create figure for categorical variable counts
fig3 = plt.figure(figsize=(10, 6))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    df[col].value_counts().plot(kind='bar', rot=45)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
fig3_path = os.path.join(DATA_DIR, "categorical_distributions.png")
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)




# Write basic dataset information to text file
info_path = os.path.join(DATA_DIR, "dataset_info.txt")
with open(info_path, "w") as f:
    f.write("Dataset Summary\n")
    f.write("========================\n\n")
    f.write(f"Total Participants: {len(df)}\n\n")
    f.write("Column Information (data types):\n")
    f.write("------------------------\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nMissing Values Per Column:\n")
    f.write("------------------------\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\nUnique Values in Categorical Variables:\n")
    f.write("------------------------\n")
    for col in categorical_cols:
        f.write(f"\n{col}:\n")
        f.write(df[col].value_counts().to_string())
        f.write("\n")

print("done")

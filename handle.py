import pandas as pd

# Read all data files
df_question = pd.read_csv("questionnaire_clean.csv")
df_demo = pd.read_csv("demographics_clean.csv")
df_response = pd.read_csv("response_clean.csv")
df_medication = pd.read_csv("medications_clean.csv")
df_mortality = pd.read_csv("mortality_clean.csv")

# Check for duplicate user IDs in each dataset
print("==================== Checking for Duplicate User IDs ====================")
print(f"Questionnaire - Total rows: {len(df_question)}, Unique SEQN: {df_question['SEQN'].nunique()}")
print(f"Demographics - Total rows: {len(df_demo)}, Unique SEQN: {df_demo['SEQN'].nunique()}")
print(f"Response - Total rows: {len(df_response)}, Unique SEQN: {df_response['SEQN'].nunique()}")
print(f"Medication - Total rows: {len(df_medication)}, Unique SEQN: {df_medication['SEQN'].nunique()}")
print(f"Mortality - Total rows: {len(df_mortality)}, Unique SEQN: {df_mortality['SEQN'].nunique()}")
print()

# Extract columns from questionnaire table (assuming SEQN is the user ID)
questionnaire_cols = ['SEQN', 'HAR12S', 'HAR2', 'MCQ300A', 'MCQ300C', 
                      'PAD440', 'BPQ090D', 'MCQ160B', 'MCQ160F', 
                      'MCQ160E', 'DIQ010', 'BPQ020', 'MCQ160M']
df_q_selected = df_question[questionnaire_cols]

# Extract columns from demographics table
demo_cols = ['SEQN', 'RIDAGEYR', 'RIDRETH1', 'RIAGENDR', 'INDFMPIR']
df_d_selected = df_demo[demo_cols]

# Extract columns from response table
response_cols = ['SEQN', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 
                 'BPXSY1', 'BPXDI1']
df_r_selected = df_response[response_cols]

# Extract columns from medication table
medication_cols = ['SEQN', 'RXDCOUNT']
df_m_selected = df_medication[medication_cols]

# For medication table: aggregate by SEQN to ensure unique user IDs
# Take the maximum RXDCOUNT for each user (most medications recorded)
df_m_selected = df_m_selected.groupby('SEQN', as_index=False)['RXDCOUNT'].max()

# Extract columns from mortality table
mortality_cols = ['SEQN', 'MORTSTAT']
df_mort_selected = df_mortality[mortality_cols]

# For mortality table: aggregate by SEQN to ensure unique user IDs
# Take the maximum MORTSTAT for each user (if died in any record, mark as 1)
df_mort_selected = df_mort_selected.groupby('SEQN', as_index=False)['MORTSTAT'].max()

# Merge all tables through inner join step by step
# Step 1: Merge questionnaire and demographics
df_merged = pd.merge(df_q_selected, df_d_selected, on='SEQN', how='inner')
print(f"After merging questionnaire and demographics: {len(df_merged)} rows")

# Step 2: Merge response
df_merged = pd.merge(df_merged, df_r_selected, on='SEQN', how='inner')
print(f"After merging response: {len(df_merged)} rows")

# Step 3: Merge medication
df_merged = pd.merge(df_merged, df_m_selected, on='SEQN', how='inner')
print(f"After merging medication: {len(df_merged)} rows")

# Step 4: Merge mortality
df_merged = pd.merge(df_merged, df_mort_selected, on='SEQN', how='inner')
print(f"After merging mortality: {len(df_merged)} rows")

# Save merged table
df_merged.to_csv("nhanes_merged_data.csv", index=False)

print(f"\nFinal merged table dimensions: {df_merged.shape}")
print(f"\nFirst 5 rows preview:")
print(df_merged.head())

print(f"\nData info:")
print(df_merged.info())

print(f"\nMissing values per column:")
print(df_merged.isnull().sum())

# ==================== Data Processing for Logistic Regression ====================

# 1. HAR12S: Fill NaN with 0
df_merged['HAR12S'] = df_merged['HAR12S'].fillna(0)

# Cap Hours_Sitting_per_Day at maximum 24 hours
df_merged['HAR12S'] = df_merged['HAR12S'].apply(lambda x: min(x, 24))
print(f"Hours_Sitting_per_Day - Max value after capping: {df_merged['HAR12S'].max()}")

# 2. RXDCOUNT: Fill NaN with 0
df_merged['RXDCOUNT'] = df_merged['RXDCOUNT'].fillna(0)

# 3. MORTSTAT: Convert to binary (1 if equals 1, else 0)
df_merged['MORTSTAT'] = df_merged['MORTSTAT'].apply(lambda x: 1 if x == 1 else 0)

# 4. Binary Yes/No columns: Keep 1 as 1, convert all other values to 0
# These columns represent binary yes/no questions
binary_columns = ['MCQ300A', 'MCQ300C', 'PAD440', 'BPQ090D', 
                  'MCQ160B', 'MCQ160F', 'MCQ160E', 'DIQ010', 
                  'BPQ020', 'MCQ160M']

for col in binary_columns:
    # First fill NaN with 0, then keep 1 as 1, convert everything else to 0
    df_merged[col] = df_merged[col].fillna(0)
    df_merged[col] = df_merged[col].apply(lambda x: 1 if x == 1 else 0)
    print(f"{col} - Unique values after processing: {sorted(df_merged[col].unique())}")

print("\n==================== Data Processing Complete ====================")
print(f"\nProcessed data dimensions: {df_merged.shape}")

print(f"\nFirst 5 rows after processing:")
print(df_merged.head())

print(f"\nMissing values after processing:")
print(df_merged.isnull().sum())

print(f"\nKey column distributions:")
print(f"\nMORTSTAT distribution:")
print(df_merged['MORTSTAT'].value_counts())

print(f"\nExample binary column (MCQ300A) distribution:")
print(df_merged['MCQ300A'].value_counts(dropna=False))

# 5. Create CVD_Diagnosed column: If any of MCQ160B, MCQ160F, MCQ160E is 1, set to 1; if all are 0, set to 0
df_merged['CVD_Diagnosed'] = df_merged[['MCQ160B', 'MCQ160F', 'MCQ160E']].apply(
    lambda row: 1 if row.sum() >= 1 else 0, axis=1
)

# Drop the three original CVD columns
df_merged = df_merged.drop(columns=['MCQ160B', 'MCQ160F', 'MCQ160E'])

print(f"\nCVD_Diagnosed distribution:")
print(df_merged['CVD_Diagnosed'].value_counts())

# 6. Rename columns to human-readable names
column_mapping = {
    'SEQN': 'User_ID',
    'HAR12S': 'Hours_Sitting_per_Day',
    'HAR2': 'Age_First_Period',
    'MCQ300A': 'Family_History_Overweight',
    'MCQ300C': 'Family_History_Diabetes',
    'PAD440': 'Physically_Active',
    'BPQ090D': 'Told_High_Cholesterol',
    'DIQ010': 'Diagnosed_Diabetes',
    'BPQ020': 'Diagnosed_Hypertension',
    'MCQ160M': 'Diagnosed_Thyroid_Problem',
    'RIDAGEYR': 'Age',
    'RIDRETH1': 'Race_Ethnicity',
    'RIAGENDR': 'Gender',
    'INDFMPIR': 'Poverty_Income_Ratio',
    'BMXHT': 'Height_cm',
    'BMXWT': 'Weight_kg',
    'BMXBMI': 'BMI',
    'BMXWAIST': 'Waist_Circumference_cm',
    'BPXSY1': 'Systolic_BP',
    'BPXDI1': 'Diastolic_BP',
    'RXDCOUNT': 'Number_of_Medications',
    'MORTSTAT': 'Mortality_Status',
    'CVD_Diagnosed': 'CVD_Diagnosed'
}

df_merged = df_merged.rename(columns=column_mapping)

print("\n==================== Final Data Table ====================")
print(f"\nFinal data dimensions: {df_merged.shape}")
print(f"\nColumn names:")
print(df_merged.columns.tolist())

# Save final processed data
df_merged.to_csv("nhanes_merged_processed.csv", index=False)

# Check for duplicate User_IDs in final merged data
print(f"\nChecking for duplicate User_IDs in merged data:")
print(f"Total rows: {len(df_merged)}")
print(f"Unique User_IDs: {df_merged['User_ID'].nunique()}")
duplicate_ids = df_merged[df_merged.duplicated(subset=['User_ID'], keep=False)]
if len(duplicate_ids) > 0:
    print(f"WARNING: Found {len(duplicate_ids)} duplicate rows!")
    print(f"Number of unique IDs with duplicates: {duplicate_ids['User_ID'].nunique()}")
    print("\nRemoving duplicate rows, keeping first occurrence...")
    df_merged = df_merged.drop_duplicates(subset=['User_ID'], keep='first')
    print(f"After removing duplicates: {len(df_merged)} rows")
else:
    print("No duplicate User_IDs found!")

print(f"\nFinal data first 5 rows:")
print(df_merged.head())

print(f"\nFinal missing values per column:")
print(df_merged.isnull().sum())

# 7. Remove rows with any missing values
print("\n==================== Removing Rows with Missing Values ====================")
print(f"\nBefore removing missing values: {len(df_merged)} rows")

# First, show rows with missing values
rows_with_missing = df_merged[df_merged.isnull().any(axis=1)]
print(f"\nTotal rows with missing values: {len(rows_with_missing)}")

if len(rows_with_missing) > 0:
    print(f"\n==================== First 20 Rows with Missing Values ====================")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # No width limit
    pd.set_option('display.max_colwidth', None)  # Show full column content
    
    print(rows_with_missing.head(20).to_string())
    
    print(f"\n\nMissing value counts by column (for rows with missing data):")
    print(rows_with_missing.isnull().sum())
    
    print(f"\n\nWhich columns have missing values (count > 0):")
    missing_cols = rows_with_missing.isnull().sum()
    print(missing_cols[missing_cols > 0].sort_values(ascending=False))

df_merged_complete = df_merged.dropna()

print(f"\nAfter removing missing values: {len(df_merged_complete)} rows")
print(f"Rows removed: {len(df_merged) - len(df_merged_complete)}")
print(f"Percentage of data retained: {len(df_merged_complete)/len(df_merged)*100:.2f}%")

# Verify no missing values remain
print(f"\nMissing values after removal:")
print(df_merged_complete.isnull().sum().sum())

# Save complete data without missing values
df_merged_complete.to_csv("nhanes_merged_complete.csv", index=False)

print(f"\nFinal complete data dimensions: {df_merged_complete.shape}")
print(f"\nFinal complete data first 5 rows:")
print(df_merged_complete.head())

print("\n==================== Summary ====================")
print(f"Original merged data: {len(df_merged)} rows")
print(f"Complete data (no missing values): {len(df_merged_complete)} rows")
print(f"Data saved to: nhanes_merged_complete.csv")
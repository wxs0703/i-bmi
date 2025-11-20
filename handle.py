import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

"""
NHANES Dataset Feature Extraction and Preprocessing
Goal: Construct obesity-related health risk index R

Tables used:
- demographics_clean.csv
- questionnaire_clean.csv
- mortality_clean.csv
- medications_clean.csv
- response_clean.csv (contains exam & lab data)
"""

# ============================================================================
# Step 0: Column Name Mapping (NHANES codes → Readable English names)
# ============================================================================

COLUMN_NAME_MAPPING = {
    # Demographics
    'SEQN': 'patient_id',
    'RIDAGEYR': 'age',
    'RIAGENDR': 'gender',
    'RIDRETH3': 'race_ethnicity',
    'DMDEDUC2': 'education',
    'INDFMPIR': 'income_poverty_ratio',
    'DMDMARTL': 'marital_status',
    
    # Body measurements
    'BMXBMI': 'bmi',
    'BMXWT': 'weight_kg',
    'BMXHT': 'height_cm',
    'BMXWAIST': 'waist_circumference',
    
    # Blood pressure
    'BPXSY1': 'systolic_bp_1',
    'BPXSY2': 'systolic_bp_2',
    'BPXDI1': 'diastolic_bp_1',
    'BPXDI2': 'diastolic_bp_2',
    
    # Glucose & diabetes markers
    'LBXGLU': 'fasting_glucose',
    'LBXGH': 'hemoglobin_a1c',
    
    # Lipid profile
    'LBXTC': 'total_cholesterol',
    'LBXTR': 'triglycerides',
    'LBDHDD': 'hdl_cholesterol',
    'LBDLDL': 'ldl_cholesterol',
    
    # Kidney function
    'LBXSCR': 'serum_creatinine',
    'LBXSUA': 'uric_acid',
    
    # Liver function
    'LBXSASSI': 'ast_liver_enzyme',
    'LBXSATSI': 'alt_liver_enzyme',
    
    # Inflammatory markers
    'LBXCRP': 'c_reactive_protein',
    
    # Other biomarkers
    'LBXFER': 'ferritin',
    'LBXTSH': 'thyroid_stimulating_hormone',
    'LBXHCY': 'homocysteine',
    
    # Blood count
    'LBXHGB': 'hemoglobin',
    'LBXWBCSI': 'white_blood_cell_count',
    
    # Questionnaire - Diabetes
    'DIQ010': 'diabetes_diagnosed',
    'DIQ050': 'taking_insulin',
    'DIQ070': 'taking_diabetes_pills',
    
    # Questionnaire - Cardiovascular
    'BPQ020': 'hypertension_diagnosed',
    'BPQ050A': 'taking_bp_medication',
    'BPQ080': 'high_cholesterol_diagnosed',
    'MCQ160B': 'congestive_heart_failure',
    'MCQ160C': 'coronary_heart_disease',
    'MCQ160D': 'angina',
    'MCQ160E': 'heart_attack',
    'MCQ160F': 'stroke',
    
    # Questionnaire - Other diseases
    'MCQ160L': 'liver_condition',
    'MCQ220': 'cancer_diagnosed',
    
    # Questionnaire - Lifestyle
    'ALQ101': 'ever_drank_alcohol',
    'ALQ120Q': 'alcohol_frequency_past_year',
    'PAQ605': 'vigorous_work_activity',
    'PAQ620': 'moderate_work_activity',
    'PAQ665': 'vigorous_recreational_activity',
    
    # Questionnaire - Family history
    'MCQ250B': 'family_diabetes',
    'MCQ250D': 'family_hypertension',
    'MCQ250E': 'family_heart_attack',
    
    # Mortality
    'MORTSTAT': 'mortality_status',
    'UCOD_LEADING': 'cause_of_death',
    'DIABETES': 'diabetes_related_death',
    'HYPERTEN': 'hypertension_related_death',
    'PERMTH_INT': 'followup_months_interview',
    'PERMTH_EXM': 'followup_months_exam',
    
    # Medication
    'RXDCOUNT': 'prescription_medication_count',
}

# ============================================================================
# Step 1: Define features to extract
# ============================================================================

SELECTED_FEATURES = {
    
    # DEMOGRAPHIC TABLE
    'demographic': {
        'id': 'SEQN',
        'features': [
            'RIDAGEYR',      # Age in years
            'RIAGENDR',      # Gender (1=Male, 2=Female)
            'RIDRETH3',      # Race/Ethnicity
            'INDFMPIR',      # Family poverty income ratio
        ]
    },
    
    # RESPONSE TABLE - Contains exam and lab data
    'response': {
        'id': 'SEQN',
        'features': {
            # Body measurements
            'anthropometric': [
                'BMXBMI',        # BMI
                'BMXWT',         # Weight (kg)
                'BMXHT',         # Height (cm)
                'BMXWAIST',      # Waist circumference (cm)
            ],
            
            # Blood pressure
            'blood_pressure': [
                'BPXSY1',        # Systolic BP reading 1 (mmHg)
                'BPXDI1',        # Diastolic BP reading 1 (mmHg)
                'BPXSY2',        # Systolic BP reading 2
                'BPXDI2',        # Diastolic BP reading 2
            ],
            
            # Glucose and diabetes markers
            'glucose': [
                'LBXGLU',        # Fasting glucose (mg/dL)
                'LBXGH',         # Glycohemoglobin (%)
            ],
            
            # Lipid profile
            'lipids': [
                'LBXTC',         # Total cholesterol (mg/dL)
                'LBXTR',         # Triglycerides (mg/dL)
                'LBDHDD',        # HDL cholesterol (mg/dL)
                'LBDLDL',        # LDL cholesterol (mg/dL)
            ],
            
            # Kidney function
            'kidney': [
                'LBXSCR',        # Serum creatinine (mg/dL)
                'LBXSUA',        # Uric acid (mg/dL)
            ],
            
            # Liver function
            'liver': [
                'LBXSASSI',      # AST (U/L)
                'LBXSATSI',      # ALT (U/L)
            ],
        }
    },
    
    # QUESTIONNAIRE TABLE
    'questionnaire': {
        'id': 'SEQN',
        'features': {
            # Diabetes
            'diabetes': [
                'DIQ010',        # Doctor told you have diabetes
                'DIQ050',        # Taking insulin now
                'DIQ070',        # Taking diabetic pills
            ],
            
            # Cardiovascular disease
            'cardiovascular': [
                'BPQ020',        # Ever told you had high blood pressure
                'BPQ050A',       # Now taking prescribed medicine for HBP
                'BPQ080',        # Doctor told you - high cholesterol
            ],
            
            # Other chronic diseases
            'chronic_diseases': [
                'MCQ160L',       # Ever had any liver condition
                'MCQ220',        # Ever told you had cancer
            ],
            
            # Lifestyle - Alcohol
            'alcohol': [
                'ALQ101',        # Ever had a drink of any alcohol
                'ALQ120Q',       # How often drink alcohol past 12 months
            ],
            
            # Physical activity
            'physical_activity': [
                'PAQ605',        # Vigorous work activity
                'PAQ620',        # Moderate work activity
                'PAQ665',        # Vigorous recreational activities
            ],
            
            # Family history
            'family_history': [
                'MCQ250B',       # Blood relatives have diabetes
                'MCQ250D',       # Blood relatives have hypertension
                'MCQ250E',       # Blood relatives have heart attack
            ],
        }
    },
    
    # MORTALITY TABLE
    'mortality': {
        'id': 'SEQN',
        'features': [
            'MORTSTAT',      # Mortality status (0=Alive, 1=Deceased)
            'UCOD_LEADING',  # Underlying leading cause of death
            'DIABETES',      # Diabetes as cause of death
        ]
    },
    
    # MEDICATION TABLE
    'medication': {
        'id': 'SEQN',
        'features': [
            'RXDCOUNT',      # Number of prescription medications
        ]
    },
}


# ============================================================================
# Step 2: Data loading function
# ============================================================================

def load_and_extract_features(file_paths):
    """
    Load CSV files and extract selected features
    """
    extracted_data = {}
    
    for table_name, path in file_paths.items():
        print(f"Loading {table_name} table...")
        try:
            df = pd.read_csv(path, low_memory=False)
            
            # Get required columns for this table
            config = SELECTED_FEATURES.get(table_name, {})
            id_col = config.get('id', 'SEQN')
            features = config.get('features', [])
            
            # Flatten nested feature dictionary
            if isinstance(features, dict):
                all_features = [id_col]
                for category_features in features.values():
                    all_features.extend(category_features)
            else:
                all_features = [id_col] + features
            
            # Keep only available columns
            available_cols = [col for col in all_features if col in df.columns]
            missing_cols = [col for col in all_features if col not in df.columns]
            
            if missing_cols:
                print(f"  Warning: Missing {len(missing_cols)} columns")
                print(f"  Examples: {missing_cols[:5]}")
            
            extracted_data[table_name] = df[available_cols]
            print(f"  ✓ Extracted {len(available_cols)-1} features from {len(df)} rows")
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {path}")
        except Exception as e:
            print(f"  ✗ Error loading {table_name}: {str(e)}")
    
    return extracted_data


# ============================================================================
# Step 3: Consolidate multiple cycles
# ============================================================================

def consolidate_cycles(df):
    """
    Consolidate data from multiple NHANES cycles
    For variables like BMXBMI, BMXBMI1, BMXBMI2, take the first non-null value
    """
    print("\nConsolidating multi-cycle variables...")
    
    # Find base variable names
    base_vars = set()
    for col in df.columns:
        if col.endswith('1') or col.endswith('2'):
            base_vars.add(col[:-1])
    
    consolidated_count = 0
    for base_var in base_vars:
        variants = [base_var, base_var + '1', base_var + '2']
        available_variants = [v for v in variants if v in df.columns]
        
        if len(available_variants) > 1:
            # Take first non-null value across cycles
            df[base_var + '_consolidated'] = df[available_variants].bfill(axis=1).iloc[:, 0]
            consolidated_count += 1
    
    print(f"  ✓ Consolidated {consolidated_count} variables")
    return df


# ============================================================================
# Step 4: Merge all tables
# ============================================================================

def merge_all_tables(extracted_data):
    """
    Merge all tables by SEQN
    """
    print("\nMerging all tables...")
    
    if 'demographic' not in extracted_data:
        print("  ✗ ERROR: demographic table is required!")
        return None
    
    merged_df = extracted_data['demographic'].copy()
    print(f"  Starting with demographic: {len(merged_df)} rows")
    
    # Merge other tables
    for table_name in ['response', 'questionnaire', 'mortality', 'medication']:
        if table_name in extracted_data:
            merged_df = merged_df.merge(
                extracted_data[table_name],
                on='SEQN',
                how='left'
            )
            print(f"  After merging {table_name}: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        else:
            print(f"  Skipping {table_name} (not loaded)")
    
    return merged_df


# ============================================================================
# Step 5: Rename columns to readable English
# ============================================================================

def rename_columns_to_readable(df):
    """
    Rename NHANES column codes to readable English names
    """
    print("\nRenaming columns to readable English...")
    
    # Create rename mapping for columns that exist in dataframe
    rename_map = {}
    for old_name, new_name in COLUMN_NAME_MAPPING.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
        # Also check for consolidated versions
        if old_name + '_consolidated' in df.columns:
            rename_map[old_name + '_consolidated'] = new_name
    
    df_renamed = df.rename(columns=rename_map)
    print(f"  ✓ Renamed {len(rename_map)} columns")
    
    # Show examples
    if rename_map:
        print(f"\n  Example renamings:")
        for i, (old, new) in enumerate(list(rename_map.items())[:5]):
            print(f"    {old} → {new}")
    
    return df_renamed


# ============================================================================
# Step 6: Data cleaning and preprocessing
# ============================================================================

def clean_and_preprocess(df):
    """
    Clean and preprocess merged data
    """
    print("\nData cleaning and preprocessing...")
    df_clean = df.copy()
    
    # 1. Handle missing value markers
    missing_markers = [7, 9, 77, 99, 777, 999, 7777, 9999, '.', '']
    df_clean = df_clean.replace(missing_markers, np.nan)
    
    # 2. Consolidate multi-cycle variables
    df_clean = consolidate_cycles(df_clean)
    
    # 3. Create main BMI column
    bmi_candidates = ['bmi', 'bmi_consolidated', 'BMXBMI', 'BMXBMI_consolidated']
    for candidate in bmi_candidates:
        if candidate in df_clean.columns:
            df_clean['bmi'] = df_clean[candidate]
            break
    
    # 4. Create main age column
    age_candidates = ['age', 'RIDAGEYR']
    for candidate in age_candidates:
        if candidate in df_clean.columns:
            df_clean['age'] = df_clean[candidate]
            break
    
    # 5. Keep only samples with BMI and age
    if 'bmi' in df_clean.columns and 'age' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['bmi', 'age'])
        print(f"  Removed missing BMI/age: {initial_count} → {len(df_clean)}")
    
    # 6. Keep only adults (>= 18 years)
    if 'age' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['age'] >= 18]
        print(f"  Kept only adults: {initial_count} → {len(df_clean)}")
    
    # 7. Report missing rates
    print("\nMissing rates for key features:")
    key_vars = ['bmi', 'age', 'systolic_bp_1', 'fasting_glucose', 
                'total_cholesterol', 'hdl_cholesterol', 'mortality_status']
    
    for var in key_vars:
        if var in df_clean.columns:
            missing_rate = (df_clean[var].isnull().sum() / len(df_clean) * 100)
            if missing_rate > 0:
                print(f"  {var}: {missing_rate:.1f}%")
    
    return df_clean


# ============================================================================
# Step 7: Calculate health risk index R
# ============================================================================

def calculate_risk_score_R(df):
    """
    Calculate obesity-related health risk index R
    """
    print("\nCalculating health risk index R...")
    df_r = df.copy()
    
    # Find BMI column
    if 'bmi' not in df_r.columns:
        print("  ✗ ERROR: No BMI column found!")
        return df_r
    
    # Initialize R with BMI
    df_r['health_risk_score'] = df_r['bmi'].copy()
    
    # Helper to get column
    def get_col(names):
        for name in names if isinstance(names, list) else [names]:
            if name in df_r.columns:
                return name
        return None
    
    # Risk adjustments
    adjustments_applied = []
    
    # 1. Glucose
    glucose_col = get_col(['fasting_glucose', 'LBXGLU'])
    if glucose_col:
        df_r.loc[df_r[glucose_col] >= 100, 'health_risk_score'] += 2
        df_r.loc[df_r[glucose_col] >= 126, 'health_risk_score'] += 3
        adjustments_applied.append('glucose')
    
    # 2. HbA1c
    a1c_col = get_col(['hemoglobin_a1c', 'LBXGH'])
    if a1c_col:
        df_r.loc[df_r[a1c_col] >= 5.7, 'health_risk_score'] += 2
        df_r.loc[df_r[a1c_col] >= 6.5, 'health_risk_score'] += 3
        adjustments_applied.append('HbA1c')
    
    # 3. Blood pressure
    bp_col = get_col(['systolic_bp_1', 'BPXSY1'])
    if bp_col:
        df_r.loc[df_r[bp_col] >= 130, 'health_risk_score'] += 2
        df_r.loc[df_r[bp_col] >= 140, 'health_risk_score'] += 2
        adjustments_applied.append('blood_pressure')
    
    # 4. Hypertension diagnosis
    htn_col = get_col(['hypertension_diagnosed', 'BPQ020'])
    if htn_col:
        df_r.loc[df_r[htn_col] == 1, 'health_risk_score'] += 2
        adjustments_applied.append('hypertension_diagnosis')
    
    # 5. Lipids
    tc_col = get_col(['total_cholesterol', 'LBXTC'])
    if tc_col:
        df_r.loc[df_r[tc_col] >= 240, 'health_risk_score'] += 2
    
    ldl_col = get_col(['ldl_cholesterol', 'LBDLDL'])
    if ldl_col:
        df_r.loc[df_r[ldl_col] >= 160, 'health_risk_score'] += 2
    
    hdl_col = get_col(['hdl_cholesterol', 'LBDHDD'])
    if hdl_col:
        df_r.loc[df_r[hdl_col] < 40, 'health_risk_score'] += 2
        adjustments_applied.append('lipids')
    
    # 6. Diabetes diagnosis
    dm_col = get_col(['diabetes_diagnosed', 'DIQ010'])
    if dm_col:
        df_r.loc[df_r[dm_col] == 1, 'health_risk_score'] += 3
        adjustments_applied.append('diabetes')
    
    # 7. Mortality
    mort_col = get_col(['mortality_status', 'MORTSTAT'])
    if mort_col:
        df_r.loc[df_r[mort_col] == 1, 'health_risk_score'] += 15
        adjustments_applied.append('mortality')
    
    # 8. Age
    age_col = get_col(['age', 'RIDAGEYR'])
    if age_col:
        df_r.loc[df_r[age_col] >= 60, 'health_risk_score'] += 2
        df_r.loc[df_r[age_col] >= 70, 'health_risk_score'] += 2
        adjustments_applied.append('age')
    
    # 9. Exercise (protective)
    exercise_col = get_col(['vigorous_recreational_activity', 'PAQ665'])
    if exercise_col:
        df_r.loc[df_r[exercise_col] == 1, 'health_risk_score'] -= 1
        adjustments_applied.append('exercise')
    
    print(f"  ✓ Applied {len(adjustments_applied)} risk adjustments:")
    print(f"    {', '.join(adjustments_applied)}")
    
    # Report distribution
    print(f"\n  Health Risk Score (R) distribution:")
    print(f"    Min: {df_r['health_risk_score'].min():.1f}")
    print(f"    25th: {df_r['health_risk_score'].quantile(0.25):.1f}")
    print(f"    Median: {df_r['health_risk_score'].median():.1f}")
    print(f"    75th: {df_r['health_risk_score'].quantile(0.75):.1f}")
    print(f"    Max: {df_r['health_risk_score'].max():.1f}")
    print(f"    Mean: {df_r['health_risk_score'].mean():.1f} ± {df_r['health_risk_score'].std():.1f}")
    
    return df_r


# ============================================================================
# Step 8: Feature normalization
# ============================================================================

def normalize_features(df, target_col='health_risk_score'):
    """
    Normalize numeric features to [0, 1]
    """
    print("\nNormalizing features to [0, 1]...")
    df_norm = df.copy()
    
    # Get numeric columns
    exclude_cols = ['patient_id', 'SEQN', target_col, 'bmi', 'age']
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Normalize
    scaler = MinMaxScaler()
    df_norm[cols_to_normalize] = scaler.fit_transform(
        df_norm[cols_to_normalize].fillna(df_norm[cols_to_normalize].median())
    )
    
    print(f"  ✓ Normalized {len(cols_to_normalize)} features")
    
    return df_norm, scaler


# ============================================================================
# Main function
# ============================================================================

def main(file_paths):
    """
    Execute complete data processing pipeline
    """
    print("="*80)
    print("NHANES DATA PROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    extracted_data = load_and_extract_features(file_paths)
    
    if not extracted_data:
        print("\n✗ No data loaded. Check file paths.")
        return None, None
    
    # Step 2: Merge tables
    df_merged = merge_all_tables(extracted_data)
    if df_merged is None:
        return None, None
    
    # Step 3: Rename columns to readable English
    df_renamed = rename_columns_to_readable(df_merged)
    
    # Step 4: Clean data
    df_clean = clean_and_preprocess(df_renamed)
    print("Before:", len(df_clean))
    cols = [
    'patient_id',
    'age',
    'gender',
    'race_ethnicity',
    'income_poverty_ratio',
    'bmi',
    'weight_kg',
    'height_cm'
    ]

    df_clean = df_clean.drop_duplicates(subset=cols, keep='first')
    print("After:", len(df_clean))
    return df_clean


# ============================================================================
# Run script
# ============================================================================

if __name__ == "__main__":
    # Configure file paths
    FILE_PATHS = {
        'demographic': 'demographics_clean.csv',
        'response': 'response_clean.csv',
        'questionnaire': 'questionnaire_clean.csv',
        'mortality': 'mortality_clean.csv',
        'medication': 'medications_clean.csv'
    }
    
    # Run pipeline
    df_processed = main(FILE_PATHS)
    
    if df_processed is not None:
        # Save results
        output_file = 'nhanes_processed_final.csv'
        df_processed.to_csv(output_file, index=False)
        print(f"\n✓ Data saved to: {output_file}")
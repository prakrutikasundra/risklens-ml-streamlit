"""
Week 2 — Preprocessing & Pipeline Builder
Loan Default Prediction System
Uses relative paths for cross-platform cloud compatibility.
Prevents Data Leakage by fitting transformers ONLY on training data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_loan_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

NUMERICAL_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
]

CATEGORICAL_FEATURES = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

TARGET_COLUMN = "Default"

def build_preprocessing_pipeline():
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline.
    - Numerical: Median Imputation -> StandardScaler
    - Categorical: Mode Imputation -> OneHotEncoder(handle_unknown='ignore', drop='first')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor

def clean_and_split_data(raw_csv_path=RAW_DATA_PATH, test_size=0.2, seed=42):
    """
    Reads raw CSV, handles duplicates & outliers, splits train/test, fits preprocessor on train,
    transforms both train and test sets, and saves preprocessor pipeline object.
    """
    df = pd.read_csv(raw_csv_path)
    print(f"[Preprocessing] Initial dataset shape: {df.shape}")
    
    # 1. Remove duplicate rows if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[Preprocessing] Dropped {duplicates} duplicate rows. New shape: {df.shape}")
    else:
        print("[Preprocessing] Zero duplicate rows detected.")
        
    # 2. Outlier Clipping on key numerical columns (IQR method)
    df_clean = df.copy()
    for col in ["Income", "LoanAmount", "DTIRatio"]:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
    # 3. Separate features X and target y
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN].values
    
    # 4. Stratified Train / Test Split (Strictly NO data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"[Preprocessing] Stratified Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # 5. Build and fit pipeline ONLY on training data
    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + encoded_cat_cols
    
    # Save transformed datasets
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save preprocessor object
    pipeline_file = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, pipeline_file)
    print(f"[Preprocessing] Preprocessing pipeline saved to '{pipeline_file}'")
    
    # Save train & test splits as CSV for downstream models
    df_train_trans = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_train_trans[TARGET_COLUMN] = y_train
    df_train_trans.to_csv(os.path.join(DATA_DIR, "train_cleaned.csv"), index=False)
    
    df_test_trans = pd.DataFrame(X_test_transformed, columns=feature_names)
    df_test_trans[TARGET_COLUMN] = y_test
    df_test_trans.to_csv(os.path.join(DATA_DIR, "test_cleaned.csv"), index=False)
    
    print("[Preprocessing] Cleaned & encoded train/test datasets saved successfully.")
    
    return preprocessor, X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed, feature_names

if __name__ == "__main__":
    clean_and_split_data()

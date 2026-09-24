"""
Week 1 — Problem Definition & Dataset Exploration
Loan Default Prediction System
Uses relative paths for cloud compatibility.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_loan_data.csv")

def explore_dataset(filepath=DEFAULT_DATA_PATH):
    print("=" * 60)
    print("      WEEK 1: PROBLEM DEFINITION & DATASET EXPLORATION")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # 1. Dataset Shape
    print(f"\n1. DATASET SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Columns & Data Types
    print("\n2. COLUMNS & DATA TYPES:")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "DataType": df.dtypes.values,
        "MissingValues": df.isnull().sum().values,
        "MissingPercentage": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    print(dtypes_df.to_string(index=False))
    
    # 3. Target Variable Analysis
    target_col = "Default"
    print(f"\n3. TARGET VARIABLE: '{target_col}'")
    print("   Meaning: 'Default' is a binary class target variable representing loan repayment outcome.")
    print("   - 0 (Non-Default): Borrower successfully repaid or is actively making scheduled payments.")
    print("   - 1 (Default): Borrower failed to meet legal obligations of loan repayment (30+ days overdue).")
    
    target_counts = df[target_col].value_counts()
    target_pcts = df[target_col].value_counts(normalize=True) * 100
    print("\n   Class Distribution:")
    for val in target_counts.index:
        label = "Non-Default (0)" if val == 0 else "Default (1)"
        print(f"   - Class {val} ({label}): {target_counts[val]} samples ({target_pcts[val]:.2f}%)")
        
    # 4. Basic Statistics (Numerical Features)
    print("\n4. NUMERICAL FEATURES SUMMARY STATISTICS:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())
    
    # 5. Categorical Features Summary
    print("\n5. CATEGORICAL FEATURES SUMMARY:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_vals = df[col].unique()
        print(f"   - {col} ({len(unique_vals)} unique values): {list(unique_vals)}")
        
    # 6. Initial Observations
    observations = [
        f"Dataset contains {df.shape[0]} borrower records with {df.shape[1] - 1} predictor features and 1 target ('Default').",
        f"Target distribution is imbalanced with {target_pcts[0]:.1f}% Non-Default and {target_pcts[1]:.1f}% Default cases.",
        f"Minor missing values detected in Income ({df['Income'].isnull().sum()}) and CreditScore ({df['CreditScore'].isnull().sum()}), requiring imputation in Week 2.",
        "Key numerical indicators include Age, Income, LoanAmount, CreditScore, MonthsEmployed, DTIRatio, and InterestRate.",
        "Categorical features include Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner."
    ]
    
    print("\n6. INITIAL OBSERVATIONS:")
    for i, obs in enumerate(observations, 1):
        print(f"   [{i}] {obs}")
        
    print("\n" + "=" * 60)
    print("Week 1 Exploration completed successfully.")
    print("=" * 60)
    
    return df, dtypes_df, observations

if __name__ == "__main__":
    explore_dataset()

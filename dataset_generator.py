"""
Dataset Generator for Loan Default Prediction System
Fulfills Week 1 SOP requirements matching Kaggle nikhil1e9/loan-default schema.
Uses relative paths for cross-platform cloud deployment compatibility.
"""

import os
import numpy as np
import pandas as pd

# Define relative paths based on project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

def generate_loan_dataset(n_samples=15000, seed=42):
    """
    Generates a realistic synthetic dataset matching the Kaggle Loan Default Prediction dataset schema.
    Uses realistic domain rules for financial variables and default risk calculation.
    """
    np.random.seed(seed)
    
    # 1. Numerical Demographics & Financials
    age = np.random.randint(18, 70, size=n_samples)
    income = np.random.exponential(scale=35000, size=n_samples) + 15000
    income = np.clip(income, 15000, 160000).astype(int)
    
    loan_amount = np.random.exponential(scale=60000, size=n_samples) + 5000
    loan_amount = np.clip(loan_amount, 5000, 450000).astype(int)
    
    credit_score = np.random.normal(loc=660, scale=85, size=n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    months_employed = np.random.exponential(scale=30, size=n_samples)
    months_employed = np.clip(months_employed, 0, 120).astype(int)
    
    num_credit_lines = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.25, 0.45, 0.20, 0.10])
    
    interest_rate = np.random.uniform(2.0, 24.0, size=n_samples).round(2)
    loan_term = np.random.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    
    # DTI Ratio = Debt / Income estimate
    dti_ratio = (loan_amount * (interest_rate / 100) / (income + 1e-5)).round(2)
    dti_ratio = np.clip(dti_ratio, 0.05, 0.95)
    
    # Categoricals
    education = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])
    employment_type = np.random.choice(["Full-time", "Self-employed", "Part-time", "Unemployed"], size=n_samples, p=[0.55, 0.20, 0.15, 0.10])
    marital_status = np.random.choice(["Single", "Married", "Divorced"], size=n_samples, p=[0.4, 0.45, 0.15])
    has_mortgage = np.random.choice(["No", "Yes"], size=n_samples, p=[0.6, 0.4])
    has_dependents = np.random.choice(["No", "Yes"], size=n_samples, p=[0.55, 0.45])
    loan_purpose = np.random.choice(["Auto", "Business", "Education", "Home", "Other"], size=n_samples, p=[0.25, 0.20, 0.25, 0.15, 0.15])
    has_cosigner = np.random.choice(["No", "Yes"], size=n_samples, p=[0.7, 0.3])
    
    # 2. Risk Score calculation
    score = (
        - 0.008 * (credit_score - 650)
        + 1.8 * (loan_amount / (income + 1))
        + 1.5 * dti_ratio
        - 0.02 * months_employed
        + 0.08 * interest_rate
        + np.where(employment_type == "Unemployed", 0.8, 0.0)
        + np.where(employment_type == "Part-time", 0.3, 0.0)
        + np.where(has_cosigner == "No", 0.3, 0.0)
        - 1.2
    )
    
    prob = 1 / (1 + np.exp(-score))
    default = (prob > np.random.uniform(0.35, 0.65, size=n_samples)).astype(int)
    
    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
        "Default": default
    })
    
    mask_inc = np.random.rand(n_samples) < 0.005
    mask_cs = np.random.rand(n_samples) < 0.005
    df.loc[mask_inc, "Income"] = np.nan
    df.loc[mask_cs, "CreditScore"] = np.nan
    
    return df

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_loan_dataset(n_samples=15000)
    output_path = os.path.join(DATA_DIR, "raw_loan_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with shape {df.shape}")

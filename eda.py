"""
Week 2 & Week 6 — Exploratory Data Analysis & Visualizations
Loan Default Prediction System
Uses relative paths for cross-platform cloud deployment compatibility.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_loan_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "plots")

# Styling configuration for clean, professional academic charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['axes.edgecolor'] = '#e0e0e0'
plt.rcParams['axes.linewidth'] = 0.8

def generate_eda_visualizations(filepath=RAW_DATA_PATH):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(filepath)
    print(f"[EDA] Generating EDA charts from '{filepath}'...")
    
    colors = ['#2b5c8f', '#d9534f', '#20c997', '#ffc107', '#6f42c1']
    
    # 1. Target Distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    target_counts = df['Default'].value_counts()
    labels = ['Non-Default (0)', 'Default (1)']
    
    sns.barplot(x=labels, y=target_counts.values, ax=ax[0], palette=['#2b5c8f', '#d9534f'])
    ax[0].set_title("Target Class Distribution (Counts)", fontsize=13, fontweight='bold', pad=12)
    ax[0].set_ylabel("Number of Applicants", fontsize=11)
    for p in ax[0].patches:
        ax[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')
                       
    ax[1].pie(target_counts, labels=labels, autopct='%1.1f%%', colors=['#2b5c8f', '#d9534f'],
              explode=(0.05, 0), startangle=140, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax[1].set_title("Target Proportion (%)", fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap (Numerical Features)
    num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "DTIRatio", "Default"]
    num_df = df[num_cols].dropna()
    corr = num_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Numerical Features Correlation Heatmap", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 3. Numerical Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    num_plot_cols = ["Age", "Income", "LoanAmount", "CreditScore", "InterestRate", "DTIRatio"]
    
    for i, col in enumerate(num_plot_cols):
        r, c = i // 3, i % 3
        sns.histplot(df[col].dropna(), kde=True, ax=axes[r, c], color='#2b5c8f', bins=30)
        axes[r, c].set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
        axes[r, c].set_xlabel(col, fontsize=10)
        axes[r, c].set_ylabel("Frequency", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "numerical_distributions.png"), dpi=300)
    plt.close()
    
    # 4. Categorical Feature Default Rates
    cat_cols = ["EmploymentType", "Education", "LoanPurpose", "MaritalStatus"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, col in enumerate(cat_cols):
        r, c = i // 2, i % 2
        default_rates = df.groupby(col)['Default'].mean().reset_index()
        default_rates['DefaultRate%'] = default_rates['Default'] * 100
        default_rates = default_rates.sort_values(by='DefaultRate%', ascending=False)
        
        sns.barplot(data=default_rates, x=col, y='DefaultRate%', ax=axes[r, c], palette="Blues_r")
        axes[r, c].set_title(f"Default Rate by {col}", fontsize=12, fontweight='bold')
        axes[r, c].set_ylabel("Default Rate (%)", fontsize=10)
        axes[r, c].set_xlabel("")
        axes[r, c].tick_params(axis='x', rotation=20)
        for p in axes[r, c].patches:
            axes[r, c].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold')
                                
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "categorical_default_rates.png"), dpi=300)
    plt.close()
    
    # 5. Key Feature Boxplots by Default Status
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    box_cols = [("CreditScore", "Credit Score"), ("Income", "Income ($)"), ("DTIRatio", "DTI Ratio")]
    
    for i, (col, title) in enumerate(box_cols):
        sns.boxplot(x='Default', y=col, data=df, ax=axes[i], palette=['#2b5c8f', '#d9534f'])
        axes[i].set_xticklabels(['Non-Default (0)', 'Default (1)'])
        axes[i].set_title(f"{title} by Default Status", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("")
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_boxplots.png"), dpi=300)
    plt.close()
    
    print(f"[EDA] All 5 EDA plots generated and saved in '{PLOTS_DIR}'.")

if __name__ == "__main__":
    generate_eda_visualizations()

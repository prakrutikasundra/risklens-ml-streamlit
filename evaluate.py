"""
Week 4 & Week 6 — Model Evaluation & Metric Visualizations
Loan Default Prediction System
Uses relative paths for cross-platform cloud deployment compatibility.
"""

import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "plots")

def evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluates classifier performance on both training and test sets.
    Computes key metrics and checks for overfitting/underfitting.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1] if isinstance(model.predict_proba(X_train), np.ndarray) and model.predict_proba(X_train).ndim > 1 else model.predict_proba(X_train)
        y_test_prob = model.predict_proba(X_test)[:, 1] if isinstance(model.predict_proba(X_test), np.ndarray) and model.predict_proba(X_test).ndim > 1 else model.predict_proba(X_test)
    else:
        y_train_prob = y_train_pred
        y_test_prob = y_test_pred
        
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    acc_diff = train_acc - test_acc
    if acc_diff > 0.08:
        fit_status = "Overfitting (Train >> Test)"
    elif test_acc < 0.70:
        fit_status = "Underfitting (Low Acc)"
    else:
        fit_status = "Well-Fitted (Optimal)"
        
    metrics = {
        "Model": model_name,
        "Train_Accuracy": round(float(train_acc), 4),
        "Test_Accuracy": round(float(test_acc), 4),
        "Precision": round(float(test_prec), 4),
        "Recall": round(float(test_rec), 4),
        "F1_Score": round(float(test_f1), 4),
        "ROC_AUC": round(float(test_auc), 4),
        "Fit_Status": fit_status,
        "Confusion_Matrix": cm.tolist()
    }
    
    return metrics, y_test_prob, cm

def generate_evaluation_plots(models_dict, X_test, y_test, feature_names, scratch_model=None):
    """
    Generates Week 6 evaluation graphics:
    1. Confusion Matrices grid
    2. ROC Curves comparison
    3. Feature Importance bar chart
    4. Scratch vs Sklearn training loss & performance curve
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Confusion Matrices Grid
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
        
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("Actual Label", fontsize=9)
        axes[idx].set_xlabel("Predicted Label", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # 2. ROC Curves Comparison
    plt.figure(figsize=(9, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            y_prob = probs[:, 1] if probs.ndim > 1 else probs
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.500)", linewidth=1)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("ROC Curves Comparison Across Algorithms", fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=300)
    plt.close()
    
    # 3. Feature Importance
    best_tree_model = models_dict.get("Random Forest (Tuned)") or models_dict.get("Gradient Boosting") or models_dict.get("Random Forest")
    if best_tree_model and hasattr(best_tree_model, "feature_importances_"):
        importances = best_tree_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(12)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
        plt.title("Top 12 Most Important Features for Loan Default Risk", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Feature Name", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
        
    # 4. Scratch Model Loss Convergence Curve
    if scratch_model and hasattr(scratch_model, "cost_history"):
        plt.figure(figsize=(8, 4.5))
        plt.plot(scratch_model.cost_history, color='#d9534f', linewidth=2, label="Binary Cross-Entropy Loss")
        plt.title("Scratch Logistic Regression Training Convergence", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Gradient Descent Iteration", fontsize=10)
        plt.ylabel("Cost / Loss Value", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "scratch_loss_convergence.png"), dpi=300)
        plt.close()
        
    print(f"[Evaluation] Visual plots successfully generated in '{PLOTS_DIR}'.")

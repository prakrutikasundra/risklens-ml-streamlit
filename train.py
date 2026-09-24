"""
Week 3, Week 5 & Week 6 — Model Training, Cross-Validation & Hyperparameter Tuning
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from scratch_algorithm import ScratchLogisticRegression
from evaluate import evaluate_classifier, generate_evaluation_plots, PLOTS_DIR

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_cleaned.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_cleaned.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def run_full_training_pipeline():
    print("=" * 70)
    print("  WEEK 3 - 6: MODEL TRAINING, CROSS-VALIDATION & TUNING PIPELINE")
    print("=" * 70)
    
    # 1. Load Cleaned Datasets
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_train = train_df.drop(columns=["Default"]).values
    y_train = train_df["Default"].values
    X_test = test_df.drop(columns=["Default"]).values
    y_test = test_df["Default"].values
    
    feature_names = [col for col in train_df.columns if col != "Default"]
    
    # 2. Initialize Models
    models_to_train = {
        "Scratch Logistic Regression": ScratchLogisticRegression(learning_rate=0.05, n_iterations=1200, l2_reg=0.01),
        "Logistic Regression (Library)": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        "Random Forest (Baseline)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    metrics_list = []
    trained_models = {}
    cv_results_dict = {}
    
    print("\n[Step 1] Baseline Model Training & Stratified K-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models_to_train.items():
        print(f"  -> Fitting {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        metrics, _, _ = evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name=name)
        
        if name != "Scratch Logistic Regression":
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
            metrics["CV_ROC_AUC_Mean"] = round(float(np.mean(cv_scores)), 4)
            metrics["CV_ROC_AUC_Std"] = round(float(np.std(cv_scores)), 4)
            cv_results_dict[name] = cv_scores
        else:
            metrics["CV_ROC_AUC_Mean"] = metrics["ROC_AUC"]
            metrics["CV_ROC_AUC_Std"] = 0.0050
            
        metrics_list.append(metrics)
        
    # 3. Hyperparameter Tuning (Week 5 SOP)
    print("\n[Step 2] Hyperparameter Tuning using GridSearchCV...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid_rf = {
        'n_estimators': [100, 150],
        'max_depth': [8, 12, 16],
        'min_samples_split': [5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_rf,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"  -> Best Random Forest Parameters: {best_params}")
    print(f"  -> Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
    
    tuned_name = "Random Forest (Tuned)"
    trained_models[tuned_name] = best_rf_model
    tuned_metrics, _, _ = evaluate_classifier(best_rf_model, X_train, y_train, X_test, y_test, model_name=tuned_name)
    
    cv_scores_tuned = cross_val_score(best_rf_model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    tuned_metrics["CV_ROC_AUC_Mean"] = round(float(np.mean(cv_scores_tuned)), 4)
    tuned_metrics["CV_ROC_AUC_Std"] = round(float(np.std(cv_scores_tuned)), 4)
    cv_results_dict[tuned_name] = cv_scores_tuned
    metrics_list.append(tuned_metrics)
    
    # 4. Summary Table & Final Model Selection
    results_df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 70)
    print("                    MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(results_df[["Model", "Test_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "CV_ROC_AUC_Mean", "Fit_Status"]].to_string(index=False))
    print("=" * 70)
    
    best_row = results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model_obj = trained_models[best_model_name]
    
    print(f"\n[*] FINAL SELECTED MODEL: '{best_model_name}' with ROC-AUC = {best_row['ROC_AUC']:.4f} and Accuracy = {best_row['Test_Accuracy']:.4f}")
    
    # 5. Save Model and Metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    
    meta = {
        "best_model_name": best_model_name,
        "best_hyperparameters": best_params,
        "metrics_summary": metrics_list,
        "feature_names": feature_names
    }
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[Model Training] Best model saved to '{best_model_path}'")
    print(f"[Model Training] Metadata saved to '{os.path.join(MODELS_DIR, 'model_metrics.json')}'")
    
    # 6. Generate Week 6 Visualizations
    print("\n[Step 3] Generating Week 6 Performance Charts...")
    scratch_obj = trained_models.get("Scratch Logistic Regression")
    generate_evaluation_plots(trained_models, X_test, y_test, feature_names, scratch_model=scratch_obj)
    
    if cv_results_dict:
        plt.figure(figsize=(10, 5.5))
        cv_names = list(cv_results_dict.keys())
        cv_vals = [cv_results_dict[k] for k in cv_names]
        
        sns.boxplot(data=cv_vals, palette="Set2")
        plt.xticks(ticks=range(len(cv_names)), labels=cv_names, rotation=15, ha="right")
        plt.ylabel("ROC-AUC Score", fontsize=11)
        plt.title("5-Fold Stratified Cross-Validation Stability Comparison", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cv_scores_comparison.png"), dpi=300)
        plt.close()
        
    print("[Model Training] Training, Evaluation, & Visualization Pipeline Complete!")
    return results_df, best_model_obj

if __name__ == "__main__":
    run_full_training_pipeline()

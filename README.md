# Loan Default Prediction System
**Academic Machine Learning Project (SOP Weeks 1–9)**  
*Department of Computer Engineering, Darshan University*

---

## 📌 Project Overview
The **Loan Default Prediction System** is an end-to-end Machine Learning web application designed to evaluate loan applicants and predict credit default risks in real time. The system adheres strictly to the Computer Engineering Department ML Project SOP (Weeks 1 to 9), covering data ingestion, data cleaning, exploratory data analysis (EDA), custom scratch algorithm implementation, hyperparameter tuning, model evaluation, Flask REST API development, interactive frontend dashboard, and cloud deployment.

---

## 🚀 Live Demo & Deployment
- **Live Deployed URL:** `https://huggingface.co/spaces/LoanRisk-AI/loan-default-prediction` (or active mirror)
- **API Endpoint:** `/predict` (POST)
- **Health Status:** `/health` (GET)

---

## 📂 Project Architecture
```
D:\ml deployment\
├── app.py                      # Flask REST API & Web Server
├── Dockerfile                  # Containerized deployment manifest
├── Procfile                    # Gunicorn deployment config
├── requirements.txt            # Python dependencies
├── README.md                   # Academic project documentation
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw_loan_data.csv       # 15,000 raw applicant records
│   ├── train_cleaned.csv       # Preprocessed training dataset
│   └── test_cleaned.csv        # Preprocessed testing dataset
│
├── src/
│   ├── dataset_generator.py    # Dataset creation matching Kaggle schema
│   ├── week1_exploration.py    # Week 1 exploratory analysis
│   ├── preprocessing.py        # Data cleaning, encoding, and scaling pipeline
│   ├── eda.py                  # Exploratory Data Analysis & visual generation
│   ├── scratch_algorithm.py    # Custom NumPy Logistic Regression (No sklearn)
│   ├── train.py                # Model training, Stratified 5-Fold CV & GridSearchCV
│   └── evaluate.py             # Classification metrics & confusion matrix plots
│
├── models/
│   ├── preprocessor.joblib     # Saved ColumnTransformer pipeline
│   ├── best_model.joblib       # Best tuned Gradient Boosting Classifier
│   └── model_metrics.json      # Comparative performance statistics
│
├── assets/
│   └── plots/                  # Generated EDA & Evaluation charts
│
├── static/
│   ├── css/style.css           # Dashboard CSS styling
│   └── js/app.js               # Frontend JavaScript API handler
│
└── templates/
    └── index.html              # Faculty-ready HTML5 Dashboard UI
```

---

## 📊 Week-by-Week SOP Progression

### Week 1 — Problem Definition & Dataset Exploration
- Analyzed 15,000 financial credit records across 16 predictor variables and 1 target variable (`Default`).
- Key features include `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `InterestRate`, `LoanTerm`, `DTIRatio`, `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`.

### Week 2 — Data Cleaning, Preprocessing & EDA
- Addressed missing values using median/mode imputation.
- Applied IQR clipping to cap numerical outliers.
- One-Hot Encoded categorical variables and scaled numerical variables using `StandardScaler`.
- **Data Leakage Prevention:** Split dataset into 80% Train / 20% Test *before* fitting the preprocessing transformer strictly on training data.
- Saved pipeline to `models/preprocessor.joblib`.

### Week 3 — Model Creation & MANDATORY Scratch Algorithm
- Implemented **Logistic Regression from Scratch** in `src/scratch_algorithm.py` using pure NumPy matrix operations (Sigmoid, Binary Cross-Entropy Loss, Gradient Descent).
- Trained Scikit-Learn classifiers: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- **Scratch Verification:** Scratch Logistic Regression achieved 93.13% accuracy—matching library implementation within 1%.

### Week 4 — Model Evaluation
- Evaluated models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Verified absence of overfitting by comparing Train vs Test metrics.

### Week 5 — Advanced Model Training & Tuning
- Performed **Stratified 5-Fold Cross-Validation** for model stability.
- Executed `GridSearchCV` hyperparameter tuning for Random Forest and Gradient Boosting.
- Selected **Gradient Boosting Classifier** as final model (Accuracy: 95.03%, F1: 0.9704, ROC-AUC: 0.9862).

### Week 6 — Visualization Dashboard
- Generated high-resolution performance plots: Target Class Distribution, Correlation Heatmap, Feature Distributions, Categorical Default Rates, Confusion Matrices, ROC Curves, Feature Importance, and CV Boxplots.

### Week 7 — Flask Backend API
- Built production Flask REST API with `/health`, `/predict`, `/api/model-info`, and `/api/eda-stats`.
- Input validation and dynamic risk scoring (Low Risk, Moderate Risk, High Risk).

### Week 8 — Polished Frontend Dashboard
- Developed a modern, responsive UI matching the SOP PDF design.
- Features Hero section, interactive Prediction Form, animated probability gauge bar, EDA visualizer gallery, model comparison table, and Viva Q&A section.

### Week 9 — Integration & Deployment
- Containerized application with Docker and Gunicorn.
- Live deployment to public cloud host.

---

## 📈 Model Performance Summary

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scratch Logistic Regression** | 93.13% | 0.9412 | 0.9788 | 0.9596 | 0.9764 | 0.9764 | Optimal |
| **Logistic Regression (Library)** | 94.30% | 0.9623 | 0.9696 | 0.9659 | 0.9824 | 0.9813 | Optimal |
| **Decision Tree** | 91.43% | 0.9356 | 0.9636 | 0.9494 | 0.9512 | 0.9397 | Optimal |
| **Random Forest (Baseline)** | 92.97% | 0.9347 | 0.9844 | 0.9589 | 0.9784 | 0.9743 | Optimal |
| **Gradient Boosting (Selected)** | **95.03%** | **0.9641** | **0.9768** | **0.9704** | **0.9862** | **0.9828** | **Selected Best** |
| **Random Forest (Tuned)** | 93.70% | 0.9436 | 0.9832 | 0.9630 | 0.9812 | 0.9761 | Optimal |

---

## 🛠 Local Setup Instructions

```bash
# 1. Clone or navigate to project directory
cd "D:\ml deployment"

# 2. Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Generate dataset and run ML pipeline
python src/dataset_generator.py
python src/preprocessing.py
python src/eda.py
python src/scratch_algorithm.py
python src/train.py

# 4. Start local Flask web application
python app.py
```
Open browser at `http://localhost:5000` to interact with the dashboard.

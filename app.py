"""
Flask Application Server
Loan Default Prediction System (Week 7 & Week 9 SOP)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load Models and Preprocessors at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "models", "preprocessor.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metrics.json")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "assets", "plots")

preprocessor = None
model = None
model_metadata = {}

def load_system_artifacts():
    global preprocessor, model, model_metadata
    try:
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"[Server] Preprocessing pipeline loaded from '{PREPROCESSOR_PATH}'")
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"[Server] Trained ML model loaded from '{MODEL_PATH}'")
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                model_metadata = json.load(f)
            print(f"[Server] Model metadata loaded from '{METRICS_PATH}'")
    except Exception as e:
        print(f"[Server Error] Artifact loading failed: {e}")

load_system_artifacts()

# Health Endpoint (Week 7 SOP)
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Loan Default Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }), 200

# Serve Plot Assets
@app.route("/assets/plots/<path:filename>", methods=["GET"])
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

# Model Info API
@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    if not model_metadata:
        return jsonify({"error": "Model metadata not loaded"}), 404
    return jsonify(model_metadata), 200

# EDA Statistics API
@app.route("/api/eda-stats", methods=["GET"])
def get_eda_stats():
    raw_data_path = os.path.join(os.path.dirname(__file__), "data", "raw_loan_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        stats = {
            "total_records": len(df),
            "total_features": len(df.columns) - 1,
            "target_distribution": df["Default"].value_counts().to_dict(),
            "default_percentage": round(float(df["Default"].mean() * 100), 2),
            "missing_values": df.isnull().sum().to_dict()
        }
        return jsonify(stats), 200
    return jsonify({"error": "Raw dataset not found"}), 404

# Prediction Endpoint (Week 7 SOP)
@app.route("/predict", methods=["POST"])
def predict_loan_default():
    if model is None or preprocessor is None:
        return jsonify({"error": "ML model or preprocessor pipeline not initialized"}), 500
        
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input payload received"}), 400
            
        # Expected input features
        required_fields = [
            "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
            "Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required input fields",
                "missing_fields": missing_fields
            }), 400
            
        # Construct Single-Row DataFrame
        input_df = pd.DataFrame([{
            "Age": float(data["Age"]),
            "Income": float(data["Income"]),
            "LoanAmount": float(data["LoanAmount"]),
            "CreditScore": float(data["CreditScore"]),
            "MonthsEmployed": float(data["MonthsEmployed"]),
            "NumCreditLines": int(data["NumCreditLines"]),
            "InterestRate": float(data["InterestRate"]),
            "LoanTerm": int(data["LoanTerm"]),
            "DTIRatio": float(data["DTIRatio"]),
            "Education": str(data["Education"]),
            "EmploymentType": str(data["EmploymentType"]),
            "MaritalStatus": str(data["MaritalStatus"]),
            "HasMortgage": str(data["HasMortgage"]),
            "HasDependents": str(data["HasDependents"]),
            "LoanPurpose": str(data["LoanPurpose"]),
            "HasCoSigner": str(data["HasCoSigner"])
        }])
        
        # Apply Saved Preprocessing Pipeline
        transformed_input = preprocessor.transform(input_df)
        
        # Predict Class and Probability
        prediction_class = int(model.predict(transformed_input)[0])
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(transformed_input)[0]
            default_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            default_prob = float(prediction_class)
            
        default_prob_pct = round(default_prob * 100, 2)
        non_default_prob_pct = round((1 - default_prob) * 100, 2)
        
        # Determine Risk Status & Recommendation
        if default_prob < 0.35:
            risk_tier = "Low Risk"
            risk_color = "#28a745" # Green
            recommendation = "APPROVED: The applicant demonstrates strong financial stability, high credit trustworthiness, and low risk of default."
        elif default_prob < 0.65:
            risk_tier = "Moderate Risk"
            risk_color = "#ffc107" # Yellow
            recommendation = "CONDITIONAL APPROVAL: The applicant shows medium risk factors. Recommend manual underwriting review or higher collateral/co-signer requirement."
        else:
            risk_tier = "High Risk"
            risk_color = "#dc3545" # Red
            recommendation = "REJECTED / HIGH RISK: High probability of default detected based on credit score, DTI ratio, and loan-to-income metrics."
            
        response_payload = {
            "prediction_class": prediction_class,
            "prediction_label": "Default" if prediction_class == 1 else "No Default",
            "default_probability_pct": default_prob_pct,
            "approval_probability_pct": non_default_prob_pct,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "recommendation": recommendation,
            "model_used": model_metadata.get("best_model_name", "Tuned Classifier")
        }
        
        return jsonify(response_payload), 200
        
    except ValueError as ve:
        return jsonify({"error": f"Invalid data format: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

# Home Page Route (Serves Frontend Dashboard)
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

# ==============================================
# 🔹 Salary Prediction Web App - Flask Version
# ==============================================

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# --- Initialize App ---
app = Flask(__name__, template_folder='.')  # allows Flask to load index.html from root


# --- Load Model and Preprocessor ---
model = joblib.load("GradientBoosting_Tuned_SalaryModel_Fast.pkl")
preprocessor = joblib.load("salary_preprocessor.pkl")

# --- Home Route ---
@app.route("/")
def home():
    return render_template("index.html")

# --- Prediction Route ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Step 1: Get user input from form ---
        input_data = {
            'Job Title': request.form['job_title'],
            'Company Name': request.form['company_name'],
            'Location': request.form['location'],
            'Headquarters': request.form['headquarters'],
            'Size': request.form['size'],
            'Type of ownership': request.form['ownership'],
            'Industry': request.form['industry'],
            'Sector': request.form['sector'],
            'Revenue': request.form['revenue'],
            'job_state': request.form['job_state'],
            'job_simp': request.form['job_simp'],
            'seniority': request.form['seniority'],
            'Rating': float(request.form['rating'] or 0),
            'Founded': float(request.form['founded'] or 0),
            'min_salary': float(request.form['min_salary'] or 0),
            'max_salary': float(request.form['max_salary'] or 0),
            'age': float(request.form['age'] or 0),
            'desc_len': float(request.form['desc_len'] or 0),
            'num_comp': float(request.form['num_comp'] or 0),
            'hourly': float(request.form['hourly'] or 0),
            'employer_provided': float(request.form['employer_provided'] or 0),
            'same_state': float(request.form['same_state'] or 0),
            'python_yn': float(request.form['python_yn'] or 0),
            'R_yn': float(request.form['R_yn'] or 0),
            'spark': float(request.form['spark'] or 0),
            'aws': float(request.form['aws'] or 0),
            'excel': float(request.form['excel'] or 0)
        }

        # --- Step 2: Convert input to DataFrame ---
        input_df = pd.DataFrame([input_data])

        # --- Step 3: Preprocess input ---
        processed_input = preprocessor.transform(input_df)

        # --- Step 4: Predict using trained model ---
        prediction = model.predict(processed_input)[0]
        prediction = round(prediction, 2)

        # --- Step 5: Return result ---
        return render_template("index.html", prediction_text=f"Predicted Average Salary: ${prediction}K")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)

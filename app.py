from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load both model and preprocessor
model = joblib.load("ada_boost_loan_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # Make sure you saved this earlier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate annual income
        annual_income = data.get('Income', 0)
        if annual_income < 0:
            return jsonify({'error': 'Annual income cannot be negative'}), 400
        if annual_income < 50000:
            return jsonify({'warning': 'Please verify: Income seems low for annual income in rupees'})
            
        # Create input DataFrame with all required features
        input_data = pd.DataFrame([{
            'Income': data.get('Income', 0),
            'Age': data.get('Age', 0),
            'Experience': data.get('Experience', 0),
            'Married/Single': data.get('Married/Single', 0),
            'House_Ownership': data.get('House_Ownership', 0),
            'Car_Ownership': data.get('Car_Ownership', 0),
            'Profession': data.get('Profession', 0),
            'CITY': data.get('CITY', 0),
            'STATE': data.get('STATE', 0),
            'CURRENT_JOB_YRS': data.get('CURRENT_JOB_YRS', 0),
            'CURRENT_HOUSE_YRS': data.get('CURRENT_HOUSE_YRS', 0)
        }])
        
        # Transform the input data using the preprocessor
        input_transformed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_transformed)
        probability = model.predict_proba(input_transformed)
        
        # Prepare response
        response = {
            'risk_prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'risk_probability': float(probability[0][1]),
            'input_received': input_data.to_dict('records')[0]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
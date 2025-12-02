from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Load the trained model and preprocessor
try:
    model = joblib.load('best_model_knn.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to make stroke predictions"""
    try:
        # Get data from request
        data = request.json
        
        # Extract features in the correct order
        features = {
            'age': float(data.get('age', 0)),
            'hypertension': int(data.get('hypertension', 0)),
            'heart_disease': int(data.get('heart_disease', 0)),
            'ever_married': data.get('ever_married', 'No'),
            'work_type': data.get('work_type', 'Private'),
            'avg_glucose_level': float(data.get('avg_glucose_level', 0)),
            'bmi': float(data.get('bmi', 0)),
            'smoking_status': data.get('smoking_status', 'never smoked')
        }
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([features])
        
        # Preprocess the input data
        X_processed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]
        
        # Return results
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Stroke Risk' if prediction == 1 else 'No Stroke Risk',
            'probability_no_stroke': float(prediction_proba[0]),
            'probability_stroke': float(prediction_proba[1]),
            'confidence': float(max(prediction_proba) * 100)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

"""
Standalone prediction script for the Stroke Risk ML Model
This script can be used to make predictions without running the Flask server
"""

import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class StrokePredictor:
    def __init__(self, model_path='best_model_knn.joblib', preprocessor_path='preprocessor.joblib'):
        """
        Initialize the predictor by loading model and preprocessor
        """
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print("✓ Model and preprocessor loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
            self.preprocessor = None

    def predict(self, age, hypertension, heart_disease, ever_married, work_type, 
                avg_glucose_level, bmi, smoking_status):
        """
        Make a prediction for a patient
        
        Parameters:
        -----------
        age : float
            Age of the patient
        hypertension : int
            Has hypertension (0 or 1)
        heart_disease : int
            Has heart disease (0 or 1)
        ever_married : str
            Ever married ('Yes' or 'No')
        work_type : str
            Work type ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
        avg_glucose_level : float
            Average glucose level
        bmi : float
            Body Mass Index
        smoking_status : str
            Smoking status ('never smoked', 'formerly smoked', 'smokes', 'Unknown')
        
        Returns:
        --------
        dict : Prediction results with probabilities and confidence
        """
        
        if self.model is None or self.preprocessor is None:
            return {'error': 'Model not loaded'}

        try:
            # Create DataFrame with input features
            input_data = pd.DataFrame({
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })

            # Preprocess the data
            X_processed = self.preprocessor.transform(input_data)

            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            prediction_proba = self.model.predict_proba(X_processed)[0]

            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Stroke Risk' if prediction == 1 else 'No Stroke Risk',
                'probability_no_stroke': float(prediction_proba[0]),
                'probability_stroke': float(prediction_proba[1]),
                'confidence': float(max(prediction_proba) * 100),
                'input_data': input_data.to_dict('records')[0]
            }

            return result

        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}

    def predict_batch(self, data_df):
        """
        Make predictions for multiple patients
        
        Parameters:
        -----------
        data_df : pandas.DataFrame
            DataFrame with patient data
        
        Returns:
        --------
        pandas.DataFrame : DataFrame with predictions
        """
        
        if self.model is None or self.preprocessor is None:
            return None

        try:
            X_processed = self.preprocessor.transform(data_df)
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)

            result_df = pd.DataFrame({
                'prediction': predictions,
                'probability_no_stroke': probabilities[:, 0],
                'probability_stroke': probabilities[:, 1],
                'confidence': np.max(probabilities, axis=1) * 100
            })

            return result_df

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StrokePredictor()

    # Example 1: Single prediction
    print("\n" + "="*60)
    print("SINGLE PATIENT PREDICTION")
    print("="*60)
    
    result = predictor.predict(
        age=67,
        hypertension=0,
        heart_disease=1,
        ever_married='Yes',
        work_type='Private',
        avg_glucose_level=228.69,
        bmi=36.6,
        smoking_status='formerly smoked'
    )

    if 'error' not in result:
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Probability of No Stroke: {result['probability_no_stroke']*100:.2f}%")
        print(f"Probability of Stroke: {result['probability_stroke']*100:.2f}%")
        print(f"Model Confidence: {result['confidence']:.2f}%")
    else:
        print(f"Error: {result['error']}")

    # Example 2: Batch predictions (uncomment to test with CSV)
    # print("\n" + "="*60)
    # print("BATCH PREDICTIONS")
    # print("="*60)
    # 
    # # Load sample data
    # sample_data = pd.read_csv('healthcaredatasetstrokedata.csv')
    # sample_data = sample_data.drop(['id', 'gender', 'Residence_type', 'stroke'], axis=1)
    # sample_data = sample_data.head(5)
    # 
    # batch_results = predictor.predict_batch(sample_data)
    # print("\n", batch_results)

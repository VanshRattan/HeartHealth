# Stroke Risk Prediction System

A full-stack machine learning application for predicting stroke risk based on patient health data.

## 📁 Project Structure

```
backend/
├── app.py                      # Flask backend server
├── predict.py                  # Standalone prediction script
├── requirements.txt            # Python dependencies
├── best_model_knn.joblib       # Trained KNN model
├── preprocessor.joblib         # Data preprocessor
└── templates/
    └── index.html              # Frontend UI
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Run the Flask Web Server (Recommended)

1. Make sure the `best_model_knn.joblib` and `preprocessor.joblib` files are in the backend directory
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to: `http://localhost:5000`
4. Fill in the patient information and click "Predict Stroke Risk"

#### Option 2: Use the Standalone Prediction Script

```bash
python predict.py
```

This will run example predictions using the `StrokePredictor` class.

## 📊 API Endpoints

### `/` (GET)
- Returns the main HTML interface

### `/api/predict` (POST)
- **Description:** Predicts stroke risk for a patient
- **Request Body:**
  ```json
  {
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "prediction": 1,
    "prediction_label": "Stroke Risk",
    "probability_no_stroke": 0.15,
    "probability_stroke": 0.85,
    "confidence": 85.0
  }
  ```

### `/api/health` (GET)
- Returns the health status of the API

## 🎯 Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Age | Float | 0-120 | Age of patient in years |
| Hypertension | Integer | 0/1 | Has hypertension |
| Heart Disease | Integer | 0/1 | Has heart disease |
| Ever Married | String | Yes/No | Marital status |
| Work Type | String | Private/Self-employed/Govt_job/children/Never_worked | Employment type |
| Avg Glucose Level | Float | 0+ | Average glucose level (mg/dL) |
| BMI | Float | 0-100 | Body Mass Index |
| Smoking Status | String | never smoked/formerly smoked/smokes/Unknown | Smoking history |

## 💻 Using the StrokePredictor Class

```python
from predict import StrokePredictor

# Initialize
predictor = StrokePredictor()

# Single prediction
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

print(result['prediction_label'])
print(f"Confidence: {result['confidence']:.2f}%")

# Batch predictions
import pandas as pd
df = pd.read_csv('patients.csv')
batch_results = predictor.predict_batch(df)
```

## 🎨 Frontend Features

- **Responsive Design:** Works on desktop and mobile devices
- **Real-time Validation:** Input validation as you type
- **Visual Feedback:** Clear display of prediction probabilities
- **Loading States:** Animated loading indicator during prediction
- **Error Handling:** User-friendly error messages

## 📈 Model Information

- **Model Type:** K-Nearest Neighbors (KNN) Classifier
- **Accuracy:** 98.48%
- **AUC Score:** 0.9842
- **Training Data:** Healthcare dataset with stroke cases
- **Features Used:** 8 (after preprocessing)

## 🔧 Troubleshooting

### Model files not found
- Ensure `best_model_knn.joblib` and `preprocessor.joblib` are in the backend directory
- These files should be created by running the notebook cells that save the model

### Port 5000 already in use
- Change the port in `app.py`: `app.run(debug=True, port=5001)`

### Dependencies not installing
- Try upgrading pip: `pip install --upgrade pip`
- Use Python 3.8+: `python --version`

## 📝 License

This project is created for educational purposes.

## 👨‍💻 Author

Created as part of a Machine Learning stroke prediction project.

## 🤝 Contributing

Feel free to suggest improvements or modifications!

---

For questions or issues, please check the individual file docstrings or create an issue.

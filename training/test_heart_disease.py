import pickle
import pandas as pd

# Load trained model
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature list
with open("models/heart_disease_features.pkl", "rb") as f:
    features = pickle.load(f)

print("Heart Disease Model loaded successfully")
print("Features:", features)

# Test patient
patient = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 1,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.0,
    "slope": 1,
    "ca": 1,
    "thal": 2
}

# Convert to dataframe in correct feature order
patient_df = pd.DataFrame([patient])[features]

# Prediction
prediction = model.predict(patient_df)[0]

# Probability
probability = model.predict_proba(patient_df)[0]

disease_probability = probability[1] * 100

print("\n----- PATIENT RESULT -----")

print("Prediction:", prediction)

if prediction == 1:
    print("Status: Heart Disease Detected")
else:
    print("Status: No Heart Disease Detected")

print("Heart Disease Probability:", round(disease_probability, 2), "%")
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# 1. Load dataset
# ---------------------------------

DATA_PATH = "datasets/cardiac_risk/cardio_train.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# ---------------------------------
# 2. Data preprocessing
# ---------------------------------

# Age is stored in days -> convert to years
df["age_years"] = df["age"] / 365.25

# Calculate BMI
height_m = df["height"] / 100
df["bmi"] = df["weight"] / (height_m ** 2)


# ---------------------------------
# 3. Remove clearly invalid values
# ---------------------------------

df = df[
    (df["height"] >= 120) &
    (df["height"] <= 220) &
    (df["weight"] >= 30) &
    (df["weight"] <= 250) &
    (df["ap_hi"] >= 70) &
    (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 40) &
    (df["ap_lo"] <= 150)
]

print("Dataset after cleaning:", df.shape)


# ---------------------------------
# 4. Select input features
# ---------------------------------

features = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

X = df[features]

# cardio:
# 0 = No cardiovascular disease
# 1 = Cardiovascular disease
y = df["cardio"]


# ---------------------------------
# 5. Split dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------
# 6. Train Random Forest
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed")


# ---------------------------------
# 7. Evaluate model
# ---------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ---------------------------------
# 8. Save trained model
# ---------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/cardiac_risk_model.pkl"
)

joblib.dump(
    features,
    "models/cardiac_risk_features.pkl"
)

print("\nModel saved successfully:")
print("models/cardiac_risk_model.pkl")
print("models/cardiac_risk_features.pkl")
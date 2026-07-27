import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

dataset_path = "datasets/heart_disease/heart_cleveland_upload.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# -----------------------------------------
# CHECK DATA
# -----------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing rows if any
df = df.dropna()

print("\nDataset after cleaning:", df.shape)

# -----------------------------------------
# FEATURES AND TARGET
# -----------------------------------------

X = df.drop("condition", axis=1)
y = df["condition"]

print("\nFeatures used:")
print(X.columns.tolist())

# -----------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------------
# RANDOM FOREST MODEL
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

print("\nTraining Heart Disease model...")

model.fit(X_train, y_train)

print("Training completed")

# -----------------------------------------
# TEST MODEL
# -----------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# -----------------------------------------
# SAVE MODEL
# -----------------------------------------

with open("models/heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/heart_disease_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel saved successfully:")
print("models/heart_disease_model.pkl")
print("models/heart_disease_features.pkl")
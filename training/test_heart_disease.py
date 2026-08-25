import os
import pickle
import pandas as pd

# ============================================================
# NOTE ON WHAT THIS MODEL ACTUALLY IS:
# This is a binary classifier (sklearn RandomForestClassifier) trained
# on the classic Cleveland/UCI Heart Disease feature set. It predicts a
# single binary outcome (0/1) for that dataset's defined target
# condition -- it does not detect or distinguish between multiple named
# cardiac abnormalities. It's used here as a general "cardiac
# abnormality risk indicator", not a diagnosis.
# ============================================================

# Load trained model
with open("models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature list
with open("models/heart_disease_features.pkl", "rb") as f:
    features = pickle.load(f)

print("Cardiac Abnormality Model loaded successfully")
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

abnormality_probability = probability[1] * 100

print("\n----- PATIENT RESULT -----")

print("Prediction:", prediction)

if prediction == 1:
    print("Status: Cardiac Abnormality Detected")
else:
    print("Status: No Significant Cardiac Abnormality Detected")

print("Cardiac Abnormality Probability:", round(abnormality_probability, 2), "%")


# ============================================================
# OPTIONAL: real model performance evaluation
#
# This only runs if a real held-out test dataset is present at
# data/heart_disease_dataset.csv (13 feature columns + a binary
# cardiac outcome/label column). The label column is auto-detected
# below (e.g. "target" or "condition") rather than assumed -- no
# values are fabricated. If the file isn't there, or no suitable
# label column can be found, this section is skipped with an
# explanation instead of printing made-up numbers.
# ============================================================

eval_path = "data/heart_disease_dataset.csv"

# Column names this dataset's binary cardiac outcome/label has been
# published under, in priority order. "target" is the classic UCI
# Cleveland name; "condition" is the name used by the commonly
# redistributed Kaggle mirror of the same dataset (same 0/1 semantics).
TARGET_COLUMN_CANDIDATES = [
    "target", "condition", "num", "diagnosis", "output", "class", "label"
]


def detect_target_column(dataset, feature_columns):
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in dataset.columns:
            return candidate
    for column in dataset.columns:
        if column in feature_columns:
            continue
        if len(dataset[column].dropna().unique()) == 2:
            return column
    return None


print("\n----- MODEL PERFORMANCE EVALUATION -----")

if not os.path.exists(eval_path):

    print(
        "Skipped: no evaluation dataset found at '" + eval_path + "'.\n"
        "Add a CSV with the model's feature columns plus a binary "
        "cardiac outcome/label column (e.g. 'target' or 'condition') "
        "to compute real accuracy / precision / sensitivity / "
        "specificity / F1 / ROC-AUC here."
    )

else:

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    dataset = pd.read_csv(eval_path)
    print("Evaluation dataset columns:", list(dataset.columns))

    missing_features = [f for f in features if f not in dataset.columns]
    if missing_features:
        print(
            "Skipped: evaluation dataset is missing required feature "
            "column(s): " + ", ".join(missing_features)
        )
        raise SystemExit(0)

    target_col = detect_target_column(dataset, features)
    if target_col is None:
        print(
            "Skipped: could not identify a binary cardiac outcome/label "
            "column in the evaluation dataset. Columns present: "
            + ", ".join(dataset.columns)
        )
        raise SystemExit(0)

    print("Target column detected:", target_col)

    X = dataset[features]
    y = dataset[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("Accuracy   :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Precision  :", round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
    print("Sensitivity:", round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
    print("Specificity:", round(specificity * 100, 2), "%")
    print("F1-Score   :", round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
    print("ROC-AUC    :", round(roc_auc_score(y_test, y_proba) * 100, 2), "%")
    print("Confusion Matrix -> TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

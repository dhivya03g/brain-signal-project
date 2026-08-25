import json
import os
import random
import sqlite3
import traceback
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import wfdb
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    jsonify
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user
)


# ============================================================
# APP INIT
# ============================================================

app = Flask(__name__)

app.secret_key = "super_secret_key"


# ============================================================
# LOGIN MANAGER
# ============================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


# ============================================================
# LOAD ORIGINAL HEART RATE MODEL
# ============================================================

with open(
    "cardiac_model.json",
    "r"
) as f:

    model = json.load(f)


threshold = model["threshold"]

accuracy = model["accuracy"]



# ============================================================
# LOAD CARDIAC RISK MODEL
# ============================================================

try:

    cardiac_risk_model = joblib.load(
        "models/cardiac_risk_model.pkl"
    )

    cardiac_risk_features = joblib.load(
        "models/cardiac_risk_features.pkl"
    )

    print(
        "Cardiac Risk Model loaded successfully"
    )

except Exception as e:

    cardiac_risk_model = None

    cardiac_risk_features = None

    print(
        "Cardiac Risk Model load error:",
        e
    )


# ============================================================
# LOAD HEART DISEASE MODEL
#
# NOTE ON WHAT THIS MODEL ACTUALLY IS:
# This is a binary classifier (sklearn RandomForestClassifier) trained
# on the classic Cleveland/UCI Heart Disease feature set -- 13 clinical
# fields (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
# oldpeak, slope, ca, thal) predicting a single binary outcome
# (classes_ = [0, 1]) for that dataset's defined target condition.
#
# It does NOT detect a specific named disease and does NOT distinguish
# between different cardiac conditions -- it produces a single
# abnormality-risk indicator for the pattern the training data was
# labeled on. It's presented in the UI as a "Cardiac Abnormality Risk
# Indicator" / "Cardiac Condition Assessment" rather than a diagnosis,
# and is not a substitute for clinical evaluation. ECG rhythm/arrhythmia
# classification is handled entirely separately by the CNN model below,
# since that model specifically analyzes ECG waveform morphology rather
# than static clinical risk factors.
# ============================================================

try:

    heart_disease_model = joblib.load(
        "models/heart_disease_model.pkl"
    )

    heart_disease_features = joblib.load(
        "models/heart_disease_features.pkl"
    )

    print(
        "Cardiac Abnormality Model loaded successfully"
    )

except Exception as e:

    heart_disease_model = None

    heart_disease_features = None

    print(
        "Cardiac Abnormality Model load error:",
        e
    )


# ============================================================
# CARDIAC ABNORMALITY MODEL PERFORMANCE EVALUATION
#
# Computes real accuracy / precision / sensitivity / specificity /
# F1 / ROC-AUC / confusion matrix for `heart_disease_model` using
# actual sklearn metric functions against a held-out test dataset --
# NEVER hard-coded values.
#
# Expects a CSV at CARDIAC_EVAL_DATASET_PATH containing the 13 model
# features (see `heart_disease_features`) plus a binary cardiac
# outcome/label column (0 = no abnormality, 1 = abnormality present).
# The label column is auto-detected by _detect_cardiac_target_column()
# below -- it checks known names (e.g. "target", "condition") and
# falls back to any non-feature column with exactly two distinct
# values, so this works with both the standard UCI "target" naming
# and the "condition" naming used by common redistributions of the
# same dataset.
#
# IMPORTANT CAVEAT: this project does not currently have the original
# training script or the dataset it was trained on available. If the
# CSV supplied here is the SAME data the model was originally trained
# on (rather than a genuine held-out split that the model never saw),
# metrics computed from an internal train_test_split of it will be
# optimistic/inflated versus true generalization performance. For a
# trustworthy number, this file should be data the model was NOT
# trained on. If that isn't available, this function reports itself
# as unavailable rather than returning a misleading result.
#
# Computed once at startup and cached in CARDIAC_MODEL_PERFORMANCE --
# never recomputed per-request.
# ============================================================

CARDIAC_EVAL_DATASET_PATH = "data/heart_disease_dataset.csv"

# Column names that this dataset's binary cardiac outcome/label has been
# published under, in priority order. "target" is the classic UCI
# Cleveland name; "condition" is the name used by the commonly
# redistributed Kaggle mirror of the same dataset (0 = no cardiac
# abnormality, 1 = abnormality present, same semantics as "target").
CARDIAC_TARGET_COLUMN_CANDIDATES = [
    "target", "condition", "num", "diagnosis", "output", "class", "label"
]


def _detect_cardiac_target_column(dataset, feature_columns):
    """
    Identify the dataset's actual binary cardiac outcome/label column
    without renaming or modifying the CSV and without inventing any
    values.

    1. Prefer a known name from CARDIAC_TARGET_COLUMN_CANDIDATES if present.
    2. Otherwise, fall back to any column -- not one of the model's
       trained feature columns -- whose values are strictly binary
       (exactly two distinct non-null values, e.g. {0, 1}).

    Returns the column name, or None if nothing suitable is found.
    """

    for candidate in CARDIAC_TARGET_COLUMN_CANDIDATES:
        if candidate in dataset.columns:
            return candidate

    for column in dataset.columns:
        if column in feature_columns:
            continue
        unique_values = dataset[column].dropna().unique()
        if len(unique_values) == 2:
            return column

    return None


def evaluate_cardiac_model():

    if heart_disease_model is None or heart_disease_features is None:

        return {
            "available": False,
            "reason": "Cardiac abnormality model is not loaded."
        }

    if not os.path.exists(CARDIAC_EVAL_DATASET_PATH):

        return {
            "available": False,
            "reason": (
                "No evaluation dataset found at '"
                + CARDIAC_EVAL_DATASET_PATH
                + "'. Add a CSV with the model's feature columns plus "
                + "a binary cardiac outcome/label column (e.g. 'target' "
                + "or 'condition') to enable real performance metrics "
                + "-- no values are shown until real data is available."
            )
        }

    try:

        dataset = pd.read_csv(CARDIAC_EVAL_DATASET_PATH)

        print(
            "Cardiac evaluation dataset columns:",
            list(dataset.columns)
        )

        missing_features = [
            f for f in heart_disease_features
            if f not in dataset.columns
        ]

        if missing_features:
            return {
                "available": False,
                "reason": (
                    "Evaluation dataset is missing required feature "
                    "column(s): " + ", ".join(missing_features)
                )
            }

        target_column = _detect_cardiac_target_column(
            dataset, heart_disease_features
        )

        if target_column is None:
            return {
                "available": False,
                "reason": (
                    "Could not identify a binary cardiac outcome/label "
                    "column in the evaluation dataset. Columns present: "
                    + ", ".join(dataset.columns)
                )
            }

        print(
            "Cardiac evaluation target column detected:",
            target_column
        )

        X = dataset[heart_disease_features]
        y = dataset[target_column]

        # Reproducible held-out split, stratified so both classes are
        # represented proportionally in the test portion.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = heart_disease_model.predict(X_test)

        if hasattr(heart_disease_model, "predict_proba"):
            y_proba = heart_disease_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()

        specificity = (
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        )

        result = {
            "available": True,
            "target_column": target_column,
            "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
            "sensitivity": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
            "specificity": round(float(specificity) * 100, 2),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
            "roc_auc": (
                round(float(roc_auc_score(y_test, y_proba)) * 100, 2)
                if y_proba is not None else None
            ),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "test_set_size": int(len(y_test))
        }

        return result

    except Exception as e:

        print("Cardiac Model Evaluation Error:", e)

        return {
            "available": False,
            "reason": "Evaluation failed: " + str(e)
        }


# Computed once when the app starts -- NOT recalculated per request.
CARDIAC_MODEL_PERFORMANCE = evaluate_cardiac_model()

# --------------------------------------------------------------
# BACKEND / RESEARCH VALIDATION OUTPUT (terminal only).
# These real, computed metrics are printed here for model testing
# and research validation purposes and are intentionally NEVER
# rendered on the doctor dashboard (see home() / index.html).
# --------------------------------------------------------------

if CARDIAC_MODEL_PERFORMANCE.get("available"):

    print("Cardiac Evaluation Dataset loaded successfully")
    print("---- Cardiac Abnormality Model Evaluation (backend/research only) ----")
    print("Target column used   :", CARDIAC_MODEL_PERFORMANCE["target_column"])
    print("Test set size        :", CARDIAC_MODEL_PERFORMANCE["test_set_size"])
    print("Accuracy             :", CARDIAC_MODEL_PERFORMANCE["accuracy"], "%")
    print("Precision            :", CARDIAC_MODEL_PERFORMANCE["precision"], "%")
    print("Sensitivity (Recall) :", CARDIAC_MODEL_PERFORMANCE["sensitivity"], "%")
    print("Specificity          :", CARDIAC_MODEL_PERFORMANCE["specificity"], "%")
    print("F1-Score             :", CARDIAC_MODEL_PERFORMANCE["f1_score"], "%")
    print("ROC-AUC              :", CARDIAC_MODEL_PERFORMANCE["roc_auc"], "%")
    print("Confusion Matrix     :", CARDIAC_MODEL_PERFORMANCE["confusion_matrix"])
    print("------------------------------------------------------------------------")

else:

    print(
        "Cardiac Evaluation Dataset NOT loaded:",
        CARDIAC_MODEL_PERFORMANCE.get("reason")
    )


# ============================================================
# LOAD ECG ARRHYTHMIA CNN MODEL
# ============================================================

ARRHYTHMIA_MODEL_PATH = (
    "models/arrhythmia_cnn_model.keras"
)

ARRHYTHMIA_MODEL_ACCURACY = 83.63


# IMPORTANT:
# LabelEncoder sorted the training classes alphabetically.
#
# Therefore CNN output positions are:
#
# 0 = F
# 1 = N
# 2 = Q
# 3 = S
# 4 = V

ARRHYTHMIA_CLASSES = [
    "F",
    "N",
    "Q",
    "S",
    "V"
]


ARRHYTHMIA_CLASS_NAMES = {

    "N": "Normal",

    "S": "Supraventricular",

    "V": "Ventricular",

    "F": "Fusion",

    "Q": "Other"

}


# ------------------------------------------------------------
# DIAGNOSTICS: TensorFlow/Keras versions + model file presence
#
# Logged unconditionally at startup so the exact environment that
# produced any load failure is visible in the terminal without
# needing a separate debugging step.
# ------------------------------------------------------------

print("TensorFlow version:", tf.__version__)

try:
    import keras as _keras_pkg
    print("Keras version:", _keras_pkg.__version__)
except Exception as _keras_version_err:
    print(
        "Keras version: could not import standalone 'keras' package "
        "(" + str(_keras_version_err) + "); using tf.keras, version:",
        getattr(tf.keras, "__version__", "unknown")
    )

print(
    "ECG model file path:", ARRHYTHMIA_MODEL_PATH,
    "| exists:", os.path.exists(ARRHYTHMIA_MODEL_PATH)
)


# ------------------------------------------------------------
# COMPATIBILITY SHIM FOR LOADING THE SAVED ECG MODEL
#
# Root cause of "GlorotUniform.__init__() got an unexpected keyword
# argument 'input_axes'": arrhythmia_cnn_model.keras was saved by a
# newer TensorFlow/Keras release, whose Variance-Scaling-family
# weight initializers (GlorotUniform, GlorotNormal, HeUniform,
# HeNormal, LecunUniform, LecunNormal -- the standard default kernel
# initializers for Dense/Conv layers) serialize extra
# "input_axes"/"output_axes" config keys. The TensorFlow/Keras
# version installed here predates that change, so those
# initializers' __init__() methods don't accept the extra keyword
# arguments and load_model() fails while reconstructing the layers
# -- before any weights are touched.
#
# Fix: register drop-in subclasses (via custom_objects) for every
# Variance-Scaling-family initializer that reconstructs the SAME
# initializer, just ignoring the newer-only "input_axes"/
# "output_axes" keys. This only changes how the initializer OBJECT
# is constructed during deserialization; it does not alter, retrain,
# or replace the model's learned weights, so the original trained
# ECG model (and its real ~83.63% CNN accuracy) is preserved
# unchanged.
# ------------------------------------------------------------

def _strip_incompatible_kwargs(kwargs):
    kwargs.pop("input_axes", None)
    kwargs.pop("output_axes", None)
    return kwargs


class _CompatGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


class _CompatGlorotNormal(tf.keras.initializers.GlorotNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


class _CompatHeUniform(tf.keras.initializers.HeUniform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


class _CompatHeNormal(tf.keras.initializers.HeNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


class _CompatLecunUniform(tf.keras.initializers.LecunUniform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


class _CompatLecunNormal(tf.keras.initializers.LecunNormal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **_strip_incompatible_kwargs(kwargs))


ARRHYTHMIA_COMPAT_CUSTOM_OBJECTS = {
    "GlorotUniform": _CompatGlorotUniform,
    "GlorotNormal": _CompatGlorotNormal,
    "HeUniform": _CompatHeUniform,
    "HeNormal": _CompatHeNormal,
    "LecunUniform": _CompatLecunUniform,
    "LecunNormal": _CompatLecunNormal,
}


def _load_arrhythmia_model():

    if not os.path.exists(ARRHYTHMIA_MODEL_PATH):
        print(
            "ECG model loading failed: no model file found at '"
            + ARRHYTHMIA_MODEL_PATH + "'"
        )
        return None

    try:

        model = tf.keras.models.load_model(
            ARRHYTHMIA_MODEL_PATH,
            custom_objects=ARRHYTHMIA_COMPAT_CUSTOM_OBJECTS
        )

        print("ECG model loaded successfully")

        try:
            print("ECG model input shape:", model.input_shape)
            print("ECG model output shape:", model.output_shape)
        except Exception as shape_err:
            print(
                "ECG model loaded, but input/output shape could not "
                "be read:", str(shape_err)
            )

        return model

    except Exception as e:

        # Full traceback (not just str(e)) so the real failure point
        # -- if it's something other than the initializer issue this
        # shim targets -- is visible in the terminal.
        print("ECG model loading failed:", str(e))
        print(traceback.format_exc())
        return None


arrhythmia_model = _load_arrhythmia_model()


# ============================================================
# DATABASE INIT
# ============================================================

def init_db():

    conn = sqlite3.connect(
        "cardiac_database.db"
    )

    cursor = conn.cursor()


    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)


    # --------------------------------------------------------
    # CARDIAC RECORDS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cardiac_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bpm INTEGER,
            status TEXT,
            risk REAL,
            timestamp TEXT
        )
    """)


    # --------------------------------------------------------
    # DEFAULT ADMIN ACCOUNT
    # --------------------------------------------------------

    cursor.execute(

        "SELECT * FROM users WHERE username = ?",

        ("admin",)

    )


    if cursor.fetchone() is None:

        cursor.execute(

            """
            INSERT INTO users
            (username, password)
            VALUES (?, ?)
            """,

            (
                "admin",
                "admin"
            )

        )


    conn.commit()

    conn.close()


# Initialize database

init_db()


# ============================================================
# USER CLASS
# ============================================================

class User(UserMixin):

    def __init__(
        self,
        id,
        username
    ):

        self.id = str(id)

        self.username = username


@login_manager.user_loader
def load_user(user_id):

    conn = sqlite3.connect(
        "cardiac_database.db"
    )

    cursor = conn.cursor()


    cursor.execute(

        """
        SELECT id, username
        FROM users
        WHERE id=?
        """,

        (user_id,)

    )


    user = cursor.fetchone()

    conn.close()


    if user:

        return User(
            user[0],
            user[1]
        )


    return None


# ============================================================
# ORIGINAL HEART RATE AI LOGIC
# ============================================================

def predict_heart_state(bpm):

    if bpm > threshold:

        return "Abnormal Heart Rate"

    return "Normal"


def calculate_risk(bpm):

    if bpm > threshold:

        return min(

            100,

            round(
                (bpm - threshold) * 3,
                2
            )

        )

    return 0


def get_bpm():

    return random.randint(
        65,
        100
    )


def log_data(
    bpm,
    status,
    risk
):

    conn = sqlite3.connect(
        "cardiac_database.db"
    )

    cursor = conn.cursor()


    cursor.execute(

        """
        INSERT INTO cardiac_records
        (bpm, status, risk, timestamp)
        VALUES (?, ?, ?, ?)
        """,

        (

            bpm,

            status,

            risk,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        )

    )


    conn.commit()

    conn.close()


# ============================================================
# CARDIAC RISK MODEL FUNCTION
# ============================================================

def predict_cardiac_risk(data):

    if cardiac_risk_model is None:

        return None


    try:

        input_data = pd.DataFrame(
            [data]
        )


        if cardiac_risk_features is not None:

            input_data = input_data[
                cardiac_risk_features
            ]


        prediction = (
            cardiac_risk_model.predict(
                input_data
            )[0]
        )


        if hasattr(
            cardiac_risk_model,
            "predict_proba"
        ):

            probability = (
                cardiac_risk_model
                .predict_proba(
                    input_data
                )[0][1]
                * 100
            )

        else:

            probability = 0


        return {

            "prediction":
                int(prediction),

            "probability":
                round(
                    float(probability),
                    2
                ),

            "status":
                (
                    "High Cardiac Risk"
                    if prediction == 1
                    else "Low Cardiac Risk"
                )

        }


    except Exception as e:

        print(
            "Cardiac Risk Prediction Error:",
            e
        )

        return None


# ============================================================
# CARDIAC ABNORMALITY ASSESSMENT FUNCTION
#
# Runs the loaded RandomForestClassifier (see load-time comment above
# for exactly what it was trained on) on one patient's clinical inputs
# and returns an abnormality-risk indicator -- not a diagnosis.
# ============================================================

def predict_heart_disease(data):

    if heart_disease_model is None:

        return None


    try:

        input_data = pd.DataFrame(
            [data]
        )


        if heart_disease_features is not None:

            input_data = input_data[
                heart_disease_features
            ]


        prediction = (
            heart_disease_model.predict(
                input_data
            )[0]
        )


        if hasattr(
            heart_disease_model,
            "predict_proba"
        ):

            probability = (
                heart_disease_model
                .predict_proba(
                    input_data
                )[0][1]
                * 100
            )

        else:

            probability = 0


        return {

            "prediction":
                int(prediction),

            "abnormality_probability":
                round(
                    float(probability),
                    2
                ),

            "status":
                (
                    "Cardiac Abnormality Detected"
                    if prediction == 1
                    else "No Significant Cardiac Abnormality Detected"
                )

        }


    except Exception as e:

        print(
            "Cardiac Abnormality Assessment Error:",
            e
        )

        return None


# ============================================================
# ECG ARRHYTHMIA SETTINGS
# ============================================================

ECG_BEFORE_R_PEAK = 90

ECG_AFTER_R_PEAK = 90

ECG_BEAT_LENGTH = 180


# ============================================================
# NORMALIZE ECG BEAT
# ============================================================

def normalize_ecg_beat(beat):

    beat = np.asarray(
        beat,
        dtype=np.float32
    )


    mean = np.mean(
        beat
    )

    std = np.std(
        beat
    )


    if std < 1e-8:

        return beat - mean


    return (
        beat - mean
    ) / std


# ============================================================
# ARRHYTHMIA CNN PREDICTION
# ============================================================

def predict_arrhythmia_from_beat(
    beat
):

    if arrhythmia_model is None:

        raise ValueError(
            "Arrhythmia CNN model is not loaded"
        )


    beat = np.asarray(
        beat,
        dtype=np.float32
    )


    if len(beat) != ECG_BEAT_LENGTH:

        raise ValueError(

            f"ECG beat must contain exactly "
            f"{ECG_BEAT_LENGTH} samples"

        )


    if np.isnan(beat).any():

        raise ValueError(
            "ECG beat contains NaN values"
        )


    if np.isinf(beat).any():

        raise ValueError(
            "ECG beat contains infinite values"
        )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    beat = normalize_ecg_beat(
        beat
    )


    # CNN expects:
    #
    # (batch, 180, 1)

    model_input = beat.reshape(
        1,
        ECG_BEAT_LENGTH,
        1
    )


    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    probabilities = (
        arrhythmia_model.predict(
            model_input,
            verbose=0
        )[0]
    )


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_class = (
        ARRHYTHMIA_CLASSES[
            predicted_index
        ]
    )


    confidence = round(

        float(
            probabilities[
                predicted_index
            ]
        ) * 100,

        2

    )


    rhythm_name = (
        ARRHYTHMIA_CLASS_NAMES[
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if predicted_class == "N":

        status = "Normal ECG Rhythm"

    else:

        status = "Arrhythmia Detected"


    # --------------------------------------------------------
    # PROBABILITY OF EACH CLASS
    # --------------------------------------------------------

    class_probabilities = {}


    for index, class_code in enumerate(
        ARRHYTHMIA_CLASSES
    ):

        class_probabilities[
            class_code
        ] = round(

            float(
                probabilities[index]
            ) * 100,

            2

        )


    return {

        "class_code":
            predicted_class,

        "rhythm":
            rhythm_name,

        "status":
            status,

        "confidence":
            confidence,

        "probabilities":
            class_probabilities

    }


# ============================================================
# GET REAL ECG BEAT FROM MIT-BIH
# ============================================================

def get_mitbih_ecg_beat(
    record_number="100"
):

    record_path = (

        "mit-bih-arrhythmia-database-1.0.0/"
        + str(record_number)

    )


    # --------------------------------------------------------
    # VERIFY THE REAL MIT-BIH DATASET IS ACTUALLY PRESENT
    #
    # This uses real PhysioNet MIT-BIH Arrhythmia Database records
    # only -- it never generates or fabricates ECG data. If the
    # required .hea/.dat/.atr files aren't on disk, that is reported
    # explicitly here rather than letting a low-level I/O error
    # surface, or worse, silently substituting synthetic data.
    # --------------------------------------------------------

    required_extensions = [".hea", ".dat", ".atr"]

    missing_files = [
        record_path + ext
        for ext in required_extensions
        if not os.path.exists(record_path + ext)
    ]

    if missing_files:

        raise FileNotFoundError(
            "MIT-BIH Arrhythmia Database record '"
            + str(record_number)
            + "' is not available on this server -- missing file(s): "
            + ", ".join(missing_files)
            + ". This is real PhysioNet ECG data that must be "
            + "downloaded separately (mit-bih-arrhythmia-database-1.0.0) "
            + "and placed in the project root; no synthetic ECG data "
            + "is generated as a substitute."
        )


    # --------------------------------------------------------
    # LOAD ECG SIGNAL
    # --------------------------------------------------------

    record = wfdb.rdrecord(
        record_path
    )


    # --------------------------------------------------------
    # LOAD EXPERT ANNOTATIONS
    # --------------------------------------------------------

    annotation = wfdb.rdann(
        record_path,
        "atr"
    )


    # Use first ECG channel

    signal = record.p_signal[:, 0]


    valid_symbols = {

        # Normal
        "N",
        "L",
        "R",
        "e",
        "j",

        # Supraventricular
        "A",
        "a",
        "J",
        "S",

        # Ventricular
        "V",
        "E",

        # Fusion
        "F",

        # Other
        "/",
        "f",
        "Q"

    }


    # --------------------------------------------------------
    # FIND VALID ECG BEAT
    # --------------------------------------------------------

    for sample, symbol in zip(

        annotation.sample,

        annotation.symbol

    ):


        if symbol not in valid_symbols:

            continue


        start = (
            sample -
            ECG_BEFORE_R_PEAK
        )


        end = (
            sample +
            ECG_AFTER_R_PEAK
        )


        if start < 0:

            continue


        if end > len(signal):

            continue


        beat = signal[
            start:end
        ]


        if len(beat) == ECG_BEAT_LENGTH:

            return (
                beat.tolist(),
                symbol
            )


    raise ValueError(
        "No valid ECG heartbeat found"
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
@login_required
def home():

    bpm = get_bpm()


    status = predict_heart_state(
        bpm
    )


    risk = calculate_risk(
        bpm
    )


    log_data(
        bpm,
        status,
        risk
    )


    return render_template(

        "index.html",

        bpm=bpm,

        status=status,

        risk=risk,

        accuracy=accuracy

    )


# ============================================================
# ADMIN ROUTE
# ============================================================

@app.route("/admin")
@login_required
def admin():

    conn = sqlite3.connect(
        "cardiac_database.db"
    )

    cursor = conn.cursor()


    cursor.execute(

        """
        SELECT *
        FROM cardiac_records
        ORDER BY id DESC
        LIMIT 20
        """

    )


    records = cursor.fetchall()

    conn.close()


    return render_template(

        "admin.html",

        records=records

    )


# ============================================================
# METAVERSE ROUTE
# ============================================================

@app.route("/metaverse")
@login_required
def metaverse():

    return render_template(
        "metaverse.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if request.method == "POST":

        username = request.form[
            "username"
        ]

        password = request.form[
            "password"
        ]


        conn = sqlite3.connect(
            "cardiac_database.db"
        )

        cursor = conn.cursor()


        cursor.execute(

            """
            SELECT id, username, password
            FROM users
            WHERE username=?
            """,

            (username,)

        )


        user = cursor.fetchone()

        conn.close()


        if user and user[2] == password:

            login_user(

                User(
                    user[0],
                    user[1]
                )

            )

            return redirect(
                url_for(
                    "home"
                )
            )


    return render_template(
        "login.html"
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=[
        "GET",
        "POST"
    ]
)
def signup():

    if request.method == "POST":

        username = request.form[
            "username"
        ]

        password = request.form[
            "password"
        ]


        conn = sqlite3.connect(
            "cardiac_database.db"
        )

        cursor = conn.cursor()


        cursor.execute(

            """
            SELECT *
            FROM users
            WHERE username=?
            """,

            (username,)

        )


        if cursor.fetchone():

            conn.close()

            return (
                "Username already exists"
            )


        cursor.execute(

            """
            INSERT INTO users
            (username,password)
            VALUES (?,?)
            """,

            (
                username,
                password
            )

        )


        conn.commit()

        conn.close()


        return redirect(
            url_for(
                "login"
            )
        )


    return render_template(
        "signup.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for(
            "login"
        )
    )


# ============================================================
# EXISTING HEART RATE API
# ============================================================

@app.route("/api/health")
@login_required
def api_health():

    bpm = get_bpm()


    status = predict_heart_state(
        bpm
    )


    risk = calculate_risk(
        bpm
    )


    log_data(
        bpm,
        status,
        risk
    )


    return jsonify({

        "bpm":
            bpm,

        "status":
            status,

        "risk":
            risk

    })


# ============================================================
# CARDIAC RISK API
# ============================================================

@app.route(
    "/api/cardiac-risk",
    methods=["POST"]
)
@login_required
def api_cardiac_risk():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No patient data received"

            }), 400


        result = predict_cardiac_risk(
            data
        )


        if result is None:

            return jsonify({

                "success": False,

                "error":
                    "Cardiac risk prediction failed"

            }), 500


        return jsonify({

            "success": True,

            **result

        })


    except Exception as e:

        print(
            "Cardiac Risk API Error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# CARDIAC ABNORMALITY ASSESSMENT API
# ============================================================

@app.route(
    "/api/heart-disease",
    methods=["POST"]
)
@login_required
def api_heart_disease():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No patient data received"

            }), 400


        result = predict_heart_disease(
            data
        )


        if result is None:

            return jsonify({

                "success": False,

                "error":
                    "Cardiac abnormality assessment failed"

            }), 500


        return jsonify({

            "success": True,

            **result

        })


    except Exception as e:

        print(
            "Cardiac Abnormality API Error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# CARDIAC MODEL EVALUATION API (backend/research use only)
#
# Returns the cached CARDIAC_MODEL_PERFORMANCE computed once at
# startup by evaluate_cardiac_model(). This is NOT called from
# index.html or any dashboard JavaScript -- it exists purely for
# backend testing / research validation, e.g. via curl or Postman:
#
#   curl http://127.0.0.1:5000/api/model-evaluation
#
# (requires being logged in first, since it's behind @login_required
# like the rest of the API surface)
# ============================================================

@app.route(
    "/api/model-evaluation",
    methods=["GET"]
)
@login_required
def api_model_evaluation():

    try:

        if not CARDIAC_MODEL_PERFORMANCE.get("available"):

            return jsonify({

                "success": False,

                "error":
                    CARDIAC_MODEL_PERFORMANCE.get(
                        "reason",
                        "Model evaluation is not available."
                    )

            }), 503


        return jsonify({

            "success": True,

            "accuracy":
                CARDIAC_MODEL_PERFORMANCE["accuracy"],

            "precision":
                CARDIAC_MODEL_PERFORMANCE["precision"],

            "sensitivity":
                CARDIAC_MODEL_PERFORMANCE["sensitivity"],

            "specificity":
                CARDIAC_MODEL_PERFORMANCE["specificity"],

            "f1_score":
                CARDIAC_MODEL_PERFORMANCE["f1_score"],

            "roc_auc":
                CARDIAC_MODEL_PERFORMANCE["roc_auc"],

            "confusion_matrix":
                CARDIAC_MODEL_PERFORMANCE["confusion_matrix"],

            "test_set_size":
                CARDIAC_MODEL_PERFORMANCE["test_set_size"],

            "target_column":
                CARDIAC_MODEL_PERFORMANCE.get("target_column")

        })


    except Exception as e:

        print(
            "Model Evaluation API Error:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# ECG ARRHYTHMIA PREDICTION API
# ============================================================

@app.route(
    "/api/arrhythmia",
    methods=["POST"]
)
@login_required
def api_arrhythmia():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No ECG data received"

            }), 400


        if "ecg" not in data:

            return jsonify({

                "success": False,

                "error":
                    "Missing ECG samples"

            }), 400


        beat = data[
            "ecg"
        ]


        result = (
            predict_arrhythmia_from_beat(
                beat
            )
        )


        return jsonify({

            "success": True,

            "class_code":
                result[
                    "class_code"
                ],

            "rhythm":
                result[
                    "rhythm"
                ],

            "status":
                result[
                    "status"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "model_accuracy":
                ARRHYTHMIA_MODEL_ACCURACY,

            "probabilities":
                result[
                    "probabilities"
                ]

        })


    except ValueError as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 400


    except Exception as e:

        print(
            "Arrhythmia Prediction Error:",
            str(e)
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# TEST ARRHYTHMIA CNN USING REAL MIT-BIH ECG
# ============================================================

@app.route(
    "/api/arrhythmia/test"
)
@login_required
def api_arrhythmia_test():

    try:

        record_number = request.args.get(
            "record",
            "100"
        )


        beat, actual_symbol = (
            get_mitbih_ecg_beat(
                record_number
            )
        )


        result = (
            predict_arrhythmia_from_beat(
                beat
            )
        )


        return jsonify({

            "success": True,

            "record":
                record_number,

            "actual_annotation":
                actual_symbol,

            "class_code":
                result[
                    "class_code"
                ],

            "rhythm":
                result[
                    "rhythm"
                ],

            "status":
                result[
                    "status"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "model_accuracy":
                ARRHYTHMIA_MODEL_ACCURACY,

            "probabilities":
                result[
                    "probabilities"
                ],

            "ecg":
                beat

        })


    except Exception as e:

        print(
            "MIT-BIH ECG Test Error:",
            str(e)
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
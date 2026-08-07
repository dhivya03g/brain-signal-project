import json
import random
import sqlite3
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import wfdb
from datetime import datetime

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
# ============================================================

try:

    heart_disease_model = joblib.load(
        "models/heart_disease_model.pkl"
    )

    heart_disease_features = joblib.load(
        "models/heart_disease_features.pkl"
    )

    print(
        "Heart Disease Model loaded successfully"
    )

except Exception as e:

    heart_disease_model = None

    heart_disease_features = None

    print(
        "Heart Disease Model load error:",
        e
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


try:

    arrhythmia_model = (
        tf.keras.models.load_model(
            ARRHYTHMIA_MODEL_PATH
        )
    )

    print(
        "ECG Arrhythmia CNN loaded successfully"
    )

except Exception as e:

    arrhythmia_model = None

    print(
        "ECG Arrhythmia CNN load error:",
        e
    )


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
# HEART DISEASE MODEL FUNCTION
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

            "probability":
                round(
                    float(probability),
                    2
                ),

            "status":
                (
                    "Heart Disease Detected"
                    if prediction == 1
                    else "No Heart Disease Detected"
                )

        }


    except Exception as e:

        print(
            "Heart Disease Prediction Error:",
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
# HEART DISEASE API
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
                    "Heart disease prediction failed"

            }), 500


        return jsonify({

            "success": True,

            **result

        })


    except Exception as e:

        print(
            "Heart Disease API Error:",
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
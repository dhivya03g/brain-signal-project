import os
import sys
import pickle
import numpy as np
import wfdb

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "mit-bih-arrhythmia-database-1.0.0"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# ECG SETTINGS
# ============================================================

# MIT-BIH sampling frequency = 360 Hz

BEFORE_R_PEAK = 90
AFTER_R_PEAK = 90

# Total samples per heartbeat

BEAT_LENGTH = BEFORE_R_PEAK + AFTER_R_PEAK


# ============================================================
# AAMI-STYLE BEAT GROUPS
# ============================================================

AAMI_GROUPS = {

    # --------------------------------------------------------
    # N = Normal / bundle branch beats
    # --------------------------------------------------------

    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",


    # --------------------------------------------------------
    # S = Supraventricular ectopic beats
    # --------------------------------------------------------

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",


    # --------------------------------------------------------
    # V = Ventricular ectopic beats
    # --------------------------------------------------------

    "V": "V",
    "E": "V",


    # --------------------------------------------------------
    # F = Fusion beats
    # --------------------------------------------------------

    "F": "F",


    # --------------------------------------------------------
    # Q = Unknown / paced / unclassifiable
    # --------------------------------------------------------

    "/": "Q",
    "f": "Q",
    "Q": "Q"

}


CLASS_NAMES = {

    "N": "Normal",

    "S": "Supraventricular",

    "V": "Ventricular",

    "F": "Fusion",

    "Q": "Other"

}


# ============================================================
# RECORD SPLIT
# ============================================================

# We split by RECORD rather than randomly splitting individual
# beats. This reduces leakage between train and test data.

TRAIN_RECORDS = [

    "100", "101", "103", "105", "106",
    "108", "109", "111", "112", "113",
    "115", "116", "118", "119", "121",
    "122", "124", "200", "201", "202",
    "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222",
    "223", "228", "230"

]


TEST_RECORDS = [

    "102",
    "104",
    "107",
    "117",
    "123",
    "232",
    "233",
    "234"

]


# ============================================================
# NORMALIZE ECG BEAT
# ============================================================

def normalize_beat(beat):

    beat = np.asarray(
        beat,
        dtype=np.float32
    )

    mean = np.mean(beat)

    std = np.std(beat)

    if std < 1e-8:

        return beat - mean

    return (
        beat - mean
    ) / std


# ============================================================
# LOAD ONE RECORD
# ============================================================

def load_record(record_number):

    record_path = os.path.join(
        DATASET_DIR,
        record_number
    )

    try:

        # ----------------------------------------------------
        # READ ECG SIGNAL
        # ----------------------------------------------------

        record = wfdb.rdrecord(
            record_path
        )


        # ----------------------------------------------------
        # READ EXPERT ANNOTATIONS
        # ----------------------------------------------------

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )


        # ----------------------------------------------------
        # USE FIRST ECG CHANNEL
        #
        # For many MIT-BIH records this is MLII.
        # ----------------------------------------------------

        signal = record.p_signal[:, 0]


        X = []

        y = []


        # ----------------------------------------------------
        # LOOP THROUGH ANNOTATED BEATS
        # ----------------------------------------------------

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):


            # Ignore annotations that are not part
            # of our selected beat groups.

            if symbol not in AAMI_GROUPS:

                continue


            start = (
                sample -
                BEFORE_R_PEAK
            )

            end = (
                sample +
                AFTER_R_PEAK
            )


            # Avoid going outside signal boundaries.

            if start < 0:

                continue


            if end > len(signal):

                continue


            beat = signal[
                start:end
            ]


            if len(beat) != BEAT_LENGTH:

                continue


            # Skip invalid values.

            if np.isnan(beat).any():

                continue


            if np.isinf(beat).any():

                continue


            # ------------------------------------------------
            # NORMALIZE ECG
            # ------------------------------------------------

            beat = normalize_beat(
                beat
            )


            # ------------------------------------------------
            # STORE ECG BEAT
            # ------------------------------------------------

            X.append(
                beat
            )


            # ------------------------------------------------
            # STORE AAMI CLASS
            # ------------------------------------------------

            y.append(
                AAMI_GROUPS[symbol]
            )


        return (
            np.array(
                X,
                dtype=np.float32
            ),
            np.array(y)
        )


    except Exception as e:

        print(
            f"Error reading record "
            f"{record_number}: {e}"
        )

        return (
            np.empty(
                (0, BEAT_LENGTH),
                dtype=np.float32
            ),
            np.array([])
        )


# ============================================================
# LOAD MULTIPLE RECORDS
# ============================================================

def load_dataset(records):

    all_X = []

    all_y = []


    for record_number in records:

        print(
            f"Reading record "
            f"{record_number}..."
        )


        X_record, y_record = load_record(
            record_number
        )


        if len(X_record) == 0:

            print(
                f"  No usable beats found "
                f"in {record_number}"
            )

            continue


        all_X.append(
            X_record
        )

        all_y.append(
            y_record
        )


        print(
            f"  Extracted "
            f"{len(X_record)} beats"
        )


    if not all_X:

        return (
            np.empty(
                (0, BEAT_LENGTH),
                dtype=np.float32
            ),
            np.array([])
        )


    X = np.vstack(
        all_X
    )


    y = np.concatenate(
        all_y
    )


    return X, y


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    title,
    labels
):

    print(
        "\n" +
        title
    )

    print(
        "-" * 45
    )


    counts = Counter(
        labels
    )


    for label in [
        "N",
        "S",
        "V",
        "F",
        "Q"
    ]:

        count = counts.get(
            label,
            0
        )

        print(
            f"{label} "
            f"({CLASS_NAMES[label]}): "
            f"{count}"
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        " ECG ARRHYTHMIA CLASSIFICATION MODEL"
    )

    print(
        "========================================\n"
    )


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not os.path.exists(
        DATASET_DIR
    ):

        print(
            "ERROR: MIT-BIH dataset folder "
            "was not found."
        )

        print(
            DATASET_DIR
        )

        sys.exit(1)


    print(
        "Dataset folder:"
    )

    print(
        DATASET_DIR
    )


    print(
        "\nECG beat length:",
        BEAT_LENGTH,
        "samples"
    )


    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " LOADING TRAINING RECORDS"
    )

    print(
        "========================================"
    )


    X_train, y_train = load_dataset(
        TRAIN_RECORDS
    )


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " LOADING TEST RECORDS"
    )

    print(
        "========================================"
    )


    X_test, y_test = load_dataset(
        TEST_RECORDS
    )


    # ========================================================
    # VALIDATE DATA
    # ========================================================

    if len(X_train) == 0:

        print(
            "ERROR: No training ECG beats "
            "were extracted."
        )

        sys.exit(1)


    if len(X_test) == 0:

        print(
            "ERROR: No testing ECG beats "
            "were extracted."
        )

        sys.exit(1)


    print(
        "\n========================================"
    )

    print(
        " DATASET SUMMARY"
    )

    print(
        "========================================"
    )


    print(
        "\nTraining shape:",
        X_train.shape
    )


    print(
        "Testing shape:",
        X_test.shape
    )


    print_distribution(
        "Training class distribution:",
        y_train
    )


    print_distribution(
        "Testing class distribution:",
        y_test
    )


    # ========================================================
    # TRAIN RANDOM FOREST
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " TRAINING RANDOM FOREST"
    )

    print(
        "========================================"
    )


    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1

    )


    print(
        "\nTraining model..."
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "Training completed."
    )


    # ========================================================
    # PREDICT
    # ========================================================

    print(
        "\nTesting model..."
    )


    predictions = model.predict(
        X_test
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        "\n========================================"
    )

    print(
        " MODEL RESULTS"
    )

    print(
        "========================================"
    )


    print(
        "\nModel Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    labels_present = [
        label
        for label in [
            "N",
            "S",
            "V",
            "F",
            "Q"
        ]
        if label in y_test
    ]


    target_names = [

        CLASS_NAMES[label]

        for label
        in labels_present

    ]


    print(
        "\nClassification Report:\n"
    )


    print(
        classification_report(

            y_test,

            predictions,

            labels=labels_present,

            target_names=target_names,

            zero_division=0

        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nConfusion Matrix:"
    )


    print(
        confusion_matrix(

            y_test,

            predictions,

            labels=labels_present

        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        "arrhythmia_model.pkl"
    )


    with open(
        model_path,
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )


    # ========================================================
    # SAVE MODEL INFORMATION
    # ========================================================

    info = {

        "before_r_peak":
            BEFORE_R_PEAK,

        "after_r_peak":
            AFTER_R_PEAK,

        "beat_length":
            BEAT_LENGTH,

        "classes":
            CLASS_NAMES,

        "aami_groups":
            AAMI_GROUPS,

        "accuracy":
            float(accuracy),

        "training_records":
            TRAIN_RECORDS,

        "testing_records":
            TEST_RECORDS

    }


    info_path = os.path.join(
        MODEL_DIR,
        "arrhythmia_model_info.pkl"
    )


    with open(
        info_path,
        "wb"
    ) as f:

        pickle.dump(
            info,
            f
        )


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " MODEL SAVED SUCCESSFULLY"
    )

    print(
        "========================================"
    )


    print(
        "\nModel:"
    )

    print(
        model_path
    )


    print(
        "\nModel information:"
    )

    print(
        info_path
    )


    print(
        "\nClasses:"
    )


    for code, name in CLASS_NAMES.items():

        print(
            f"{code} = {name}"
        )


    print(
        "\nArrhythmia training completed!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
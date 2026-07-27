import os
import json
import numpy as np
import wfdb
import tensorflow as tf

from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    GlobalAveragePooling1D,
    Dense
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.utils import to_categorical


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


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

# MIT-BIH = 360 Hz
#
# We take 90 samples before R peak
# and 90 samples after R peak.

BEFORE_R_PEAK = 90
AFTER_R_PEAK = 90

BEAT_LENGTH = (
    BEFORE_R_PEAK +
    AFTER_R_PEAK
)


# ============================================================
# AAMI-STYLE CLASSES
# ============================================================

AAMI_GROUPS = {

    # NORMAL
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # SUPRAVENTRICULAR
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # VENTRICULAR
    "V": "V",
    "E": "V",

    # FUSION
    "F": "F",

    # OTHER / UNKNOWN
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
# TRAIN / TEST RECORDS
# ============================================================

# IMPORTANT:
# Records are separated between train and test.
#
# This avoids beats from the same ECG record
# appearing in both sets.

TRAIN_RECORDS = [

    "100",
    "101",
    "103",
    "105",
    "106",
    "108",
    "109",
    "111",
    "112",
    "113",
    "115",
    "116",
    "118",
    "119",
    "121",
    "122",
    "124",

    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230"
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
# NORMALIZE ECG
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
# LOAD ONE ECG RECORD
# ============================================================

def load_record(record_number):

    record_path = os.path.join(
        DATASET_DIR,
        record_number
    )

    try:

        # ----------------------------------------------------
        # ECG SIGNAL
        # ----------------------------------------------------

        record = wfdb.rdrecord(
            record_path
        )


        # ----------------------------------------------------
        # EXPERT BEAT ANNOTATIONS
        # ----------------------------------------------------

        annotation = wfdb.rdann(
            record_path,
            "atr"
        )


        # ----------------------------------------------------
        # FIRST ECG CHANNEL
        # ----------------------------------------------------

        signal = record.p_signal[:, 0]


        X = []
        y = []


        # ----------------------------------------------------
        # EXTRACT EACH HEARTBEAT
        # ----------------------------------------------------

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):

            # Ignore annotation types
            # outside our selected groups.

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


            # Boundary protection

            if start < 0:

                continue


            if end > len(signal):

                continue


            beat = signal[
                start:end
            ]


            if len(beat) != BEAT_LENGTH:

                continue


            if np.isnan(beat).any():

                continue


            if np.isinf(beat).any():

                continue


            # Normalize beat

            beat = normalize_beat(
                beat
            )


            X.append(
                beat
            )


            y.append(
                AAMI_GROUPS[symbol]
            )


        return (

            np.asarray(
                X,
                dtype=np.float32
            ),

            np.asarray(y)

        )


    except Exception as error:

        print(
            f"ERROR reading "
            f"{record_number}: "
            f"{error}"
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

    X_all = []
    y_all = []


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
                "  No usable beats."
            )

            continue


        X_all.append(
            X_record
        )

        y_all.append(
            y_record
        )


        print(
            f"  Extracted "
            f"{len(X_record)} beats"
        )


    if len(X_all) == 0:

        return (

            np.empty(
                (0, BEAT_LENGTH),
                dtype=np.float32
            ),

            np.array([])

        )


    return (

        np.vstack(
            X_all
        ),

        np.concatenate(
            y_all
        )

    )


# ============================================================
# DISPLAY DISTRIBUTION
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
        "-" * 50
    )


    counter = Counter(
        labels
    )


    for label in [
        "N",
        "S",
        "V",
        "F",
        "Q"
    ]:

        print(

            f"{label} "
            f"({CLASS_NAMES[label]}): "
            f"{counter.get(label, 0)}"

        )


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def balance_training_data(
    X,
    y,
    max_per_class=12000
):

    """
    Reduces extreme Normal-class dominance while
    retaining minority classes.

    We DO NOT alter the test dataset.
    """

    X_balanced = []
    y_balanced = []


    rng = np.random.default_rng(
        42
    )


    for label in [
        "N",
        "S",
        "V",
        "F",
        "Q"
    ]:

        indexes = np.where(
            y == label
        )[0]


        if len(indexes) == 0:

            continue


        # Limit huge classes such as Normal.

        if len(indexes) > max_per_class:

            indexes = rng.choice(

                indexes,

                size=max_per_class,

                replace=False

            )


        X_balanced.append(
            X[indexes]
        )

        y_balanced.append(
            y[indexes]
        )


    X_balanced = np.vstack(
        X_balanced
    )


    y_balanced = np.concatenate(
        y_balanced
    )


    # Shuffle

    permutation = rng.permutation(
        len(X_balanced)
    )


    return (

        X_balanced[
            permutation
        ],

        y_balanced[
            permutation
        ]

    )


# ============================================================
# CREATE CNN
# ============================================================

def build_model(
    number_of_classes
):

    model = Sequential([

        Input(
            shape=(
                BEAT_LENGTH,
                1
            )
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 1
        # ----------------------------------------------------

        Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(
            0.20
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 2
        # ----------------------------------------------------

        Conv1D(
            filters=64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(
            0.25
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 3
        # ----------------------------------------------------

        Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(
            0.30
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 4
        # ----------------------------------------------------

        Conv1D(
            filters=256,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        BatchNormalization(),


        # ----------------------------------------------------
        # GLOBAL FEATURE EXTRACTION
        # ----------------------------------------------------

        GlobalAveragePooling1D(),


        # ----------------------------------------------------
        # DENSE LAYER
        # ----------------------------------------------------

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.40
        ),


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        Dense(
            number_of_classes,
            activation="softmax"
        )

    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n============================================"
    )

    print(
        " ECG ARRHYTHMIA 1D CNN MODEL"
    )

    print(
        "============================================\n"
    )


    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not os.path.exists(
        DATASET_DIR
    ):

        print(
            "ERROR: MIT-BIH dataset "
            "folder not found:"
        )

        print(
            DATASET_DIR
        )

        return


    # ========================================================
    # LOAD TRAINING RECORDS
    # ========================================================

    print(
        "============================================"
    )

    print(
        " LOADING TRAINING ECG RECORDS"
    )

    print(
        "============================================"
    )


    X_train, y_train = load_dataset(
        TRAIN_RECORDS
    )


    # ========================================================
    # LOAD TEST RECORDS
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " LOADING TEST ECG RECORDS"
    )

    print(
        "============================================"
    )


    X_test, y_test = load_dataset(
        TEST_RECORDS
    )


    if len(X_train) == 0:

        print(
            "No training beats found."
        )

        return


    if len(X_test) == 0:

        print(
            "No testing beats found."
        )

        return


    # ========================================================
    # ORIGINAL DISTRIBUTION
    # ========================================================

    print(
        "\nOriginal training shape:",
        X_train.shape
    )


    print(
        "Testing shape:",
        X_test.shape
    )


    print_distribution(

        "Original Training Distribution",

        y_train

    )


    print_distribution(

        "Testing Distribution",

        y_test

    )


    # ========================================================
    # BALANCE TRAINING DATA
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " BALANCING TRAINING DATA"
    )

    print(
        "============================================"
    )


    X_train, y_train = balance_training_data(

        X_train,

        y_train,

        max_per_class=12000

    )


    print(
        "\nBalanced training shape:",
        X_train.shape
    )


    print_distribution(

        "Balanced Training Distribution",

        y_train

    )


    # ========================================================
    # LABEL ENCODING
    # ========================================================

    label_encoder = LabelEncoder()


    # Explicitly establish all five labels.

    label_encoder.fit(
        [
            "N",
            "S",
            "V",
            "F",
            "Q"
        ]
    )


    y_train_encoded = label_encoder.transform(
        y_train
    )


    y_test_encoded = label_encoder.transform(
        y_test
    )


    number_of_classes = len(
        label_encoder.classes_
    )


    print(
        "\nLabel mapping:"
    )


    for index, label in enumerate(
        label_encoder.classes_
    ):

        print(
            index,
            "->",
            label,
            "->",
            CLASS_NAMES[label]
        )


    # ========================================================
    # ONE-HOT LABELS
    # ========================================================

    y_train_categorical = to_categorical(

        y_train_encoded,

        num_classes=number_of_classes

    )


    # ========================================================
    # CNN INPUT SHAPE
    # ========================================================

    X_train = np.expand_dims(
        X_train,
        axis=-1
    )


    X_test = np.expand_dims(
        X_test,
        axis=-1
    )


    print(
        "\nCNN Training Shape:",
        X_train.shape
    )


    print(
        "CNN Testing Shape:",
        X_test.shape
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " BUILDING CNN"
    )

    print(
        "============================================"
    )


    model = build_model(
        number_of_classes
    )


    model.summary()


    # ========================================================
    # CALLBACKS
    # ========================================================

    best_model_path = os.path.join(

        MODEL_DIR,

        "arrhythmia_cnn_model.keras"

    )


    callbacks = [


        EarlyStopping(

            monitor="val_loss",

            patience=6,

            restore_best_weights=True,

            verbose=1

        ),


        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=0.00001,

            verbose=1

        ),


        ModelCheckpoint(

            best_model_path,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        )

    ]


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " TRAINING CNN"
    )

    print(
        "============================================"
    )


    history = model.fit(

        X_train,

        y_train_categorical,

        validation_split=0.15,

        epochs=30,

        batch_size=128,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # TEST PREDICTION
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " TESTING CNN"
    )

    print(
        "============================================"
    )


    probabilities = model.predict(

        X_test,

        batch_size=256,

        verbose=1

    )


    predicted_encoded = np.argmax(

        probabilities,

        axis=1

    )


    predictions = label_encoder.inverse_transform(

        predicted_encoded

    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(

        y_test,

        predictions

    )


    print(
        "\n============================================"
    )

    print(
        " CNN RESULTS"
    )

    print(
        "============================================"
    )


    print(

        "\nCNN Accuracy:",

        round(
            accuracy * 100,
            2
        ),

        "%"

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    evaluation_labels = [

        "N",
        "S",
        "V",
        "F",
        "Q"

    ]


    print(
        "\nClassification Report:\n"
    )


    print(

        classification_report(

            y_test,

            predictions,

            labels=evaluation_labels,

            target_names=[

                CLASS_NAMES[label]

                for label
                in evaluation_labels

            ],

            zero_division=0

        )

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nConfusion Matrix:\n"
    )


    print(

        confusion_matrix(

            y_test,

            predictions,

            labels=evaluation_labels

        )

    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = os.path.join(

        MODEL_DIR,

        "arrhythmia_cnn_final.keras"

    )


    model.save(
        final_model_path
    )


    # ========================================================
    # SAVE MODEL INFORMATION
    # ========================================================

    info = {

        "model_type":
            "1D CNN",

        "dataset":
            "MIT-BIH Arrhythmia Database",

        "sampling_frequency":
            360,

        "before_r_peak":
            BEFORE_R_PEAK,

        "after_r_peak":
            AFTER_R_PEAK,

        "beat_length":
            BEAT_LENGTH,

        "accuracy":
            float(accuracy),

        "classes":
            CLASS_NAMES,

        "label_order":
            label_encoder.classes_.tolist(),

        "train_records":
            TRAIN_RECORDS,

        "test_records":
            TEST_RECORDS

    }


    info_path = os.path.join(

        MODEL_DIR,

        "arrhythmia_cnn_info.json"

    )


    with open(
        info_path,
        "w"
    ) as file:

        json.dump(

            info,

            file,

            indent=4

        )


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " CNN TRAINING COMPLETED"
    )

    print(
        "============================================"
    )


    print(
        "\nBest model:"
    )

    print(
        best_model_path
    )


    print(
        "\nFinal model:"
    )

    print(
        final_model_path
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


    for label in evaluation_labels:

        print(

            label,
            "=",
            CLASS_NAMES[label]

        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
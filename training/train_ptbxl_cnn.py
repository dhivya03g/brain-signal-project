from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "datasets" / "ptb-xl"

DATABASE_CSV = DATASET_DIR / "ptbxl_database.csv"
SCP_CSV = DATASET_DIR / "scp_statements.csv"

RECORDS100_DIR = DATASET_DIR / "records100"

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODELS_DIR / "ptbxl_arrhythmia_cnn.keras"
)

INFO_PATH = (
    MODELS_DIR / "ptbxl_arrhythmia_cnn_info.json"
)

EVALUATION_PATH = (
    MODELS_DIR / "ptbxl_evaluation.json"
)

CONFUSION_MATRIX_PATH = (
    MODELS_DIR / "ptbxl_confusion_matrix.png"
)

CHECKPOINT_PATH = (
    MODELS_DIR / "ptbxl_training_best.keras"
)


# =====================================================================
# PTB-XL CONFIGURATION
# =====================================================================

CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

INPUT_SAMPLES = 1000
INPUT_LEADS = 12
SAMPLING_RATE = 100

INPUT_SHAPE = (
    INPUT_SAMPLES,
    INPUT_LEADS,
)

BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 0.001

DROPOUT = 0.4
THRESHOLD = 0.50


# =====================================================================
# CONFIRMED PHASE-1 COUNTS
# =====================================================================

EXPECTED_TOTAL_RECORDS = 21799
EXPECTED_USABLE_RECORDS = 21388
EXPECTED_EXCLUDED_RECORDS = 411

EXPECTED_UNIQUE_PATIENTS = 18869
EXPECTED_USABLE_PATIENTS = 18617

EXPECTED_TRAIN_RECORDS = 14920
EXPECTED_VALIDATION_RECORDS = 3220
EXPECTED_TEST_RECORDS = 3248

EXPECTED_TRAIN_PATIENTS = 13031
EXPECTED_VALIDATION_PATIENTS = 2793
EXPECTED_TEST_PATIENTS = 2793


# =====================================================================
# UTILITY
# =====================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =====================================================================
# REQUIRED FILE CHECK
# =====================================================================

def check_required_files() -> None:
    required = [
        DATABASE_CSV,
        SCP_CSV,
        RECORDS100_DIR,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required PTB-XL files/directories are missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )

    print("PTB-XL dataset location:")
    print(f"  {DATASET_DIR}")
    print(f"  Database: {DATABASE_CSV}")
    print(f"  SCP statements: {SCP_CSV}")
    print(f"  records100: {RECORDS100_DIR}")


# =====================================================================
# METADATA
# =====================================================================

def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:

    print_header("Loading PTB-XL metadata")

    db = pd.read_csv(
        DATABASE_CSV,
        index_col=0,
    )

    scp = pd.read_csv(
        SCP_CSV,
        index_col=0,
    )

    print(
        f"Database records loaded: "
        f"{len(db):,}"
    )

    print(
        f"SCP statement rows loaded: "
        f"{len(scp):,}"
    )

    required_db_columns = [
        "scp_codes",
        "patient_id",
        "filename_lr",
    ]

    for column in required_db_columns:

        if column not in db.columns:
            raise ValueError(
                f"Database is missing required column: "
                f"{column}"
            )

    required_scp_columns = [
        "diagnostic",
        "diagnostic_class",
    ]

    for column in required_scp_columns:

        if column not in scp.columns:
            raise ValueError(
                f"SCP statements are missing required "
                f"column: {column}"
            )

    return db, scp


# =====================================================================
# SCP → SUPERCLASS
# =====================================================================

def build_scp_superclass_mapping(
    scp: pd.DataFrame,
) -> Dict[str, str]:

    mapping = {}

    for code, row in scp.iterrows():

        diagnostic_flag = row.get(
            "diagnostic"
        )

        if pd.isna(diagnostic_flag):
            continue

        try:
            is_diagnostic = (
                float(diagnostic_flag) == 1.0
            )

        except (
            TypeError,
            ValueError,
        ):
            is_diagnostic = (
                str(diagnostic_flag)
                .strip()
                .lower()
                in {
                    "1",
                    "true",
                    "yes",
                }
            )

        if not is_diagnostic:
            continue

        superclass = row.get(
            "diagnostic_class"
        )

        if pd.isna(superclass):
            continue

        superclass = (
            str(superclass)
            .strip()
            .upper()
        )

        if superclass in CLASS_NAMES:

            mapping[
                str(code).strip()
            ] = superclass

    if not mapping:

        raise ValueError(
            "No diagnostic SCP mappings "
            "were found."
        )

    return mapping


# =====================================================================
# SCP PARSING
# =====================================================================

def parse_scp_codes(
    value,
) -> Dict[str, float]:

    if pd.isna(value):
        return {}

    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if not text:
        return {}

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(parsed, dict):
            return parsed

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    raise ValueError(
        f"Unable to parse scp_codes: {text}"
    )


# =====================================================================
# MULTI-LABEL TARGET CREATION
# =====================================================================

# =====================================================================
# MULTI-LABEL TARGET CREATION
#
# IMPORTANT:
# This must preserve the original PTB-XL patient_id type.
#
# Phase 1 used:
#
#     sorted(usable_db["patient_id"].unique())
#
# without converting patient_id to string.
#
# Therefore we MUST NOT convert patient_id to str here.
# =====================================================================

def create_multilabel_targets(
    db: pd.DataFrame,
    scp_mapping: Dict[str, str],
) -> Tuple[
    pd.DataFrame,
    Dict[str, int],
    int,
]:

    rows = []

    class_counts = {
        name: 0
        for name in CLASS_NAMES
    }

    multi_label_count = 0

    invalid_scp_count = 0
    no_target_count = 0

    for record_id, row in db.iterrows():

        # -------------------------------------------------------------
        # Parse SCP codes exactly like Phase 1
        # -------------------------------------------------------------

        try:

            scp_codes = parse_scp_codes(
                row["scp_codes"]
            )

        except Exception:

            invalid_scp_count += 1
            continue

        # -------------------------------------------------------------
        # Determine diagnostic superclasses
        # -------------------------------------------------------------

        classes = sorted({
            scp_mapping[
                str(code).strip()
            ]
            for code in scp_codes.keys()
            if str(code).strip()
            in scp_mapping
        })

        # -------------------------------------------------------------
        # No target diagnostic superclass
        # -------------------------------------------------------------

        if len(classes) == 0:

            no_target_count += 1
            continue

        # -------------------------------------------------------------
        # Multi-label vector
        #
        # CLASS_NAMES:
        #     NORM, MI, STTC, CD, HYP
        # -------------------------------------------------------------

        label_vector = [
            int(
                class_name in classes
            )
            for class_name in CLASS_NAMES
        ]

        # -------------------------------------------------------------
        # Class counts
        # -------------------------------------------------------------

        for class_name in classes:

            class_counts[
                class_name
            ] += 1

        # -------------------------------------------------------------
        # Multi-label count
        # -------------------------------------------------------------

        if len(classes) > 1:

            multi_label_count += 1

        # -------------------------------------------------------------
        # filename_lr
        # -------------------------------------------------------------

        filename = row["filename_lr"]

        if pd.isna(filename):

            raise ValueError(
                f"Usable record {record_id} "
                "has no filename_lr."
            )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT convert patient_id to str.
        #
        # Phase 1 used the original numeric patient IDs.
        # Keeping the original dtype guarantees that the exact
        # patient sorting and train/validation/test split are reproduced.
        # -------------------------------------------------------------

        rows.append(
            {
                "record_id": str(
                    record_id
                ),

                "patient_id":
                    row["patient_id"],

                "filename_lr": str(
                    filename
                ),

                "classes":
                    classes,

                "label_vector":
                    label_vector,
            }
        )

    labels = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------

    expected_usable = (
        EXPECTED_USABLE_RECORDS
    )

    if len(labels) != expected_usable:

        raise ValueError(
            "PTB-XL usable-record construction "
            "does not match Phase 1.\n"
            f"Expected usable records: "
            f"{expected_usable:,}\n"
            f"Actual usable records:   "
            f"{len(labels):,}\n"
            f"Invalid SCP records:     "
            f"{invalid_scp_count:,}\n"
            f"No target records:       "
            f"{no_target_count:,}"
        )

    expected_excluded = (
        EXPECTED_EXCLUDED_RECORDS
    )

    actual_excluded = (
        invalid_scp_count
        + no_target_count
    )

    if actual_excluded != expected_excluded:

        raise ValueError(
            "PTB-XL excluded-record count "
            "does not match Phase 1.\n"
            f"Expected excluded: "
            f"{expected_excluded:,}\n"
            f"Actual excluded:   "
            f"{actual_excluded:,}"
        )

    # -------------------------------------------------------------
    # Patient count check
    # -------------------------------------------------------------

    usable_patients = (
        labels[
            "patient_id"
        ].nunique()
    )

    if (
        usable_patients
        != EXPECTED_USABLE_PATIENTS
    ):

        raise ValueError(
            "PTB-XL usable-patient count "
            "does not match Phase 1.\n"
            f"Expected: "
            f"{EXPECTED_USABLE_PATIENTS:,}\n"
            f"Actual:   "
            f"{usable_patients:,}"
        )

    print()
    print(
        "Usable-record construction matches "
        "Phase 1 exactly."
    )

    print(
        f"  Usable records: "
        f"{len(labels):,}"
    )

    print(
        f"  Excluded records: "
        f"{actual_excluded:,}"
    )

    print(
        f"  Usable patients: "
        f"{usable_patients:,}"
    )

    return (
        labels,
        class_counts,
        multi_label_count,
    )


# =====================================================================
# PATIENT-LEVEL SPLIT
#
# IMPORTANT:
# This MUST reproduce phase1_ptbxl_inspection.py exactly.
#
# Phase 1 does:
#
#     patients = np.array(
#         sorted(
#             usable_db["patient_id"].unique()
#         )
#     )
#
#     train_patients, temp_patients = train_test_split(
#         patients,
#         test_size=0.30,
#         random_state=42,
#     )
#
#     validation_patients, test_patients = train_test_split(
#         temp_patients,
#         test_size=0.50,
#         random_state=42,
#     )
#
# No string conversion is performed before splitting.
# No new split is generated using another method.
# =====================================================================

def create_phase1_patient_split(
    usable: pd.DataFrame,
) -> pd.DataFrame:

    # ---------------------------------------------------------------
    # EXACT PHASE-1 PATIENT ARRAY
    # ---------------------------------------------------------------

    patients = np.array(
        sorted(
            usable["patient_id"].unique()
        )
    )

    # ---------------------------------------------------------------
    # EXACT PHASE-1 SPLIT
    # ---------------------------------------------------------------

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=42,
    )

    validation_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=42,
    )

    # Convert to sets AFTER the split.
    train_patients = set(train_patients)
    validation_patients = set(validation_patients)
    test_patients = set(test_patients)

    # ---------------------------------------------------------------
    # PATIENT LEAKAGE CHECK
    # ---------------------------------------------------------------

    train_val_overlap = (
        train_patients
        & validation_patients
    )

    train_test_overlap = (
        train_patients
        & test_patients
    )

    validation_test_overlap = (
        validation_patients
        & test_patients
    )

    if train_val_overlap:
        raise RuntimeError(
            "Patient leakage detected between "
            "train and validation."
        )

    if train_test_overlap:
        raise RuntimeError(
            "Patient leakage detected between "
            "train and test."
        )

    if validation_test_overlap:
        raise RuntimeError(
            "Patient leakage detected between "
            "validation and test."
        )

    # ---------------------------------------------------------------
    # ASSIGN RECORDS
    #
    # IMPORTANT:
    # Use the original patient_id values.
    # Do NOT use astype(str) here.
    # ---------------------------------------------------------------

    result = usable.copy()

    train_mask = result["patient_id"].isin(
        train_patients
    )

    validation_mask = result["patient_id"].isin(
        validation_patients
    )

    test_mask = result["patient_id"].isin(
        test_patients
    )

    # Every record must belong to exactly one split.
    assignment_count = (
        train_mask.astype(int)
        + validation_mask.astype(int)
        + test_mask.astype(int)
    )

    if not np.all(
        assignment_count.to_numpy() == 1
    ):
        raise RuntimeError(
            "Every usable record must belong to "
            "exactly one split."
        )

    result["split"] = np.select(
        [
            train_mask,
            validation_mask,
            test_mask,
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default=None,
    )

    # ---------------------------------------------------------------
    # PRINT THE SPLIT BEFORE VERIFICATION
    # ---------------------------------------------------------------

    print()
    print("Phase-1 reproduction:")
    print(
        f"Train patients: "
        f"{len(train_patients):,}"
    )
    print(
        f"Validation patients: "
        f"{len(validation_patients):,}"
    )
    print(
        f"Test patients: "
        f"{len(test_patients):,}"
    )
    print(
        f"Train records: "
        f"{int(train_mask.sum()):,}"
    )
    print(
        f"Validation records: "
        f"{int(validation_mask.sum()):,}"
    )
    print(
        f"Test records: "
        f"{int(test_mask.sum()):,}"
    )

    return result

# =====================================================================
# SPLIT VERIFICATION
# =====================================================================

def verify_phase1_split(
    df: pd.DataFrame,
) -> None:

    print_header(
        "Verifying Phase-1 patient-level split"
    )

    train_df = df[
        df["split"] == "train"
    ]

    val_df = df[
        df["split"] == "validation"
    ]

    test_df = df[
        df["split"] == "test"
    ]

    train_records = len(train_df)
    val_records = len(val_df)
    test_records = len(test_df)

    print(
        f"Train records:      {train_records:,}"
    )

    print(
        f"Validation records: {val_records:,}"
    )

    print(
        f"Test records:       {test_records:,}"
    )

    # -------------------------------------------------------------
    # EXACT PHASE-1 RECORD COUNTS
    # -------------------------------------------------------------

    expected_records = {
        "train": EXPECTED_TRAIN_RECORDS,
        "validation": EXPECTED_VALIDATION_RECORDS,
        "test": EXPECTED_TEST_RECORDS,
    }

    actual_records = {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }

    if actual_records != expected_records:

        raise ValueError(
            "Phase-1 record counts do not match.\n"
            f"Expected: {expected_records}\n"
            f"Actual:   {actual_records}"
        )

    # -------------------------------------------------------------
    # PATIENT COUNTS
    #
    # DO NOT CONVERT PATIENT IDs TO STRING.
    # -------------------------------------------------------------

    train_patients = set(
        train_df["patient_id"]
    )

    val_patients = set(
        val_df["patient_id"]
    )

    test_patients = set(
        test_df["patient_id"]
    )

    print(
        f"Train patients:      "
        f"{len(train_patients):,}"
    )

    print(
        f"Validation patients: "
        f"{len(val_patients):,}"
    )

    print(
        f"Test patients:       "
        f"{len(test_patients):,}"
    )

    # -------------------------------------------------------------
    # LEAKAGE CHECK
    # -------------------------------------------------------------

    train_val = (
        train_patients
        & val_patients
    )

    train_test = (
        train_patients
        & test_patients
    )

    val_test = (
        val_patients
        & test_patients
    )

    print(
        f"Train ∩ Validation: "
        f"{len(train_val)}"
    )

    print(
        f"Train ∩ Test:       "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test:  "
        f"{len(val_test)}"
    )

    if train_val:
        raise ValueError(
            "Patient leakage detected "
            "between train and validation."
        )

    if train_test:
        raise ValueError(
            "Patient leakage detected "
            "between train and test."
        )

    if val_test:
        raise ValueError(
            "Patient leakage detected "
            "between validation and test."
        )

    # -------------------------------------------------------------
    # EXACT PHASE-1 PATIENT COUNTS
    # -------------------------------------------------------------

    expected_patients = {
        "train": EXPECTED_TRAIN_PATIENTS,
        "validation": EXPECTED_VALIDATION_PATIENTS,
        "test": EXPECTED_TEST_PATIENTS,
    }

    actual_patients = {
        "train": len(train_patients),
        "validation": len(val_patients),
        "test": len(test_patients),
    }

    if actual_patients != expected_patients:

        raise ValueError(
            "Patient counts do not match Phase 1.\n"
            f"Expected: {expected_patients}\n"
            f"Actual:   {actual_patients}"
        )

    print()
    print(
        "PASS: exact Phase-1 "
        "record counts verified."
    )

    print(
        "PASS: exact Phase-1 "
        "patient counts verified."
    )

    print(
        "PASS: patient leakage = 0."
    )

# =====================================================================
# WAVEFORM PATH
# =====================================================================

def resolve_record_path(
    filename_lr: str,
) -> Path:

    relative = Path(
        str(filename_lr)
    )

    if (
        relative.parts
        and
        relative.parts[0].lower()
        == "records100"
    ):

        return DATASET_DIR / relative

    return DATASET_DIR / relative


# =====================================================================
# LOAD ONE WAVEFORM
# =====================================================================

def load_waveform(
    record_row: pd.Series,
) -> Tuple[
    np.ndarray,
    dict,
]:

    record_path = (
        resolve_record_path(
            record_row[
                "filename_lr"
            ]
        )
    )

    header_path = (
        record_path.with_suffix(
            ".hea"
        )
    )

    if not header_path.exists():

        raise FileNotFoundError(
            f"Header file not found: "
            f"{header_path}"
        )

    try:

        record = wfdb.rdrecord(
            str(record_path)
        )

    except Exception as exc:

        raise RuntimeError(
            f"WFDB failed to read "
            f"{record_path}: {exc}"
        ) from exc

    signal = np.asarray(
        record.p_signal,
        dtype=np.float32,
    )

    if signal.ndim != 2:

        raise ValueError(
            f"Expected 2D waveform, "
            f"got {signal.shape}"
        )

    if signal.shape != INPUT_SHAPE:

        raise ValueError(
            f"Expected waveform shape "
            f"{INPUT_SHAPE}, "
            f"got {signal.shape}"
        )

    fs = float(
        record.fs
    )

    if not np.isclose(
        fs,
        SAMPLING_RATE,
    ):

        raise ValueError(
            f"Expected {SAMPLING_RATE} Hz, "
            f"got {fs}"
        )

    if not np.isfinite(
        signal
    ).all():

        raise ValueError(
            "Waveform contains NaN/Inf."
        )

    return signal, {
        "record_path":
            str(record_path),

        "sampling_rate":
            fs,

        "shape":
            list(signal.shape),

        "leads":
            int(signal.shape[1]),

        "samples":
            int(signal.shape[0]),
    }


# =====================================================================
# TRAINING NORMALIZATION
# =====================================================================

def calculate_training_normalization(
    train_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
]:

    sums = np.zeros(
        INPUT_LEADS,
        dtype=np.float64,
    )

    squared_sums = np.zeros(
        INPUT_LEADS,
        dtype=np.float64,
    )

    total_samples = 0
    successful = 0

    for _, row in train_df.iterrows():

        signal, _ = load_waveform(
            row
        )

        sums += np.sum(
            signal,
            axis=0,
            dtype=np.float64,
        )

        squared_sums += np.sum(
            signal.astype(
                np.float64
            ) ** 2,
            axis=0,
        )

        total_samples += (
            signal.shape[0]
        )

        successful += 1

    if successful == 0:

        raise RuntimeError(
            "No training waveforms "
            "could be loaded."
        )

    means = (
        sums / total_samples
    )

    variances = (
        squared_sums
        / total_samples
    ) - (
        means ** 2
    )

    variances = np.maximum(
        variances,
        1e-12,
    )

    stds = np.sqrt(
        variances
    )

    return (
        means.astype(
            np.float32
        ),
        stds.astype(
            np.float32
        ),
        successful,
    )


# =====================================================================
# NORMALIZE
# =====================================================================

def normalize_signal(
    signal: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:

    normalized = (
        signal.astype(
            np.float32
        )
        - means.reshape(
            1,
            -1,
        )
    ) / stds.reshape(
        1,
        -1,
    )

    return normalized.astype(
        np.float32
    )


# =====================================================================
# VALIDATION ONLY
# =====================================================================

def run_validation_only(
    limit: int,
) -> None:

    print_header(
        "PTB-XL PHASE 2 VALIDATION-ONLY"
    )

    check_required_files()

    db, scp = load_metadata()

    if len(db) != EXPECTED_TOTAL_RECORDS:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RECORDS:,} "
            f"records, found {len(db):,}."
        )

    mapping = (
        build_scp_superclass_mapping(
            scp
        )
    )

    print()
    print(
        f"Diagnostic SCP codes mapped: "
        f"{len(mapping):,}"
    )

    labels, class_counts, multi = (
        create_multilabel_targets(
            db,
            mapping,
        )
    )

    print()
    print(
        "TARGET CLASS COUNTS"
    )
    print("-" * 50)

    for name in CLASS_NAMES:

        print(
            f"{name:<6}"
            f"{CLASS_DESCRIPTIONS[name]:<28}"
            f"{class_counts[name]:,}"
        )

    print()
    print(
        f"Multi-label records: "
        f"{multi:,}"
    )

    usable = labels[
        labels["classes"].apply(
            lambda x: len(x) > 0
        )
    ].copy()

    excluded = labels[
        labels["classes"].apply(
            lambda x: len(x) == 0
        )
    ].copy()

    # The target creation only puts records with a target
    # into labels, so verify against the confirmed count.
    if len(usable) != EXPECTED_USABLE_RECORDS:

        raise ValueError(
            f"Expected {EXPECTED_USABLE_RECORDS:,} "
            f"usable records, got {len(usable):,}."
        )

    expected_excluded = (
        EXPECTED_TOTAL_RECORDS
        - EXPECTED_USABLE_RECORDS
    )

    print(
        f"Usable labelled records: "
        f"{len(usable):,}"
    )

    print(
        f"Excluded records: "
        f"{expected_excluded:,}"
    )

    usable_patients = (
        usable["patient_id"]
        .astype(str)
        .nunique()
    )

    if (
        usable_patients
        != EXPECTED_USABLE_PATIENTS
    ):

        raise ValueError(
            f"Expected "
            f"{EXPECTED_USABLE_PATIENTS:,} "
            f"usable patients, got "
            f"{usable_patients:,}."
        )

    # Reproduce the approved Phase-1 split.
    usable = (
        create_phase1_patient_split(
            usable
        )
    )

    verify_phase1_split(
        usable
    )

    # -------------------------------------------------------------
    # Validate only a small real waveform sample.
    # -------------------------------------------------------------

    selected_parts = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = usable[
            usable["split"]
            == split_name
        ]

        if not subset.empty:

            selected_parts.append(
                subset.head(1)
            )

    if selected_parts:

        selected = pd.concat(
            selected_parts
        ).drop_duplicates(
            subset=["record_id"]
        )

    else:

        selected = usable.head(0)

    remaining = usable[
        ~usable["record_id"].isin(
            selected["record_id"]
        )
    ].head(
        max(
            0,
            limit - len(selected),
        )
    )

    selected = pd.concat(
        [
            selected,
            remaining,
        ]
    ).head(limit)

    # Training-only provisional normalization.
    training_sample = usable[
        usable["split"] == "train"
    ].head(
        min(
            12,
            len(
                usable[
                    usable["split"]
                    == "train"
                ]
            ),
        )
    )

    means, stds, norm_count = (
        calculate_training_normalization(
            training_sample
        )
    )

    successful = 0
    failed = 0

    print()
    print(
        f"Limited waveform attempts: "
        f"{len(selected)}"
    )

    for _, row in selected.iterrows():

        try:

            signal, details = (
                load_waveform(row)
            )

            normalized = (
                normalize_signal(
                    signal,
                    means,
                    stds,
                )
            )

            labels_vector = np.asarray(
                row["label_vector"],
                dtype=np.float32,
            )

            if labels_vector.shape != (
                NUM_CLASSES,
            ):

                raise ValueError(
                    "Invalid label vector shape."
                )

            if not np.isfinite(
                normalized
            ).all():

                raise ValueError(
                    "Normalized waveform "
                    "contains non-finite values."
                )

            successful += 1

            if successful == 1:

                print(
                    f"Waveform shape: "
                    f"{signal.shape}"
                )

                print(
                    f"Finite: "
                    f"{np.isfinite(signal).all()}"
                )

                print(
                    "Normalized lead means:",
                    np.round(
                        normalized.mean(
                            axis=0
                        ),
                        4,
                    ).tolist(),
                )

                print(
                    "Normalized lead stds:",
                    np.round(
                        normalized.std(
                            axis=0
                        ),
                        4,
                    ).tolist(),
                )

                print(
                    "Label vector:",
                    labels_vector.astype(
                        int
                    ).tolist(),
                )

                print(
                    "Assignment:",
                    row["split"],
                )

        except Exception as exc:

            failed += 1

            print(
                f"Validation failure for "
                f"{row['record_id']}: "
                f"{exc}"
            )

    print()
    print(
        f"Successful waveform loads: "
        f"{successful}"
    )

    print(
        f"Failed waveform loads: "
        f"{failed}"
    )

    print(
        f"Normalization statistics from "
        f"{norm_count} training records."
    )

    if successful != len(selected):

        raise RuntimeError(
            "Validation failed."
        )

    print()
    print(
        "PASS: validation-only "
        "preprocessing succeeded."
    )

    print(
        "PASS: no CNN training "
        "was performed."
    )


# =====================================================================
# CNN MODEL
# =====================================================================

def build_model() -> tf.keras.Model:

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="ecg_input",
    )

    x = tf.keras.layers.Conv1D(
        32,
        7,
        padding="same",
        name="conv1",
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name="bn1"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu1"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        2,
        name="pool1",
    )(x)

    x = tf.keras.layers.Conv1D(
        64,
        7,
        padding="same",
        name="conv2",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn2"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu2"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        2,
        name="pool2",
    )(x)

    x = tf.keras.layers.Conv1D(
        128,
        5,
        padding="same",
        name="conv3",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn3"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu3"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        2,
        name="pool3",
    )(x)

    x = tf.keras.layers.Conv1D(
        256,
        5,
        padding="same",
        name="conv4",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="bn4"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu4"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        2,
        name="pool4",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="dense128",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",
        name="output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PTBXL_5Superclass_MultiLabel_CNN",
    )


# =====================================================================
# WEIGHTED BCE
# =====================================================================

class WeightedBinaryCrossEntropy(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        class_weights: Dict[int, float],
        name: str = (
            "weighted_binary_crossentropy"
        ),
    ):

        super().__init__(
            name=name
        )

        vector = [
            class_weights[i]
            for i in range(
                NUM_CLASSES
            )
        ]

        self.class_weights = (
            tf.constant(
                vector,
                dtype=tf.float32,
            )
        )

    def call(
        self,
        y_true,
        y_pred,
    ):

        y_true = tf.cast(
            y_true,
            tf.float32,
        )

        epsilon = (
            tf.keras.backend.epsilon()
        )

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon,
        )

        bce = -(
            y_true
            * tf.math.log(y_pred)
            +
            (1.0 - y_true)
            * tf.math.log(
                1.0 - y_pred
            )
        )

        sample_weights = (
            1.0
            +
            y_true
            * (
                self.class_weights
                - 1.0
            )
        )

        return tf.reduce_mean(
            bce * sample_weights
        )


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

def calculate_class_weights(
    y_train: np.ndarray,
) -> Dict[int, float]:

    weights = {}

    for index, name in enumerate(
        CLASS_NAMES
    ):

        positives = int(
            np.sum(
                y_train[:, index]
                == 1
            )
        )

        negatives = int(
            np.sum(
                y_train[:, index]
                == 0
            )
        )

        if positives == 0:

            raise RuntimeError(
                f"No positive examples "
                f"for {name}."
            )

        weights[index] = (
            negatives
            / positives
        )

    return weights


# =====================================================================
# LOAD A COMPLETE SPLIT
# =====================================================================

def load_dataset_arrays(
    df: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
]:

    X = []
    y = []
    record_ids = []

    total = len(df)

    print(
        f"Loading {total:,} waveforms..."
    )

    for counter, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        signal, _ = load_waveform(
            row
        )

        normalized = (
            normalize_signal(
                signal,
                means,
                stds,
            )
        )

        X.append(
            normalized
        )

        y.append(
            np.asarray(
                row["label_vector"],
                dtype=np.float32,
            )
        )

        record_ids.append(
            str(row["record_id"])
        )

        if (
            counter == 1
            or counter % 500 == 0
            or counter == total
        ):

            print(
                f"  Loaded "
                f"{counter:,}/{total:,}"
            )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if X.shape != (
        len(df),
        INPUT_SAMPLES,
        INPUT_LEADS,
    ):

        raise RuntimeError(
            f"Unexpected X shape: "
            f"{X.shape}"
        )

    if y.shape != (
        len(df),
        NUM_CLASSES,
    ):

        raise RuntimeError(
            f"Unexpected y shape: "
            f"{y.shape}"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Non-finite values in dataset."
        )

    return (
        X,
        y,
        record_ids,
    )


# =====================================================================
# EVALUATION
# =====================================================================

def calculate_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Optional[float]]:

    result = []

    for index in range(
        NUM_CLASSES
    ):

        cm = confusion_matrix(
            y_true[:, index],
            y_pred[:, index],
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        denominator = (
            tn + fp
        )

        if denominator == 0:
            result.append(None)

        else:
            result.append(
                float(
                    tn / denominator
                )
            )

    return result


def calculate_evaluation(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:

    y_pred = (
        probabilities
        >= THRESHOLD
    ).astype(np.int32)

    specificity = (
        calculate_specificity(
            y_true,
            y_pred,
        )
    )

    macro_auc = None

    try:

        macro_auc = roc_auc_score(
            y_true,
            probabilities,
            average="macro",
        )

    except ValueError:
        pass

    per_class = {}

    for index, name in enumerate(
        CLASS_NAMES
    ):

        yt = y_true[:, index]
        yp = y_pred[:, index]
        prob = probabilities[:, index]

        cm = confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        try:

            auc = roc_auc_score(
                yt,
                prob,
            )

        except ValueError:

            auc = None

        per_class[name] = {
            "precision": float(
                precision_score(
                    yt,
                    yp,
                    zero_division=0,
                )
            ),

            "recall_sensitivity":
                float(
                    recall_score(
                        yt,
                        yp,
                        zero_division=0,
                    )
                ),

            "specificity":
                (
                    float(
                        tn
                        / (tn + fp)
                    )
                    if tn + fp > 0
                    else None
                ),

            "f1": float(
                f1_score(
                    yt,
                    yp,
                    zero_division=0,
                )
            ),

            "roc_auc":
                (
                    float(auc)
                    if auc is not None
                    else None
                ),

            "confusion_matrix":
                cm.tolist(),

            "support_positive":
                int(np.sum(yt == 1)),

            "support_negative":
                int(np.sum(yt == 0)),
        }

    return {
        "threshold":
            THRESHOLD,

        "accuracy_exact_match":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_recall_sensitivity":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_specificity":
            (
                float(
                    np.nanmean(
                        [
                            x
                            if x is not None
                            else np.nan
                            for x
                            in specificity
                        ]
                    )
                )
                if any(
                    x is not None
                    for x in specificity
                )
                else None
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_roc_auc":
            (
                float(macro_auc)
                if macro_auc is not None
                else None
            ),

        "per_class":
            per_class,

        "multilabel_confusion_matrix":
            [
                confusion_matrix(
                    y_true[:, i],
                    y_pred[:, i],
                    labels=[0, 1],
                ).tolist()
                for i in range(
                    NUM_CLASSES
                )
            ],
    }


# =====================================================================
# FULL TRAINING
# =====================================================================

def train_full() -> None:

    print_header(
        "FULL PTB-XL CNN TRAINING"
    )

    print(
        "Starting PTB-XL CNN training..."
    )

    check_required_files()

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    db, scp = load_metadata()

    if len(db) != (
        EXPECTED_TOTAL_RECORDS
    ):

        raise RuntimeError(
            "Unexpected PTB-XL record count."
        )

    mapping = (
        build_scp_superclass_mapping(
            scp
        )
    )

    labels, class_counts, multi = (
        create_multilabel_targets(
            db,
            mapping,
        )
    )

    if len(labels) != (
        EXPECTED_USABLE_RECORDS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_USABLE_RECORDS:,} "
            f"usable records, got "
            f"{len(labels):,}."
        )

    usable_patients = (
        labels["patient_id"]
        .astype(str)
        .nunique()
    )

    if usable_patients != (
        EXPECTED_USABLE_PATIENTS
    ):

        raise RuntimeError(
            "Unexpected usable patient count."
        )

    # -------------------------------------------------------------
    # EXACT PATIENT-LEVEL SPLIT
    # -------------------------------------------------------------

    usable = (
        create_phase1_patient_split(
            labels
        )
    )

    verify_phase1_split(
        usable
    )

    train_df = usable[
        usable["split"] == "train"
    ].copy()

    val_df = usable[
        usable["split"] == "validation"
    ].copy()

    test_df = usable[
        usable["split"] == "test"
    ].copy()

    # -------------------------------------------------------------
    # TRAINING-ONLY NORMALIZATION
    # -------------------------------------------------------------

    print_header(
        "TRAINING-ONLY NORMALIZATION"
    )

    print(
        "Calculating normalization "
        "statistics from TRAIN only."
    )

    means, stds, norm_count = (
        calculate_training_normalization(
            train_df
        )
    )

    print(
        "Normalization statistics "
        "calculated."
    )

    print(
        "Means:",
        np.round(
            means,
            6,
        ).tolist(),
    )

    print(
        "Stds:",
        np.round(
            stds,
            6,
        ).tolist(),
    )

    # -------------------------------------------------------------
    # TRAINING DATA
    # -------------------------------------------------------------

    print_header(
        "LOADING TRAINING DATA"
    )

    X_train, y_train, train_ids = (
        load_dataset_arrays(
            train_df,
            means,
            stds,
        )
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    # -------------------------------------------------------------
    # VALIDATION DATA
    # -------------------------------------------------------------

    print_header(
        "LOADING VALIDATION DATA"
    )

    X_val, y_val, val_ids = (
        load_dataset_arrays(
            val_df,
            means,
            stds,
        )
    )

    print(
        f"X_val shape: "
        f"{X_val.shape}"
    )

    print(
        f"y_val shape: "
        f"{y_val.shape}"
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # NO TEST DATA IS LOADED HERE.
    #
    # The test set remains completely untouched until training,
    # early stopping and model selection are finished.
    # -------------------------------------------------------------

    print()
    print(
        "Held-out test set remains UNUSED "
        "until final evaluation."
    )

    # -------------------------------------------------------------
    # CLASS WEIGHTS
    # -------------------------------------------------------------

    print_header(
        "TRAINING-ONLY CLASS WEIGHTS"
    )

    class_weights = (
        calculate_class_weights(
            y_train
        )
    )

    named_weights = {
        CLASS_NAMES[index]:
            float(weight)

        for index, weight
        in class_weights.items()
    }

    for name, weight in (
        named_weights.items()
    ):

        print(
            f"{name}: "
            f"{weight:.8f}"
        )

    # -------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------

    print_header(
        "BUILDING PTB-XL CNN"
    )

    model = build_model()

    loss = (
        WeightedBinaryCrossEntropy(
            class_weights
        )
    )

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy",
                threshold=THRESHOLD,
            )
        ],
    )

    model.summary()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                CHECKPOINT_PATH
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # -------------------------------------------------------------
    # ACTUAL TRAINING
    # -------------------------------------------------------------

    print_header(
        "TRAINING"
    )

    print(
        "Starting PTB-XL CNN training..."
    )

    print(
        f"Training records: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation records: "
        f"{len(X_val):,}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    start_time = (
        time.perf_counter()
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    epochs_completed = len(
        history.history["loss"]
    )

    print()
    print(
        "Training completed."
    )

    print(
        f"Epochs completed: "
        f"{epochs_completed}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    # -------------------------------------------------------------
    # SAVE FINAL MODEL
    # -------------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print(
        f"Saved model: "
        f"{MODEL_PATH}"
    )

    # -------------------------------------------------------------
    # FREE TRAIN/VALIDATION MEMORY
    #
    # The test set is loaded only NOW.
    # -------------------------------------------------------------

    del X_train
    del y_train
    del X_val
    del y_val

    import gc
    gc.collect()

    # -------------------------------------------------------------
    # FINAL HELD-OUT TEST DATA
    # -------------------------------------------------------------

    print_header(
        "LOADING HELD-OUT TEST DATA"
    )

    print(
        "The CNN has finished training."
    )

    print(
        "Now loading the held-out "
        "test set for FINAL evaluation."
    )

    X_test, y_test, test_ids = (
        load_dataset_arrays(
            test_df,
            means,
            stds,
        )
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"y_test shape: "
        f"{y_test.shape}"
    )

    # -------------------------------------------------------------
    # FINAL TEST PREDICTION
    # -------------------------------------------------------------

    print_header(
        "HELD-OUT TEST EVALUATION"
    )

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.shape != (
        len(X_test),
        NUM_CLASSES,
    ):

        raise RuntimeError(
            f"Unexpected prediction "
            f"shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Non-finite model probabilities."
        )

    evaluation = (
        calculate_evaluation(
            y_test.astype(
                np.int32
            ),
            probabilities,
        )
    )

    # -------------------------------------------------------------
    # CONFUSION MATRIX FIGURE
    # -------------------------------------------------------------

    confusion_saved = False

    try:

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1,
            NUM_CLASSES,
            figsize=(15, 3),
        )

        for index, name in enumerate(
            CLASS_NAMES
        ):

            cm = np.asarray(
                evaluation[
                    "multilabel_confusion_matrix"
                ][index]
            )

            ax = axes[index]

            ax.imshow(cm)

            ax.set_title(name)

            ax.set_xticks(
                [0, 1]
            )

            ax.set_yticks(
                [0, 1]
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            for r in range(2):

                for c in range(2):

                    ax.text(
                        c,
                        r,
                        str(
                            cm[r, c]
                        ),
                        ha="center",
                        va="center",
                    )

        fig.tight_layout()

        fig.savefig(
            CONFUSION_MATRIX_PATH,
            dpi=150,
        )

        plt.close(fig)

        confusion_saved = True

    except Exception as exc:

        print(
            "WARNING: Could not save "
            f"confusion matrix: {exc}"
        )

    # -------------------------------------------------------------
    # SAVE METADATA
    # -------------------------------------------------------------

    metadata = {

        "model_name":
            "PTB-XL 5-Superclass Multi-Label CNN",

        "dataset":
            "PTB-XL",

        "total_records":
            int(len(db)),

        "usable_records":
            int(len(usable)),

        "excluded_records":
            int(
                len(db)
                - len(usable)
            ),

        "usable_patients":
            int(
                usable[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

        "multi_label_records":
            int(multi),

        "class_names":
            CLASS_NAMES,

        "class_descriptions":
            CLASS_DESCRIPTIONS,

        "class_counts":
            {
                name:
                    int(
                        class_counts[
                            name
                        ]
                    )

                for name
                in CLASS_NAMES
            },

        "train_records":
            int(len(train_df)),

        "validation_records":
            int(len(val_df)),

        "test_records":
            int(len(test_df)),

        "train_patients":
            int(
                train_df[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

        "validation_patients":
            int(
                val_df[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

        "test_patients":
            int(
                test_df[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

        "patient_leakage_verified":
            True,

        "input_shape":
            list(INPUT_SHAPE),

        "num_leads":
            INPUT_LEADS,

        "num_samples":
            INPUT_SAMPLES,

        "sampling_rate_hz":
            SAMPLING_RATE,

        "waveform_source":
            "records100",

        "normalization":
            {
                "method":
                    "per-lead z-score",

                "statistics_source":
                    "training split only",

                "means":
                    means.tolist(),

                "stds":
                    stds.tolist(),

                "training_waveforms_used":
                    int(norm_count),
            },

        "architecture":
            [
                "Input(1000,12)",
                "Conv1D(32,7) + BatchNormalization + ReLU + MaxPooling1D",
                "Conv1D(64,7) + BatchNormalization + ReLU + MaxPooling1D",
                "Conv1D(128,5) + BatchNormalization + ReLU + MaxPooling1D",
                "Conv1D(256,5) + BatchNormalization + ReLU + MaxPooling1D",
                "GlobalAveragePooling1D",
                "Dense(128,relu)",
                "Dropout(0.4)",
                "Dense(5,sigmoid)",
            ],

        "loss":
            "Weighted Binary Cross-Entropy",

        "class_weights":
            named_weights,

        "optimizer":
            "Adam",

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "maximum_epochs":
            MAX_EPOCHS,

        "epochs_completed":
            int(epochs_completed),

        "early_stopping":
            {
                "monitor":
                    "val_loss",

                "patience":
                    8,

                "restore_best_weights":
                    True,
            },

        "reduce_lr_on_plateau":
            {
                "monitor":
                    "val_loss",

                "factor":
                    0.5,

                "patience":
                    4,

                "min_lr":
                    1e-7,
            },

        "random_seed":
            SEED,

        "threshold":
            THRESHOLD,

        "training_time_seconds":
            float(training_time),

        "model_path":
            str(MODEL_PATH),

        "evaluation_path":
            str(EVALUATION_PATH),

        "confusion_matrix_path":
            str(
                CONFUSION_MATRIX_PATH
            ),
    }

    evaluation_output = {

        "dataset":
            "PTB-XL",

        "model":
            "PTB-XL 5-Superclass Multi-Label CNN",

        "test_records":
            int(len(test_df)),

        "test_patients":
            int(
                test_df[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

        "class_names":
            CLASS_NAMES,

        "metrics":
            evaluation,

        "training_time_seconds":
            float(training_time),

        "epochs_completed":
            int(epochs_completed),

        "threshold":
            THRESHOLD,

        "confusion_matrix_figure_saved":
            confusion_saved,

        "confusion_matrix_figure":
            str(
                CONFUSION_MATRIX_PATH
            ),
    }

    with open(
        INFO_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # REAL TEST RESULTS
    # -------------------------------------------------------------

    print_header(
        "REAL HELD-OUT TEST RESULTS"
    )

    print(
        f"Exact-match Accuracy: "
        f"{evaluation['accuracy_exact_match']:.6f}"
    )

    print(
        f"Macro Precision: "
        f"{evaluation['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall/Sensitivity: "
        f"{evaluation['macro_recall_sensitivity']:.6f}"
    )

    if (
        evaluation[
            "macro_specificity"
        ]
        is not None
    ):

        print(
            f"Macro Specificity: "
            f"{evaluation['macro_specificity']:.6f}"
        )

    else:

        print(
            "Macro Specificity: N/A"
        )

    print(
        f"Macro F1: "
        f"{evaluation['macro_f1']:.6f}"
    )

    if (
        evaluation[
            "macro_roc_auc"
        ]
        is not None
    ):

        print(
            f"Macro ROC-AUC: "
            f"{evaluation['macro_roc_auc']:.6f}"
        )

    else:

        print(
            "Macro ROC-AUC: N/A"
        )

    print()
    print(
        "Per-class metrics:"
    )

    for name in CLASS_NAMES:

        metrics = (
            evaluation[
                "per_class"
            ][name]
        )

        print()
        print(name)

        print(
            f"  Precision: "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"  Recall/Sensitivity: "
            f"{metrics['recall_sensitivity']:.6f}"
        )

        if (
            metrics["specificity"]
            is not None
        ):

            print(
                f"  Specificity: "
                f"{metrics['specificity']:.6f}"
            )

        else:

            print(
                "  Specificity: N/A"
            )

        print(
            f"  F1: "
            f"{metrics['f1']:.6f}"
        )

        if (
            metrics["roc_auc"]
            is not None
        ):

            print(
                f"  ROC-AUC: "
                f"{metrics['roc_auc']:.6f}"
            )

        else:

            print(
                "  ROC-AUC: N/A"
            )

        print(
            "  Confusion matrix:"
        )

        print(
            np.asarray(
                metrics[
                    "confusion_matrix"
                ]
            )
        )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"  {INFO_PATH}"
    )

    print(
        f"  {EVALUATION_PATH}"
    )

    if confusion_saved:

        print(
            f"  {CONFUSION_MATRIX_PATH}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These metrics were calculated "
        "from the held-out test set "
        "only."
    )


# =====================================================================
# CLI
# =====================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "PTB-XL 5-superclass "
            "multi-label CNN pipeline"
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run lightweight real-data "
            "validation only. "
            "No training."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help=(
            "Number of real records "
            "to validate."
        ),
    )

    args = parser.parse_args()

    if args.limit <= 0:

        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.validate_only:

        run_validation_only(
            args.limit
        )

        return

    # -------------------------------------------------------------
    # NORMAL COMMAND
    #
    # There is intentionally NO --train argument.
    # Normal execution goes directly to training.
    # -------------------------------------------------------------

    train_full()


if __name__ == "__main__":
    main()
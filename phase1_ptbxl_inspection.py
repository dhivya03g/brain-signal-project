from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("datasets/ptb-xl")

DATABASE_FILE = DATASET_DIR / "ptbxl_database.csv"
SCP_FILE = DATASET_DIR / "scp_statements.csv"

RANDOM_SEED = 42

TARGET_CLASSES = {
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
}


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("PTB-XL PHASE 1 INSPECTION")
print("=" * 70)

if not DATABASE_FILE.exists():
    raise FileNotFoundError(f"Missing: {DATABASE_FILE}")

if not SCP_FILE.exists():
    raise FileNotFoundError(f"Missing: {SCP_FILE}")

db = pd.read_csv(DATABASE_FILE, index_col=0)
scp = pd.read_csv(SCP_FILE, index_col=0)

print(f"\nDatabase records: {len(db):,}")
print(f"Unique patients: {db['patient_id'].nunique():,}")

print("\nDatabase columns:")
print(db.columns.tolist())

print("\nSCP columns:")
print(scp.columns.tolist())


# ============================================================
# BUILD SCP → DIAGNOSTIC SUPERCLASS MAPPING
# ============================================================

required_scp_columns = {
    "diagnostic",
    "diagnostic_class",
}

missing = required_scp_columns - set(scp.columns)

if missing:
    raise ValueError(f"Missing SCP columns: {sorted(missing)}")

diagnostic_scp = scp[scp["diagnostic"] == 1]

scp_mapping = {}

for code, row in diagnostic_scp.iterrows():

    superclass = row["diagnostic_class"]

    if pd.isna(superclass):
        continue

    superclass = str(superclass).strip().upper()

    if superclass in TARGET_CLASSES:
        scp_mapping[str(code).strip()] = superclass


print("\n" + "=" * 70)
print("DIAGNOSTIC SCP → SUPERCLASS MAPPING")
print("=" * 70)

for superclass in sorted(TARGET_CLASSES):

    codes = sorted(
        code
        for code, value in scp_mapping.items()
        if value == superclass
    )

    print(f"\n{superclass}:")
    print(", ".join(codes) if codes else "(none)")


# ============================================================
# PARSE RECORD LABELS
# ============================================================

record_results = []

invalid_scp_records = 0
no_diagnostic_records = 0

for record_id, row in db.iterrows():

    raw_codes = row["scp_codes"]

    try:
        codes = ast.literal_eval(raw_codes)

        if not isinstance(codes, dict):
            raise ValueError("scp_codes is not a dictionary")

    except Exception:
        invalid_scp_records += 1

        record_results.append({
            "record_id": record_id,
            "patient_id": row["patient_id"],
            "classes": [],
            "status": "invalid_scp_codes",
        })

        continue

    classes = sorted({
        scp_mapping[str(code).strip()]
        for code in codes.keys()
        if str(code).strip() in scp_mapping
    })

    if len(classes) == 0:

        no_diagnostic_records += 1

        record_results.append({
            "record_id": record_id,
            "patient_id": row["patient_id"],
            "classes": [],
            "status": "no_target_diagnostic_class",
        })

        continue

    record_results.append({
        "record_id": record_id,
        "patient_id": row["patient_id"],
        "classes": classes,
        "status": (
            "multi_label"
            if len(classes) > 1
            else "single_label"
        ),
    })


# ============================================================
# COUNTS
# ============================================================

usable_records = [
    r
    for r in record_results
    if r["status"] in {
        "single_label",
        "multi_label",
    }
]

excluded_records = [
    r
    for r in record_results
    if r["status"] not in {
        "single_label",
        "multi_label",
    }
]

multi_label_records = [
    r
    for r in usable_records
    if len(r["classes"]) > 1
]


class_counts = {
    class_name: sum(
        class_name in r["classes"]
        for r in usable_records
    )
    for class_name in sorted(TARGET_CLASSES)
}


# ============================================================
# REPORT CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("EXACT CLASS DISTRIBUTION")
print("=" * 70)

print(f"\nTotal PTB-XL records : {len(db):,}")
print(f"Usable records      : {len(usable_records):,}")
print(f"Excluded records    : {len(excluded_records):,}")

for class_name in sorted(TARGET_CLASSES):
    print(
        f"{class_name:6s}               : "
        f"{class_counts[class_name]:,}"
    )

print(
    f"\nMulti-label records : "
    f"{len(multi_label_records):,}"
)

print(
    f"Invalid scp_codes  : "
    f"{invalid_scp_records:,}"
)

print(
    f"No target class    : "
    f"{no_diagnostic_records:,}"
)


# ============================================================
# MULTI-LABEL DISTRIBUTION
# ============================================================

multi_label_combinations = {}

for record in multi_label_records:

    combination = tuple(record["classes"])

    multi_label_combinations[combination] = (
        multi_label_combinations.get(combination, 0) + 1
    )


print("\n" + "=" * 70)
print("MULTI-LABEL COMBINATIONS")
print("=" * 70)

if multi_label_combinations:

    for combination, count in sorted(
        multi_label_combinations.items(),
        key=lambda x: (-x[1], x[0])
    ):
        print(
            f"{' + '.join(combination):30s} : "
            f"{count:,}"
        )

else:
    print("No multi-label records found.")


# ============================================================
# PATIENT INFORMATION
# ============================================================

usable_record_ids = [
    r["record_id"]
    for r in usable_records
]

usable_db = db.loc[usable_record_ids].copy()

unique_patients = usable_db["patient_id"].nunique()

print("\n" + "=" * 70)
print("PATIENT INFORMATION")
print("=" * 70)

print(
    f"Unique patients in complete dataset : "
    f"{db['patient_id'].nunique():,}"
)

print(
    f"Unique patients in usable dataset   : "
    f"{unique_patients:,}"
)


# ============================================================
# MULTI-LABEL DECISION
# ============================================================

print("\n" + "=" * 70)
print("MULTI-LABEL HANDLING")
print("=" * 70)

print(
    """
The records remain multi-label during Phase 1.

For example:

    MI + STTC

is NOT silently converted to MI or STTC.

The class counts therefore represent:

    number of records containing each superclass.

No single-label conversion is performed in this Phase 1 script.
"""
)


# ============================================================
# PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

patients = np.array(
    sorted(
        usable_db["patient_id"].unique()
    )
)

print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT")
print("=" * 70)

# 70% train
# 15% validation
# 15% test

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=RANDOM_SEED,
)

validation_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_SEED,
)


train_patients = set(train_patients)
validation_patients = set(validation_patients)
test_patients = set(test_patients)


train_mask = usable_db["patient_id"].isin(train_patients)
validation_mask = usable_db["patient_id"].isin(validation_patients)
test_mask = usable_db["patient_id"].isin(test_patients)


train_records = int(train_mask.sum())
validation_records = int(validation_mask.sum())
test_records = int(test_mask.sum())


print(
    f"\nRandom seed        : {RANDOM_SEED}"
)

print(
    f"\nTrain patients     : "
    f"{len(train_patients):,}"
)

print(
    f"Validation patients: "
    f"{len(validation_patients):,}"
)

print(
    f"Test patients      : "
    f"{len(test_patients):,}"
)

print(
    f"\nTrain records      : "
    f"{train_records:,}"
)

print(
    f"Validation records : "
    f"{validation_records:,}"
)

print(
    f"Test records       : "
    f"{test_records:,}"
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_val_overlap = train_patients & validation_patients
train_test_overlap = train_patients & test_patients
val_test_overlap = validation_patients & test_patients

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)

if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "PATIENT LEAKAGE DETECTED!"
    )

print("\nPASS: No patient appears in multiple splits.")


# ============================================================
# WAVEFORM INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("WAVEFORM INFORMATION")
print("=" * 70)

if "fs" in db.columns:

    print("\nSampling frequency distribution:")

    print(
        db.loc[usable_record_ids, "fs"]
        .value_counts(dropna=False)
        .to_string()
    )

filename_columns = [
    c
    for c in db.columns
    if "filename" in c.lower()
]

print(
    "\nWaveform filename columns:",
    filename_columns
)


# ============================================================
# SAVE PHASE 1 REPORT
# ============================================================

report = {
    "total_records": int(len(db)),
    "usable_records": int(len(usable_records)),
    "excluded_records": int(len(excluded_records)),
    "unique_patients_total": int(
        db["patient_id"].nunique()
    ),
    "unique_patients_usable": int(
        unique_patients
    ),
    "class_counts": {
        k: int(v)
        for k, v in class_counts.items()
    },
    "multi_label_records": int(
        len(multi_label_records)
    ),
    "invalid_scp_codes": int(
        invalid_scp_records
    ),
    "no_target_diagnostic_class": int(
        no_diagnostic_records
    ),
    "split": {
        "method": "patient-level",
        "random_seed": RANDOM_SEED,
        "train_patients": len(train_patients),
        "validation_patients": len(validation_patients),
        "test_patients": len(test_patients),
        "train_records": train_records,
        "validation_records": validation_records,
        "test_records": test_records,
    },
    "multi_label_combinations": {
        "+".join(k): int(v)
        for k, v in multi_label_combinations.items()
    },
}

output_file = Path(
    "ptbxl_phase1_report.json"
)

output_file.write_text(
    json.dumps(
        report,
        indent=2
    ),
    encoding="utf-8"
)

print(
    f"\nPhase 1 report saved to: "
    f"{output_file}"
)

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE — NO TRAINING PERFORMED")
print("=" * 70)
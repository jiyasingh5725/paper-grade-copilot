# ============================================================
# PAPER GRADE TRANSITION COPILOT
# Stage 2 - Feature Engineering
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("STAGE 2 - FEATURE ENGINEERING")
print("=" * 80)

print("\nLoading:")
print(INPUT_PATH)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_PATH}\n"
        "Run 01_eda.py first."
    )


df = pd.read_csv(INPUT_PATH)


print("\nDataset loaded successfully.")

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ============================================================
# 3. CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# 4. SORT DATA
# ============================================================

df = df.sort_values(
    [
        "transition_id",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 5. CREATE FUTURE DEVIATION TARGET
# ============================================================
#
# This is the target our regression model will eventually
# predict.
#
# Future Basis Weight:
#     future_basis_weight_5min
#
# Target:
#     future_deviation_pct
#
# Formula:
#
# |Future BW - Setpoint|
# ---------------------- × 100
#       Setpoint
# ============================================================

df["future_deviation_pct"] = (
    (
        df["future_basis_weight_5min"]
        -
        df["basis_weight_setpoint"]
    )
    .abs()
    /
    df["basis_weight_setpoint"]
    * 100
)


# ============================================================
# 6. CREATE FUTURE OFF-SPEC TARGET
# ============================================================
#
# Honeywell requirement:
#
# Deviation > 2.5%
#       ↓
# OFF-SPEC
#
# Otherwise:
# SAFE
# ============================================================

df["future_off_spec"] = (
    df["future_deviation_pct"] > 2.5
).astype(int)


# ============================================================
# 7. CREATE CURRENT DEVIATION FROM RAW PROCESS VALUES
# ============================================================
#
# This is an independent calculation.
#
# We don't use the existing deviation_pct as our source
# for future prediction.
# ============================================================

df["current_deviation_pct"] = (
    (
        df["basis_weight"]
        -
        df["basis_weight_setpoint"]
    )
    .abs()
    /
    df["basis_weight_setpoint"]
    * 100
)


# ============================================================
# 8. CREATE CURRENT OFF-SPEC FLAG
# ============================================================

df["current_off_spec"] = (
    df["current_deviation_pct"] > 2.5
).astype(int)


# ============================================================
# 9. PROCESS DEVIATION FROM OPERATING LIMITS
# ============================================================
#
# These features tell the recommendation engine how close
# the current process is to its allowed operating limits.
#
# Example:
#
# machine_speed = 820
# minimum = 700
# maximum = 900
#
# normalized position:
#
# (820 - 700) / (900 - 700)
# = 0.60
#
# This means the machine is 60% through its allowed range.
# ============================================================

LIMIT_FEATURES = [

    (
        "machine_speed",
        "machine_speed_min",
        "machine_speed_max",
        "machine_speed_position"
    ),

    (
        "stock_flow",
        "stock_flow_min",
        "stock_flow_max",
        "stock_flow_position"
    ),

    (
        "filler_flow",
        "filler_flow_min",
        "filler_flow_max",
        "filler_flow_position"
    ),

    (
        "steam_pressure",
        "steam_pressure_min",
        "steam_pressure_max",
        "steam_pressure_position"
    ),

    (
        "moisture",
        "moisture_min",
        "moisture_max",
        "moisture_position"
    ),

    (
        "ash",
        "ash_min",
        "ash_max",
        "ash_position"
    ),

    (
        "caliper",
        "caliper_min",
        "caliper_max",
        "caliper_position"
    ),
]


for value_col, min_col, max_col, output_col in LIMIT_FEATURES:

    denominator = (
        df[max_col]
        -
        df[min_col]
    )

    denominator = denominator.replace(
        0,
        np.nan
    )

    df[output_col] = (
        (
            df[value_col]
            -
            df[min_col]
        )
        /
        denominator
    )


# ============================================================
# 10. DISTANCE TO OPERATING LIMITS
# ============================================================

for value_col, min_col, max_col, output_col in LIMIT_FEATURES:

    min_distance_name = (
        output_col
        + "_from_min"
    )

    max_distance_name = (
        output_col
        + "_from_max"
    )

    df[min_distance_name] = (
        df[value_col]
        -
        df[min_col]
    )

    df[max_distance_name] = (
        df[max_col]
        -
        df[value_col]
    )


# ============================================================
# 11. GRADE TRANSITION FEATURE
# ============================================================

df["grade_transition"] = (
    df["grade_from"].astype(str)
    +
    "_TO_"
    +
    df["grade_to"].astype(str)
)


# ============================================================
# 12. GRADE CHANGE FLAG
# ============================================================

df["grade_changed"] = (
    df["grade_from"]
    !=
    df["grade_to"]
).astype(int)


# ============================================================
# 13. TRANSITION PHASE ENCODING
# ============================================================
#
# We use a numerical representation for the process phase.
#
# pre_transition = 0
# transition     = 1
# stabilization  = 2
# stable         = 3
# ============================================================

phase_mapping = {

    "pre_transition": 0,

    "transition": 1,

    "stabilization": 2,

    "stable": 3
}


df["transition_phase_encoded"] = (
    df["transition_phase"]
    .map(phase_mapping)
)


# ============================================================
# 14. OPERATOR ACTION FLAG
# ============================================================

df["operator_adjustment_flag"] = (
    df["operator_action"]
    !=
    "none"
).astype(int)


# ============================================================
# 15. INTERACTION FEATURES
# ============================================================
#
# These help the ML model understand relationships between
# important process variables.
# ============================================================

df["speed_stock_ratio"] = (
    df["machine_speed"]
    /
    df["stock_flow"].replace(
        0,
        np.nan
    )
)


df["steam_moisture_ratio"] = (
    df["steam_pressure"]
    /
    df["moisture"].replace(
        0,
        np.nan
    )
)


df["filler_stock_ratio"] = (
    df["filler_flow"]
    /
    df["stock_flow"].replace(
        0,
        np.nan
    )
)


# ============================================================
# 16. SETPOINT ERROR
# ============================================================

df["basis_weight_error"] = (
    df["basis_weight"]
    -
    df["basis_weight_setpoint"]
)


# ============================================================
# 17. SETPOINT ERROR DIRECTION
# ============================================================
#
# -1 = below setpoint
#  0 = exactly at setpoint
# +1 = above setpoint
# ============================================================

df["basis_weight_error_direction"] = np.select(

    [
        df["basis_weight_error"] < 0,

        df["basis_weight_error"] > 0
    ],

    [
        -1,

        1
    ],

    default=0
)


# ============================================================
# 18. DROP ROWS WITH INVALID VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 19. CHECK MISSING VALUES CREATED BY ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING MISSING VALUES")
print("=" * 80)

missing = (
    df.isnull()
    .sum()
)

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print(
        "No missing values created."
    )

else:

    print(
        missing
    )


# ============================================================
# 20. SELECT FEATURES FOR ML
# ============================================================
#
# IMPORTANT:
#
# We deliberately exclude:
#
# future_basis_weight_5min
# future_deviation_pct
# future_off_spec
#
# because these are future information.
#
# We also exclude:
#
# transition_outcome
# stabilization_time_min
#
# because these are outcome variables.
#
# These would create DATA LEAKAGE.
# ============================================================

NUMERIC_FEATURES = [

    # Current process variables

    "machine_speed",

    "stock_flow",

    "filler_flow",

    "steam_pressure",

    "moisture",

    "ash",

    "caliper",


    # Current Basis Weight information

    "basis_weight",

    "basis_weight_setpoint",

    "basis_weight_lag_1",

    "basis_weight_lag_5",

    "basis_weight_change_5min",


    # Process changes

    "speed_change_5min",

    "stock_flow_change_5min",


    # Operating limit position

    "machine_speed_position",

    "stock_flow_position",

    "filler_flow_position",

    "steam_pressure_position",

    "moisture_position",

    "ash_position",

    "caliper_position",


    # Distance from limits

    "machine_speed_position_from_min",

    "machine_speed_position_from_max",

    "stock_flow_position_from_min",

    "stock_flow_position_from_max",

    "filler_flow_position_from_min",

    "filler_flow_position_from_max",

    "steam_pressure_position_from_min",

    "steam_pressure_position_from_max",

    "moisture_position_from_min",

    "moisture_position_from_max",

    "ash_position_from_min",

    "ash_position_from_max",

    "caliper_position_from_min",

    "caliper_position_from_max",


    # Transition information

    "grade_changed",

    "transition_phase_encoded",

    "operator_adjustment_flag",


    # Interaction features

    "speed_stock_ratio",

    "steam_moisture_ratio",

    "filler_stock_ratio",


    # Current error

    "basis_weight_error",

    "basis_weight_error_direction",
]


CATEGORICAL_FEATURES = [

    "grade_from",

    "grade_to",

    "recipe_id",

    "transition_phase",

    "operator_action",
]


TARGET_COLUMNS = [

    "future_basis_weight_5min",

    "future_deviation_pct",

    "future_off_spec",
]


# ============================================================
# 21. VERIFY FEATURE COLUMNS
# ============================================================

required_ml_columns = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
    +
    TARGET_COLUMNS
)


missing_ml_columns = [

    column

    for column
    in required_ml_columns

    if column not in df.columns
]


if missing_ml_columns:

    print(
        "\nMissing ML columns:"
    )

    for column in missing_ml_columns:

        print(
            "-",
            column
        )

    raise ValueError(
        "Feature engineering could not "
        "be completed."
    )


# ============================================================
# 22. CREATE FINAL ML DATASET
# ============================================================

ML_COLUMNS = (

    [
        "timestamp",
        "transition_id",
        "grade_transition",
    ]

    +
    NUMERIC_FEATURES

    +
    CATEGORICAL_FEATURES

    +
    TARGET_COLUMNS
)


ml_df = df[
    ML_COLUMNS
].copy()


# ============================================================
# 23. REMOVE INVALID ROWS
# ============================================================

before_rows = len(
    ml_df
)


ml_df = ml_df.dropna(
    subset=NUMERIC_FEATURES
    +
    TARGET_COLUMNS
)


after_rows = len(
    ml_df
)


print("\nRows before final cleaning:")
print(before_rows)

print("\nRows after final cleaning:")
print(after_rows)

print(
    "\nRows removed:",
    before_rows - after_rows
)


# ============================================================
# 24. SAVE ML DATASET
# ============================================================

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 25. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)


print(
    "\nFuture Basis Weight:"
)

print(
    ml_df[
        "future_basis_weight_5min"
    ].describe()
)


print(
    "\nFuture Deviation %:"
)

print(
    ml_df[
        "future_deviation_pct"
    ].describe()
)


print(
    "\nFuture OFF-SPEC:"
)

print(
    ml_df[
        "future_off_spec"
    ].value_counts()
)


print(
    "\nFuture OFF-SPEC percentage:"
)

print(
    (
        ml_df[
            "future_off_spec"
        ]
        .value_counts(
            normalize=True
        )
        *
        100
    ).round(2)
)


# ============================================================
# 26. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL ML DATASET")
print("=" * 80)


print(
    "\nFeature count:"
)

print(
    len(
        NUMERIC_FEATURES
        +
        CATEGORICAL_FEATURES
    )
)


print(
    "\nNumeric feature count:"
)

print(
    len(
        NUMERIC_FEATURES
    )
)


print(
    "\nCategorical feature count:"
)

print(
    len(
        CATEGORICAL_FEATURES
    )
)


print(
    "\nTarget columns:"
)

for target in TARGET_COLUMNS:

    print(
        "-",
        target
    )


print(
    "\nFinal ML dataset shape:"
)

print(
    ml_df.shape
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 27. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 80)
print("FIRST 5 ML RECORDS")
print("=" * 80)

print(
    ml_df.head().to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("STAGE 2 FEATURE ENGINEERING COMPLETED")
print("=" * 80)
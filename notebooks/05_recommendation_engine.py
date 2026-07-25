"""
STAGE 5 - SAFE RECOMMENDATION ENGINE

Paper Grade Transition Copilot
Honeywell Project

Purpose:
1. Load trained XGBoost model
2. Load ML features
3. Load original process limits
4. Combine current process state with process limits
5. Generate candidate control settings
6. Recalculate dependent features
7. Predict future Basis Weight
8. Calculate predicted deviation
9. Enforce <= 2.5% specification
10. Return a SAFE recommendation only when one exists
11. Otherwise report that no safe recommendation was found
"""

from pathlib import Path
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.csv"
)

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "basis_weight_xgboost.pkl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SPEC_LIMIT_PERCENT = 2.5

# How much each controllable variable can be changed
# around its current value when searching candidates.
SPEED_STEP = 10.0
STOCK_STEP = 2.0
STEAM_STEP = 0.1

# Maximum number of candidate values on each side
SPEED_RANGE = 3
STOCK_RANGE = 3
STEAM_RANGE = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_model(model_path):
    """
    Load the trained model.

    First try joblib because XGBoost/Scikit-learn pipelines
    are commonly saved using joblib.

    If that fails, try pickle.
    """

    errors = []

    try:
        model = joblib.load(model_path)
        return model
    except Exception as exc:
        errors.append(f"joblib: {exc}")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    raise RuntimeError(
        "Unable to load the trained model.\n"
        + "\n".join(errors)
    )


def find_model_feature_names(model):
    """
    Try to identify the feature names used during training.

    Works with:
    - sklearn pipelines
    - ColumnTransformer pipelines
    - XGBoost models
    """

    # Direct XGBoost feature names
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names

            if names is not None:
                return list(names)
        except Exception:
            pass

    # Pipeline
    if hasattr(model, "named_steps"):

        # Search the final estimator
        for _, step in reversed(model.named_steps.items()):

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

            if hasattr(step, "get_booster"):
                try:
                    names = step.get_booster().feature_names

                    if names is not None:
                        return list(names)
                except Exception:
                    pass

        # Search preprocessing transformer
        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def predict_model(model, dataframe):
    """
    Generate predictions using the trained model.
    """

    predictions = model.predict(dataframe)

    predictions = np.asarray(predictions).reshape(-1)

    return predictions


def calculate_deviation(predicted_basis_weight, setpoint):
    """
    Calculate percentage deviation from the target setpoint.
    """

    if setpoint == 0:
        return np.inf

    return (
        abs(predicted_basis_weight - setpoint)
        / abs(setpoint)
    ) * 100.0


def calculate_change_penalty(
    current_speed,
    current_stock,
    current_steam,
    new_speed,
    new_stock,
    new_steam,
):
    """
    Penalize large process changes.

    Smaller changes are preferred when multiple
    candidates are safe.
    """

    speed_penalty = abs(new_speed - current_speed) / SPEED_STEP

    stock_penalty = abs(new_stock - current_stock) / STOCK_STEP

    steam_penalty = abs(new_steam - current_steam) / STEAM_STEP

    return (
        speed_penalty
        + stock_penalty
        + steam_penalty
    )


def recalculate_features(row):
    """
    Recalculate features that depend on the controllable
    process variables.

    This is critical.

    If machine_speed, stock_flow or steam_pressure changes,
    dependent features must also change.
    """

    row = row.copy()

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

    if "machine_speed_min" in row and "machine_speed_max" in row:
        speed_range = row["machine_speed_max"] - row["machine_speed_min"]

        if speed_range != 0:
            row["machine_speed_position"] = (
                row["machine_speed"]
                - row["machine_speed_min"]
            ) / speed_range

            row["machine_speed_position_from_min"] = (
                row["machine_speed"]
                - row["machine_speed_min"]
            )

            row["machine_speed_position_from_max"] = (
                row["machine_speed_max"]
                - row["machine_speed"]
            )

    if "stock_flow_min" in row and "stock_flow_max" in row:
        stock_range = row["stock_flow_max"] - row["stock_flow_min"]

        if stock_range != 0:
            row["stock_flow_position"] = (
                row["stock_flow"]
                - row["stock_flow_min"]
            ) / stock_range

            row["stock_flow_position_from_min"] = (
                row["stock_flow"]
                - row["stock_flow_min"]
            )

            row["stock_flow_position_from_max"] = (
                row["stock_flow_max"]
                - row["stock_flow"]
            )

    if "steam_pressure_min" in row and "steam_pressure_max" in row:
        steam_range = (
            row["steam_pressure_max"]
            - row["steam_pressure_min"]
        )

        if steam_range != 0:
            row["steam_pressure_position"] = (
                row["steam_pressure"]
                - row["steam_pressure_min"]
            ) / steam_range

            row["steam_pressure_position_from_min"] = (
                row["steam_pressure"]
                - row["steam_pressure_min"]
            )

            row["steam_pressure_position_from_max"] = (
                row["steam_pressure_max"]
                - row["steam_pressure"]
            )

    if "filler_flow_min" in row and "filler_flow_max" in row:
        filler_range = (
            row["filler_flow_max"]
            - row["filler_flow_min"]
        )

        if filler_range != 0:
            row["filler_flow_position"] = (
                row["filler_flow"]
                - row["filler_flow_min"]
            ) / filler_range

            row["filler_flow_position_from_min"] = (
                row["filler_flow"]
                - row["filler_flow_min"]
            )

            row["filler_flow_position_from_max"] = (
                row["filler_flow_max"]
                - row["filler_flow"]
            )

    if "moisture_min" in row and "moisture_max" in row:
        moisture_range = (
            row["moisture_max"]
            - row["moisture_min"]
        )

        if moisture_range != 0:
            row["moisture_position"] = (
                row["moisture"]
                - row["moisture_min"]
            ) / moisture_range

            row["moisture_position_from_min"] = (
                row["moisture"]
                - row["moisture_min"]
            )

            row["moisture_position_from_max"] = (
                row["moisture_max"]
                - row["moisture"]
            )

    if "ash_min" in row and "ash_max" in row:
        ash_range = row["ash_max"] - row["ash_min"]

        if ash_range != 0:
            row["ash_position"] = (
                row["ash"]
                - row["ash_min"]
            ) / ash_range

            row["ash_position_from_min"] = (
                row["ash"]
                - row["ash_min"]
            )

            row["ash_position_from_max"] = (
                row["ash_max"]
                - row["ash"]
            )

    if "caliper_min" in row and "caliper_max" in row:
        caliper_range = (
            row["caliper_max"]
            - row["caliper_min"]
        )

        if caliper_range != 0:
            row["caliper_position"] = (
                row["caliper"]
                - row["caliper_min"]
            ) / caliper_range

            row["caliper_position_from_min"] = (
                row["caliper"]
                - row["caliper_min"]
            )

            row["caliper_position_from_max"] = (
                row["caliper_max"]
                - row["caliper"]
            )

    # --------------------------------------------------------
    # Ratio features
    # --------------------------------------------------------

    if row["stock_flow"] != 0:
        row["speed_stock_ratio"] = (
            row["machine_speed"]
            / row["stock_flow"]
        )

        row["filler_stock_ratio"] = (
            row["filler_flow"]
            / row["stock_flow"]
        )

    if row["moisture"] != 0:
        row["steam_moisture_ratio"] = (
            row["steam_pressure"]
            / row["moisture"]
        )

    # --------------------------------------------------------
    # Basis Weight error
    # --------------------------------------------------------

    row["basis_weight_error"] = (
        row["basis_weight"]
        - row["basis_weight_setpoint"]
    )

    if row["basis_weight_error"] > 0:
        row["basis_weight_error_direction"] = 1
    elif row["basis_weight_error"] < 0:
        row["basis_weight_error_direction"] = -1
    else:
        row["basis_weight_error_direction"] = 0

    # --------------------------------------------------------
    # Grade transition
    # --------------------------------------------------------

    if "grade_from" in row and "grade_to" in row:
        row["grade_changed"] = int(
            row["grade_from"] != row["grade_to"]
        )

    # --------------------------------------------------------
    # Operator adjustment
    # --------------------------------------------------------

    if "operator_action" in row:
        row["operator_adjustment_flag"] = int(
            row["operator_action"]
            == "corrective_setpoint_adjustment"
        )

    return row


def check_process_limits(row):
    """
    Verify that all controllable variables are inside
    their process limits.
    """

    checks = [
        (
            "machine_speed",
            "machine_speed_min",
            "machine_speed_max",
        ),
        (
            "stock_flow",
            "stock_flow_min",
            "stock_flow_max",
        ),
        (
            "steam_pressure",
            "steam_pressure_min",
            "steam_pressure_max",
        ),
    ]

    for value_col, min_col, max_col in checks:

        value = float(row[value_col])
        lower = float(row[min_col])
        upper = float(row[max_col])

        if value < lower or value > upper:
            return False

    return True


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 5 - SAFE RECOMMENDATION ENGINE")
print("=" * 80)


# ============================================================
# LOAD ML DATA
# ============================================================

print("\nLoading ML dataset:")
print(ML_DATA_PATH)

if not ML_DATA_PATH.exists():
    raise FileNotFoundError(
        f"ML dataset not found:\n{ML_DATA_PATH}"
    )

ml_df = pd.read_csv(ML_DATA_PATH)

print("\nML dataset loaded successfully.")
print(f"Rows    : {len(ml_df)}")
print(f"Columns : {len(ml_df.columns)}")


# ============================================================
# LOAD CLEANED DATA
# ============================================================

print("\nLoading original cleaned dataset:")
print(CLEANED_DATA_PATH)

if not CLEANED_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{CLEANED_DATA_PATH}"
    )

cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

print("\nOriginal cleaned dataset loaded successfully.")
print(f"Rows    : {len(cleaned_df)}")
print(f"Columns : {len(cleaned_df.columns)}")


# ============================================================
# PROCESS LIMIT COLUMNS
# ============================================================

LIMIT_COLUMNS = [
    "machine_speed_min",
    "machine_speed_max",

    "stock_flow_min",
    "stock_flow_max",

    "filler_flow_min",
    "filler_flow_max",

    "steam_pressure_min",
    "steam_pressure_max",

    "moisture_min",
    "moisture_max",

    "ash_min",
    "ash_max",

    "caliper_min",
    "caliper_max",
]

missing_limits = [
    col
    for col in LIMIT_COLUMNS
    if col not in cleaned_df.columns
]

if missing_limits:

    print("\nERROR: Process limits missing from cleaned_data.csv:")

    for col in missing_limits:
        print(f"- {col}")

    raise ValueError(
        "Required process limits are missing from cleaned_data.csv."
    )

print("\nAll process-limit columns found in cleaned_data.csv.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained XGBoost model:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# MODEL FEATURE INFORMATION
# ============================================================

model_features = find_model_feature_names(model)

if model_features is not None:

    print("\nModel feature count:")
    print(len(model_features))

else:

    print(
        "\nModel feature names could not be extracted directly."
    )


# ============================================================
# MERGE ML DATA WITH ORIGINAL PROCESS LIMITS
# ============================================================

print("\nCombining ML features with process limits...")

# timestamp is the safest row-level identifier in this dataset.
# transition_id is included as an additional identifier.

merge_keys = []

if "timestamp" in ml_df.columns and "timestamp" in cleaned_df.columns:
    merge_keys.append("timestamp")

if (
    "transition_id" in ml_df.columns
    and "transition_id" in cleaned_df.columns
):
    merge_keys.append("transition_id")


if not merge_keys:
    raise ValueError(
        "Could not find common keys between ml_features.csv "
        "and cleaned_data.csv."
    )


limit_df = cleaned_df[
    merge_keys + LIMIT_COLUMNS
].copy()


df = ml_df.merge(
    limit_df,
    on=merge_keys,
    how="left",
    validate="one_to_one",
)


# ============================================================
# VERIFY MERGE
# ============================================================

missing_after_merge = df[LIMIT_COLUMNS].isna().sum()

missing_after_merge = missing_after_merge[
    missing_after_merge > 0
]

if len(missing_after_merge) > 0:

    print("\nERROR: Process limits could not be merged.")

    print(missing_after_merge)

    raise ValueError(
        "Some process-limit values are missing after merge."
    )

print("Process limits successfully merged.")


# ============================================================
# SELECT CURRENT EXAMPLE
# ============================================================

# We use the first record for the demonstration.
# Later, the Flask API/dashboard will provide a live row.

current = df.iloc[0].copy()


# ============================================================
# CURRENT PROCESS
# ============================================================

print("\n")
print("=" * 80)
print("CURRENT PROCESS")
print("-" * 80)

print(
    f"Transition: {current['transition_id']}"
)

print(
    f"Grade: {current['grade_from']} → {current['grade_to']}"
)

print(
    f"Machine Speed: {current['machine_speed']:.3f}"
)

print(
    f"Stock Flow: {current['stock_flow']:.3f}"
)

print(
    f"Steam Pressure: {current['steam_pressure']:.3f}"
)

print(
    f"Current Basis Weight: {current['basis_weight']:.3f}"
)

print(
    f"Setpoint: {current['basis_weight_setpoint']}"
)


# ============================================================
# BASELINE MODEL FEATURES
# ============================================================

BASELINE_FEATURES = [
    col
    for col in ml_df.columns
    if col not in [
        "future_basis_weight_5min",
        "future_deviation_pct",
        "future_off_spec",
    ]
]


# Make sure current row has all ML features
missing_ml_features = [
    col
    for col in BASELINE_FEATURES
    if col not in current.index
]

if missing_ml_features:

    print("\nMissing ML features:")

    for col in missing_ml_features:
        print(f"- {col}")

    raise ValueError(
        "The current row does not contain all ML features."
    )


# ============================================================
# CURRENT PREDICTION
# ============================================================

baseline_input = pd.DataFrame(
    [current[BASELINE_FEATURES]]
)


# Remove future target columns if accidentally present
baseline_input = baseline_input.drop(
    columns=[
        "future_basis_weight_5min",
        "future_deviation_pct",
        "future_off_spec",
    ],
    errors="ignore",
)


try:

    current_prediction = float(
        predict_model(
            model,
            baseline_input
        )[0]
    )

except Exception as exc:

    print("\nModel prediction failed.")

    print("Error:")
    print(exc)

    print(
        "\nThis means the saved model expects a feature "
        "format different from ml_features.csv."
    )

    raise


current_deviation = calculate_deviation(
    current_prediction,
    current["basis_weight_setpoint"],
)


print("\n")
print("=" * 80)
print("CURRENT PREDICTION")
print("=" * 80)

print(
    f"Predicted future Basis Weight: "
    f"{current_prediction:.3f} GSM"
)

print(
    f"Predicted deviation: "
    f"{current_deviation:.3f} %"
)


if current_deviation <= SPEC_LIMIT_PERCENT:

    print("Current status: SAFE")

else:

    print("Current status: OFF-SPEC / HIGH RISK")


# ============================================================
# GENERATE CANDIDATE VALUES
# ============================================================

current_speed = float(
    current["machine_speed"]
)

current_stock = float(
    current["stock_flow"]
)

current_steam = float(
    current["steam_pressure"]
)


speed_min = float(
    current["machine_speed_min"]
)

speed_max = float(
    current["machine_speed_max"]
)

stock_min = float(
    current["stock_flow_min"]
)

stock_max = float(
    current["stock_flow_max"]
)

steam_min = float(
    current["steam_pressure_min"]
)

steam_max = float(
    current["steam_pressure_max"]
)


speed_candidates = [
    np.clip(
        current_speed + i * SPEED_STEP,
        speed_min,
        speed_max,
    )
    for i in range(
        -SPEED_RANGE,
        SPEED_RANGE + 1,
    )
]

stock_candidates = [
    np.clip(
        current_stock + i * STOCK_STEP,
        stock_min,
        stock_max,
    )
    for i in range(
        -STOCK_RANGE,
        STOCK_RANGE + 1,
    )
]

steam_candidates = [
    np.clip(
        current_steam + i * STEAM_STEP,
        steam_min,
        steam_max,
    )
    for i in range(
        -STEAM_RANGE,
        STEAM_RANGE + 1,
    )
]


# Remove duplicate values created by clipping
speed_candidates = sorted(
    set(round(x, 6) for x in speed_candidates)
)

stock_candidates = sorted(
    set(round(x, 6) for x in stock_candidates)
)

steam_candidates = sorted(
    set(round(x, 6) for x in steam_candidates)
)


# ============================================================
# GENERATE CANDIDATE DATA
# ============================================================

candidate_rows = []

for speed in speed_candidates:

    for stock in stock_candidates:

        for steam in steam_candidates:

            candidate = current.copy()

            candidate["machine_speed"] = speed
            candidate["stock_flow"] = stock
            candidate["steam_pressure"] = steam

            # Recalculate dependent features
            candidate = recalculate_features(
                candidate
            )

            # Check process limits
            if not check_process_limits(candidate):
                continue

            candidate_rows.append(candidate)


candidate_df = pd.DataFrame(candidate_rows)


print("\n")
print("=" * 80)
print("CANDIDATE SEARCH")
print("=" * 80)

print(
    f"Candidate settings evaluated: "
    f"{len(candidate_df)}"
)


# ============================================================
# PREPARE CANDIDATE ML FEATURES
# ============================================================

prediction_input = candidate_df[
    BASELINE_FEATURES
].copy()


# ============================================================
# PREDICT ALL CANDIDATES
# ============================================================

candidate_predictions = predict_model(
    model,
    prediction_input,
)


candidate_df[
    "predicted_basis_weight"
] = candidate_predictions


candidate_df[
    "predicted_deviation_pct"
] = candidate_df.apply(
    lambda row: calculate_deviation(
        row["predicted_basis_weight"],
        row["basis_weight_setpoint"],
    ),
    axis=1,
)


candidate_df["off_spec"] = (
    candidate_df["predicted_deviation_pct"]
    > SPEC_LIMIT_PERCENT
).astype(int)


# ============================================================
# CHANGE PENALTY
# ============================================================

candidate_df["change_penalty"] = candidate_df.apply(
    lambda row: calculate_change_penalty(
        current_speed,
        current_stock,
        current_steam,

        row["machine_speed"],
        row["stock_flow"],
        row["steam_pressure"],
    ),
    axis=1,
)


# ============================================================
# SAFE CANDIDATES
# ============================================================

safe_candidates = candidate_df[
    candidate_df["predicted_deviation_pct"]
    <= SPEC_LIMIT_PERCENT
].copy()


print("\n")
print("=" * 80)
print("SAFETY FILTER")
print("=" * 80)

print(
    f"Specification limit: "
    f"≤ {SPEC_LIMIT_PERCENT:.2f}%"
)

print(
    f"Safe candidates found: "
    f"{len(safe_candidates)}"
)


# ============================================================
# RECOMMENDATION
# ============================================================

if len(safe_candidates) > 0:

    # First minimize deviation.
    # Then minimize process change.

    safe_candidates = safe_candidates.sort_values(
        by=[
            "predicted_deviation_pct",
            "change_penalty",
        ],
        ascending=[
            True,
            True,
        ],
    )

    recommendation = safe_candidates.iloc[0]

    print("\n")
    print("=" * 80)
    print("AI RECOMMENDATION")
    print("=" * 80)

    print(
        f"Machine Speed: "
        f"{current_speed:.3f} → "
        f"{recommendation['machine_speed']:.3f}"
    )

    print(
        f"Stock Flow: "
        f"{current_stock:.3f} → "
        f"{recommendation['stock_flow']:.3f}"
    )

    print(
        f"Steam Pressure: "
        f"{current_steam:.3f} → "
        f"{recommendation['steam_pressure']:.3f}"
    )

    print(
        f"Expected future Basis Weight: "
        f"{recommendation['predicted_basis_weight']:.3f} GSM"
    )

    print(
        f"Expected deviation: "
        f"{recommendation['predicted_deviation_pct']:.3f} %"
    )

    print("Status: SAFE")

    print(
        "\nRecommendation rationale:"
    )

    print(
        "The recommended settings are within the "
        "current process limits and are predicted to "
        "keep Basis Weight within the ±2.5% specification."
    )

    print(
        "\nChange penalty:"
        f" {recommendation['change_penalty']:.3f}"
    )

else:

    # No safe candidate exists.
    # Find the least-deviating candidate only for
    # diagnostic purposes.

    best_candidate = candidate_df.sort_values(
        by=[
            "predicted_deviation_pct",
            "change_penalty",
        ],
        ascending=[
            True,
            True,
        ],
    ).iloc[0]

    print("\n")
    print("=" * 80)
    print("⚠ NO SAFE RECOMMENDATION FOUND")
    print("=" * 80)

    print(
        f"Current predicted Basis Weight: "
        f"{current_prediction:.3f} GSM"
    )

    print(
        f"Target Basis Weight: "
        f"{current['basis_weight_setpoint']:.3f} GSM"
    )

    print(
        f"Current predicted deviation: "
        f"{current_deviation:.3f} %"
    )

    print(
        f"Required deviation: "
        f"≤ {SPEC_LIMIT_PERCENT:.2f} %"
    )

    print("\nBest achievable candidate within tested limits:")

    print(
        f"Machine Speed: "
        f"{best_candidate['machine_speed']:.3f}"
    )

    print(
        f"Stock Flow: "
        f"{best_candidate['stock_flow']:.3f}"
    )

    print(
        f"Steam Pressure: "
        f"{best_candidate['steam_pressure']:.3f}"
    )

    print(
        f"Predicted Basis Weight: "
        f"{best_candidate['predicted_basis_weight']:.3f} GSM"
    )

    print(
        f"Predicted deviation: "
        f"{best_candidate['predicted_deviation_pct']:.3f} %"
    )

    print(
        "\nConclusion:"
    )

    print(
        "No combination of the tested controllable "
        "variables within their current operating limits "
        "is predicted to achieve the required "
        "Basis Weight specification."
    )

    print(
        "\nRecommended action:"
    )

    print(
        "Escalate to operator / consider recipe-level "
        "or additional process-variable adjustment."
    )


# ============================================================
# TOP CANDIDATES
# ============================================================

print("\n")
print("=" * 80)
print("TOP CANDIDATES")
print("=" * 80)

top_candidates = candidate_df.sort_values(
    by=[
        "predicted_deviation_pct",
        "change_penalty",
    ],
    ascending=[
        True,
        True,
    ],
).head(10)


display_columns = [
    "machine_speed",
    "stock_flow",
    "steam_pressure",
    "predicted_basis_weight",
    "predicted_deviation_pct",
    "off_spec",
    "change_penalty",
]


print(
    top_candidates[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

all_candidates_path = (
    RESULTS_DIR
    / "recommendation_candidates.csv"
)

top_candidates_path = (
    RESULTS_DIR
    / "top_recommendations.csv"
)


candidate_df[
    [
        "machine_speed",
        "stock_flow",
        "steam_pressure",
        "predicted_basis_weight",
        "predicted_deviation_pct",
        "off_spec",
        "change_penalty",
    ]
].to_csv(
    all_candidates_path,
    index=False,
)


top_candidates[
    display_columns
].to_csv(
    top_candidates_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("STAGE 5 COMPLETED")
print("=" * 80)

print(
    f"Total candidates evaluated: "
    f"{len(candidate_df)}"
)

print(
    f"Safe candidates: "
    f"{len(safe_candidates)}"
)

print(
    f"Safety threshold: "
    f"{SPEC_LIMIT_PERCENT:.2f}%"
)

print(
    "\nAll candidate results saved to:"
)

print(
    all_candidates_path
)

print(
    "\nTop recommendations saved to:"
)

print(
    top_candidates_path
)

print("\n")
print("=" * 80)
print("READY FOR STAGE 6 - STABILIZATION + EVIDENCE")
print("=" * 80)
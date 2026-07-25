# ============================================================
# PAPER GRADE TRANSITION COPILOT
# Stage 3 - ML Model Training
#
# Models:
#   1. XGBoost Regression
#   2. XGBoost Classification
#
# Important:
#   Train/test split is performed by transition_id
#   to prevent transition-level data leakage.
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestRegressor

from xgboost import (
    XGBRegressor,
    XGBClassifier,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("STAGE 3 - ML MODEL TRAINING")
print("=" * 80)

print("\nLoading:")
print(DATA_PATH)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Run 02_feature_engineering.py first."
    )


df = pd.read_csv(DATA_PATH)


print("\nDataset loaded successfully.")

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)


# ============================================================
# 3. TARGETS
# ============================================================

REGRESSION_TARGET = (
    "future_basis_weight_5min"
)

CLASSIFICATION_TARGET = (
    "future_off_spec"
)


# ============================================================
# 4. FEATURE LIST
# ============================================================
#
# We explicitly define features instead of blindly passing
# every column to the model.
#
# This prevents future targets and identifiers from becoming
# accidental model inputs.
# ============================================================

NUMERIC_FEATURES = [

    "machine_speed",
    "stock_flow",
    "filler_flow",
    "steam_pressure",
    "moisture",
    "ash",
    "caliper",

    "basis_weight",
    "basis_weight_setpoint",

    "basis_weight_lag_1",
    "basis_weight_lag_5",

    "basis_weight_change_5min",

    "speed_change_5min",
    "stock_flow_change_5min",

    "machine_speed_position",
    "stock_flow_position",
    "filler_flow_position",
    "steam_pressure_position",
    "moisture_position",
    "ash_position",
    "caliper_position",

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

    "grade_changed",

    "transition_phase_encoded",

    "operator_adjustment_flag",

    "speed_stock_ratio",

    "steam_moisture_ratio",

    "filler_stock_ratio",

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


FEATURES = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)


# ============================================================
# 5. VERIFY COLUMNS
# ============================================================

required_columns = (
    FEATURES
    +
    [
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET,
        "transition_id",
    ]
)


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print("\nMissing columns:")

    for col in missing_columns:
        print("-", col)

    raise ValueError(
        "Required columns are missing."
    )


# ============================================================
# 6. REMOVE INVALID ROWS
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


df = df.dropna(
    subset=FEATURES
    +
    [
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET,
    ]
).copy()


print(
    "\nRows after cleaning:",
    len(df)
)


# ============================================================
# 7. TRANSITION-LEVEL TRAIN/TEST SPLIT
# ============================================================
#
# We deliberately split by transition_id.
#
# This prevents rows from the same physical transition from
# appearing in both training and testing.
# ============================================================

unique_transitions = (
    df["transition_id"]
    .drop_duplicates()
    .to_numpy()
)


rng = np.random.default_rng(
    42
)


rng.shuffle(
    unique_transitions
)


split_index = int(
    len(unique_transitions) * 0.80
)


train_transition_ids = (
    unique_transitions[
        :split_index
    ]
)


test_transition_ids = (
    unique_transitions[
        split_index:
    ]
)


train_df = df[
    df["transition_id"].isin(
        train_transition_ids
    )
].copy()


test_df = df[
    df["transition_id"].isin(
        test_transition_ids
    )
].copy()


print("\n" + "=" * 80)
print("TRANSITION-LEVEL SPLIT")
print("=" * 80)

print(
    "\nTotal transitions:",
    len(unique_transitions)
)

print(
    "Training transitions:",
    len(train_transition_ids)
)

print(
    "Testing transitions:",
    len(test_transition_ids)
)

print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


# ============================================================
# 8. VERIFY NO TRANSITION OVERLAP
# ============================================================

overlap = set(
    train_transition_ids
).intersection(
    set(test_transition_ids)
)


print(
    "\nTransition overlap:",
    len(overlap)
)


if len(overlap) != 0:

    raise RuntimeError(
        "DATA LEAKAGE DETECTED: "
        "train/test transition IDs overlap."
    )


print(
    "No transition leakage detected."
)


# ============================================================
# 9. PREPARE X / y
# ============================================================

X_train = train_df[
    FEATURES
].copy()


X_test = test_df[
    FEATURES
].copy()


y_train_reg = train_df[
    REGRESSION_TARGET
].copy()


y_test_reg = test_df[
    REGRESSION_TARGET
].copy()


y_train_cls = train_df[
    CLASSIFICATION_TARGET
].copy()


y_test_cls = test_df[
    CLASSIFICATION_TARGET
].copy()


# ============================================================
# 10. ONE-HOT ENCODING
# ============================================================
#
# XGBoost needs numerical input.
#
# Categorical variables such as:
#
# grade_from = A/B/C/D
#
# are converted using OneHotEncoder.
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            CATEGORICAL_FEATURES
        ),

        (
            "numeric",

            "passthrough",

            NUMERIC_FEATURES
        ),
    ]
)


# ============================================================
# 11. XGBOOST REGRESSION MODEL
# ============================================================

regressor = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


regression_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            regressor
        ),
    ]
)


# ============================================================
# 12. TRAIN REGRESSION MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST REGRESSOR")
print("=" * 80)


regression_pipeline.fit(
    X_train,
    y_train_reg
)


print(
    "\nRegression model trained successfully."
)


# ============================================================
# 13. REGRESSION PREDICTION
# ============================================================

reg_predictions = (
    regression_pipeline.predict(
        X_test
    )
)


# ============================================================
# 14. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_reg,
    reg_predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        reg_predictions
    )
)


r2 = r2_score(
    y_test_reg,
    reg_predictions
)


print("\n" + "=" * 80)
print("REGRESSION RESULTS")
print("=" * 80)

print(
    f"\nMAE  : {mae:.4f} GSM"
)

print(
    f"RMSE : {rmse:.4f} GSM"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 15. XGBOOST CLASSIFIER
# ============================================================
#
# Class imbalance:
#
# SAFE = 0
# OFF-SPEC = 1
#
# We calculate scale_pos_weight from the training data.
# ============================================================

negative_count = (
    (y_train_cls == 0)
    .sum()
)


positive_count = (
    (y_train_cls == 1)
    .sum()
)


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print("\n" + "=" * 80)
print("CLASSIFICATION DATA")
print("=" * 80)

print(
    "SAFE training rows:",
    negative_count
)

print(
    "OFF-SPEC training rows:",
    positive_count
)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


classifier = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


classification_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            ColumnTransformer(

                transformers=[

                    (
                        "categorical",

                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        ),

                        CATEGORICAL_FEATURES
                    ),

                    (
                        "numeric",

                        "passthrough",

                        NUMERIC_FEATURES
                    ),
                ]
            )
        ),

        (
            "model",
            classifier
        ),
    ]
)


# ============================================================
# 16. TRAIN CLASSIFIER
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST CLASSIFIER")
print("=" * 80)


classification_pipeline.fit(
    X_train,
    y_train_cls
)


print(
    "\nClassification model trained successfully."
)


# ============================================================
# 17. CLASSIFICATION PREDICTION
# ============================================================

class_predictions = (
    classification_pipeline.predict(
        X_test
    )
)


class_probabilities = (
    classification_pipeline.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 18. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_cls,
    class_predictions
)


precision = precision_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)


recall = recall_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)


f1 = f1_score(
    y_test_cls,
    class_predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    y_test_cls,
    class_probabilities
)


cm = confusion_matrix(
    y_test_cls,
    class_predictions
)


print("\n" + "=" * 80)
print("CLASSIFICATION RESULTS")
print("=" * 80)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC AUC   : {roc_auc:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test_cls,
        class_predictions,
        target_names=[
            "SAFE",
            "OFF_SPEC"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. CALCULATE PREDICTED DEVIATION
# ============================================================

predicted_setpoints = (
    test_df[
        "basis_weight_setpoint"
    ].to_numpy()
)


predicted_deviation = (

    np.abs(
        reg_predictions
        -
        predicted_setpoints
    )

    /

    predicted_setpoints

    *

    100
)


predicted_off_spec_from_regression = (
    predicted_deviation > 2.5
).astype(int)


# ============================================================
# 20. COMPARE REGRESSION-DERIVED OFF-SPEC WITH TRUE LABEL
# ============================================================

deviation_accuracy = accuracy_score(

    y_test_cls,

    predicted_off_spec_from_regression
)


print("\n" + "=" * 80)
print("REGRESSION → OFF-SPEC VALIDATION")
print("=" * 80)

print(
    f"\nOff-spec accuracy based on "
    f"predicted Basis Weight: "
    f"{deviation_accuracy:.4f}"
)


# ============================================================
# 21. CREATE PREDICTION RESULTS TABLE
# ============================================================

results = test_df[
    [
        "timestamp",
        "transition_id",
        "grade_from",
        "grade_to",
        "basis_weight",
        "basis_weight_setpoint",
        "future_basis_weight_5min",
        "future_deviation_pct",
        "future_off_spec",
    ]
].copy()


results[
    "predicted_future_basis_weight"
] = reg_predictions


results[
    "predicted_deviation_pct"
] = predicted_deviation


results[
    "predicted_off_spec"
] = class_predictions


results[
    "off_spec_probability"
] = class_probabilities


results[
    "prediction_error_gsm"
] = (

    results[
        "predicted_future_basis_weight"
    ]

    -

    results[
        "future_basis_weight_5min"
    ]
)


# ============================================================
# 22. SAVE PREDICTIONS
# ============================================================

prediction_path = (
    RESULT_DIR
    /
    "test_predictions.csv"
)


results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

metrics = {

    "dataset_rows": int(len(df)),

    "total_transitions":
        int(len(unique_transitions)),

    "training_transitions":
        int(len(train_transition_ids)),

    "testing_transitions":
        int(len(test_transition_ids)),

    "training_rows":
        int(len(train_df)),

    "testing_rows":
        int(len(test_df)),

    "regression": {

        "MAE_GSM":
            float(mae),

        "RMSE_GSM":
            float(rmse),

        "R2":
            float(r2),
    },

    "classification": {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "F1":
            float(f1),

        "ROC_AUC":
            float(roc_auc),
    },

    "regression_to_off_spec_accuracy":
        float(deviation_accuracy),
}


metrics_path = (
    RESULT_DIR
    /
    "model_metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 24. SAVE MODELS
# ============================================================
#
# We use joblib to save the complete pipelines, including
# preprocessing + XGBoost.
# ============================================================

import joblib


regression_model_path = (
    MODEL_DIR
    /
    "basis_weight_xgboost.pkl"
)


classification_model_path = (
    MODEL_DIR
    /
    "off_spec_xgboost.pkl"
)


joblib.dump(
    regression_pipeline,
    regression_model_path
)


joblib.dump(
    classification_pipeline,
    classification_model_path
)


# ============================================================
# 25. SAVE TRAIN/TEST TRANSITION IDS
# ============================================================

split_info = {

    "random_seed": 42,

    "train_transition_ids":
        train_transition_ids.tolist(),

    "test_transition_ids":
        test_transition_ids.tolist(),
}


split_path = (
    RESULT_DIR
    /
    "transition_split.json"
)


with open(
    split_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        split_info,
        file,
        indent=4
    )


# ============================================================
# 26. SHOW EXAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)


display_columns = [

    "transition_id",

    "grade_from",

    "grade_to",

    "basis_weight",

    "basis_weight_setpoint",

    "future_basis_weight_5min",

    "predicted_future_basis_weight",

    "future_deviation_pct",

    "predicted_deviation_pct",

    "future_off_spec",

    "predicted_off_spec",

    "off_spec_probability",
]


print(
    results[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("STAGE 3 COMPLETED")
print("=" * 80)


print(
    "\nRegression model:"
)

print(
    regression_model_path
)


print(
    "\nClassification model:"
)

print(
    classification_model_path
)


print(
    "\nMetrics:"
)

print(
    metrics_path
)


print(
    "\nPredictions:"
)

print(
    prediction_path
)


print(
    "\nTransition split:"
)

print(
    split_path
)


print("\n" + "=" * 80)
print("READY FOR STAGE 4 - SHAP + EXPLAINABILITY")
print("=" * 80)
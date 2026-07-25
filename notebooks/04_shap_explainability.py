# ============================================================
# PAPER GRADE TRANSITION COPILOT
# Stage 4 - SHAP Explainability
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "basis_weight_xgboost.pkl"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
    / "shap_feature_importance.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("\n" + "=" * 80)
print("STAGE 4 - SHAP EXPLAINABILITY")
print("=" * 80)

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print(
    "\nDataset rows:",
    len(df)
)


print("\nLoading trained model:")
print(MODEL_PATH)

pipeline = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# 3. FEATURES
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


X = df[
    FEATURES
].copy()


# ============================================================
# 4. GET PREPROCESSOR AND XGBOOST MODEL
# ============================================================

preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)

model = (
    pipeline.named_steps[
        "model"
    ]
)


# ============================================================
# 5. TRANSFORM FEATURES
# ============================================================

print("\nTransforming features...")

X_transformed = (
    preprocessor.transform(X)
)


print(
    "Transformed shape:",
    X_transformed.shape
)


# ============================================================
# 6. GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print(
    "Number of transformed features:",
    len(feature_names)
)


# ============================================================
# 7. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# 8. CALCULATE SHAP VALUES
# ============================================================

print(
    "\nCalculating SHAP values..."
)

shap_values = (
    explainer.shap_values(
        X_transformed
    )
)


print(
    "SHAP calculation completed."
)


# ============================================================
# 9. GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = (
    np.abs(shap_values)
    .mean(axis=0)
)


importance_df = pd.DataFrame({

    "feature":
        feature_names,

    "mean_absolute_shap":
        mean_abs_shap

})


importance_df = (
    importance_df
    .sort_values(
        "mean_absolute_shap",
        ascending=False
    )
)


# ============================================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    RESULT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("TOP SHAP FEATURES")
print("=" * 80)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 11. SHAP SUMMARY PLOT
# ============================================================

print(
    "\nGenerating SHAP summary plot..."
)


plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

summary_path = (
    FIGURE_DIR
    /
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = (
    FIGURE_DIR
    /
    "shap_feature_importance.png"
)

plt.savefig(
    bar_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. LOCAL EXPLANATION FOR AN OFF-SPEC CASE
# ============================================================

print(
    "\nSearching for an OFF-SPEC example..."
)


off_spec_indices = df.index[
    df["future_off_spec"] == 1
]


if len(off_spec_indices) > 0:

    example_index = (
        off_spec_indices[0]
    )

    position = (
        df.index.get_loc(
            example_index
        )
    )

    example_row = df.loc[
        example_index
    ]

    example_shap = (
        shap_values[position]
    )


    local_df = pd.DataFrame({

        "feature":
            feature_names,

        "shap_value":
            example_shap,

        "absolute_shap":
            np.abs(example_shap)

    })


    local_df = (
        local_df
        .sort_values(
            "absolute_shap",
            ascending=False
        )
    )


    print("\n" + "=" * 80)
    print("LOCAL EXPLANATION")
    print("=" * 80)

    print(
        "\nTransition:",
        example_row[
            "transition_id"
        ]
    )

    print(
        "Grade:",
        example_row[
            "grade_from"
        ],
        "→",
        example_row[
            "grade_to"
        ]
    )

    print(
        "Current Basis Weight:",
        example_row[
            "basis_weight"
        ]
    )

    print(
        "Setpoint:",
        example_row[
            "basis_weight_setpoint"
        ]
    )

    print(
        "Future Basis Weight:",
        example_row[
            "future_basis_weight_5min"
        ]
    )

    print(
        "\nTop contributing features:"
    )

    print(
        local_df
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("STAGE 4 COMPLETED")
print("=" * 80)

print(
    "\nSHAP importance:"
)

print(
    RESULT_PATH
)

print(
    "\nSHAP summary:"
)

print(
    summary_path
)

print(
    "\nSHAP bar chart:"
)

print(
    bar_path
)

print(
    "\nReady for Stage 5 - Recommendation Engine."
)
# ============================================================
# PAPER GRADE TRANSITION COPILOT
# Stage 1 - Exploratory Data Analysis (EDA)
# ============================================================
#
# Purpose:
# 1. Load and validate the paper manufacturing dataset
# 2. Inspect columns and data types
# 3. Check missing values and duplicates
# 4. Analyze Basis Weight and its setpoint
# 5. Analyze the 2.5% off-spec condition
# 6. Analyze grade transitions
# 7. Analyze successful and failed transitions
# 8. Analyze operator actions
# 9. Analyze process variables
# 10. Calculate Pearson, Spearman and Mutual Information
# 11. Analyze transition-specific correlations
# 12. Generate useful charts
# 13. Save cleaned data for Stage 2
#
# IMPORTANT:
# We explicitly use the correct dataset columns.
# We DO NOT automatically search for a Basis Weight column.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Current file:
#
# paper-grade-copilot/
# └── notebooks/
#     └── 01_eda.py
#
# parent       = notebooks
# parent.parent = paper-grade-copilot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# INPUT DATA
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paper_process.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
)


# Create directories if they do not exist.

PROCESSED_PATH.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CHECK DATASET
# ============================================================

print("\n" + "=" * 80)
print("PAPER GRADE TRANSITION COPILOT - EDA")
print("=" * 80)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset path:")
print(DATA_PATH)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found!\n"
        f"Expected location:\n{DATA_PATH}\n\n"
        f"Make sure paper_process.csv is inside:\n"
        f"data/raw/"
    )


print("\nDataset found successfully.")


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)


print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 5. COLUMN NAMES
# ============================================================

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)

for i, column in enumerate(df.columns, start=1):

    print(
        f"{i:02d}. {column}"
    )


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(
    df.dtypes.to_string()
)


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing_values = (
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(missing_values)


# ============================================================
# 8. DUPLICATE ROWS
# ============================================================

duplicate_count = (
    df.duplicated()
    .sum()
)

print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)

print(
    f"Duplicate rows: {duplicate_count}"
)


# ============================================================
# 9. NUMERIC COLUMNS
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

print("\n" + "=" * 80)
print("NUMERIC COLUMNS")
print("=" * 80)

for column in numeric_columns:

    print(
        "-",
        column
    )


# ============================================================
# 10. CATEGORICAL COLUMNS
# ============================================================

categorical_columns = (
    df.select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)

print("\n" + "=" * 80)
print("CATEGORICAL COLUMNS")
print("=" * 80)

for column in categorical_columns:

    print(
        "-",
        column
    )


# ============================================================
# 11. VERIFY IMPORTANT PROJECT COLUMNS
# ============================================================
#
# We explicitly define the correct columns.
#
# This prevents the previous problem where:
#
# basis_weight_setpoint
#
# was incorrectly selected instead of:
#
# basis_weight
# ============================================================

REQUIRED_COLUMNS = [

    # Time
    "timestamp",

    # Grade transition
    "transition_id",
    "grade_from",
    "grade_to",
    "recipe_id",
    "transition_phase",
    "transition_outcome",
    "operator_action",

    # Process variables
    "machine_speed",
    "stock_flow",
    "filler_flow",
    "steam_pressure",
    "moisture",
    "ash",
    "caliper",

    # Basis Weight
    "basis_weight_setpoint",
    "basis_weight",

    # Historical / lag variables
    "basis_weight_lag_1",
    "basis_weight_lag_5",
    "basis_weight_change_5min",
    "speed_change_5min",
    "stock_flow_change_5min",

    # Future target
    "future_basis_weight_5min",

    # Quality / outcome
    "deviation_pct",
    "off_spec",

    # Stabilization
    "stabilization_time_min",
]


missing_required = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]


if missing_required:

    print(
        "\nThe following required columns are missing:"
    )

    for column in missing_required:

        print(
            "-",
            column
        )

    raise ValueError(
        "\nDataset does not contain all required "
        "columns."
    )


print("\n" + "=" * 80)
print("IMPORTANT PROJECT COLUMNS")
print("=" * 80)

print(
    "\nBasis Weight column:"
)

print(
    "basis_weight"
)

print(
    "\nBasis Weight setpoint:"
)

print(
    "basis_weight_setpoint"
)

print(
    "\nDeviation column:"
)

print(
    "deviation_pct"
)

print(
    "\nOff-spec column:"
)

print(
    "off_spec"
)

print(
    "\nFuture Basis Weight:"
)

print(
    "future_basis_weight_5min"
)

print(
    "\nTransition outcome:"
)

print(
    "transition_outcome"
)

print(
    "\nOperator action:"
)

print(
    "operator_action"
)


# ============================================================
# 12. CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# 13. SORT DATA
# ============================================================

df = df.sort_values(
    [
        "transition_id",
        "timestamp"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 14. BASIS WEIGHT ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("BASIS WEIGHT ANALYSIS")
print("=" * 80)


print("\nCurrent Basis Weight:")
print(
    df["basis_weight"]
    .describe()
)


print("\nBasis Weight Setpoint:")
print(
    df["basis_weight_setpoint"]
    .describe()
)


print("\nExisting Deviation %:")
print(
    df["deviation_pct"]
    .describe()
)


# ============================================================
# 15. VERIFY DEVIATION CALCULATION
# ============================================================
#
# We already have deviation_pct in the dataset.
#
# We calculate an independent verification column
# ONLY for checking correctness.
#
# This is NOT used as the main deviation column.
# ============================================================

calculated_deviation_pct = (
    (
        df["basis_weight"]
        - df["basis_weight_setpoint"]
    )
    .abs()
    /
    df["basis_weight_setpoint"]
    * 100
)


deviation_difference = (
    calculated_deviation_pct
    - df["deviation_pct"]
).abs()


print("\n" + "=" * 80)
print("DEVIATION VALIDATION")
print("=" * 80)

print(
    "\nMaximum difference between "
    "dataset deviation and independently "
    "calculated deviation:"
)

print(
    deviation_difference.max()
)


if deviation_difference.max() < 0.001:

    print(
        "\nDeviation calculation is consistent."
    )

else:

    print(
        "\nWARNING: deviation_pct differs "
        "from the calculated value."
    )


# ============================================================
# 16. 2.5% OFF-SPEC ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("2.5% OFF-SPEC ANALYSIS")
print("=" * 80)


off_spec_counts = (
    df["off_spec"]
    .value_counts()
    .sort_index()
)


print("\nOff-spec counts:")

print(
    off_spec_counts
)


print("\nOff-spec percentages:")

off_spec_percentages = (
    df["off_spec"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print(
    off_spec_percentages.round(2)
)


# ============================================================
# 17. VERIFY OFF-SPEC LABEL
# ============================================================
#
# Our project rule:
#
# deviation > 2.5%
#
# => OFF-SPEC
#
# deviation <= 2.5%
#
# => SAFE
# ============================================================

calculated_off_spec = (
    df["deviation_pct"]
    > 2.5
).astype(int)


off_spec_mismatch = (
    calculated_off_spec
    != df["off_spec"]
).sum()


print(
    "\nOff-spec label mismatches:"
)

print(
    off_spec_mismatch
)


if off_spec_mismatch == 0:

    print(
        "OFF-SPEC labels correctly follow "
        "the 2.5% rule."
    )

else:

    print(
        "WARNING: OFF-SPEC labels do not "
        "completely match the 2.5% rule."
    )


# ============================================================
# 18. FUTURE BASIS WEIGHT
# ============================================================

print("\n" + "=" * 80)
print("FUTURE BASIS WEIGHT")
print("=" * 80)

print(
    df["future_basis_weight_5min"]
    .describe()
)


# Calculate future deviation.

df["future_deviation_pct"] = (
    (
        df["future_basis_weight_5min"]
        - df["basis_weight_setpoint"]
    )
    .abs()
    /
    df["basis_weight_setpoint"]
    * 100
)


# Future off-spec target.

df["future_off_spec"] = (
    df["future_deviation_pct"]
    > 2.5
).astype(int)


print(
    "\nFuture OFF-SPEC distribution:"
)

print(
    df["future_off_spec"]
    .value_counts()
)


print(
    "\nFuture OFF-SPEC percentage:"
)

print(
    (
        df["future_off_spec"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# 19. TRANSITION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("GRADE TRANSITION ANALYSIS")
print("=" * 80)


print(
    "\nNumber of unique transitions:"
)

print(
    df["transition_id"]
    .nunique()
)


print(
    "\nGrade From distribution:"
)

print(
    df["grade_from"]
    .value_counts()
)


print(
    "\nGrade To distribution:"
)

print(
    df["grade_to"]
    .value_counts()
)


# ============================================================
# 20. GRADE TRANSITION PAIRS
# ============================================================

df["grade_transition"] = (
    df["grade_from"].astype(str)
    + " → "
    + df["grade_to"].astype(str)
)


print(
    "\nGrade transition pairs:"
)

print(
    df["grade_transition"]
    .value_counts()
)


# ============================================================
# 21. TRANSITION PHASE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("TRANSITION PHASE")
print("=" * 80)

print(
    df["transition_phase"]
    .value_counts()
)


# ============================================================
# 22. SUCCESS / FAILURE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SUCCESS / FAILURE ANALYSIS")
print("=" * 80)


print(
    "\nTransition outcomes:"
)

print(
    df["transition_outcome"]
    .value_counts()
)


print(
    "\nTransition outcome percentages:"
)

print(
    (
        df["transition_outcome"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# 23. OPERATOR ACTION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("OPERATOR ACTION ANALYSIS")
print("=" * 80)


print(
    df["operator_action"]
    .value_counts()
)


print(
    "\nOperator action percentages:"
)

print(
    (
        df["operator_action"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# 24. STABILIZATION TIME
# ============================================================

print("\n" + "=" * 80)
print("STABILIZATION TIME")
print("=" * 80)


print(
    df["stabilization_time_min"]
    .describe()
)


# ============================================================
# 25. PROCESS VARIABLES
# ============================================================

PROCESS_VARIABLES = [

    "machine_speed",
    "stock_flow",
    "filler_flow",
    "steam_pressure",
    "moisture",
    "ash",
    "caliper",
]


print("\n" + "=" * 80)
print("PROCESS VARIABLE STATISTICS")
print("=" * 80)


print(
    df[PROCESS_VARIABLES]
    .describe()
    .T
)


# ============================================================
# 26. PEARSON CORRELATION
# ============================================================
#
# Pearson measures linear relationship.
# ============================================================

CORRELATION_TARGET = (
    "future_basis_weight_5min"
)


pearson_results = []


for feature in PROCESS_VARIABLES:

    valid_data = df[
        [
            feature,
            CORRELATION_TARGET
        ]
    ].dropna()


    if len(valid_data) > 2:

        correlation, p_value = pearsonr(
            valid_data[feature],
            valid_data[
                CORRELATION_TARGET
            ]
        )


        pearson_results.append({

            "feature": feature,

            "pearson_correlation":
                correlation,

            "p_value":
                p_value

        })


pearson_df = pd.DataFrame(
    pearson_results
)


pearson_df = (
    pearson_df
    .sort_values(
        "pearson_correlation",
        key=abs,
        ascending=False
    )
)


print("\n" + "=" * 80)
print("PEARSON CORRELATION WITH FUTURE BASIS WEIGHT")
print("=" * 80)

print(
    pearson_df.to_string(
        index=False
    )
)


# ============================================================
# 27. SPEARMAN CORRELATION
# ============================================================
#
# Spearman measures monotonic relationships.
# ============================================================

spearman_results = []


for feature in PROCESS_VARIABLES:

    valid_data = df[
        [
            feature,
            CORRELATION_TARGET
        ]
    ].dropna()


    if len(valid_data) > 2:

        correlation, p_value = spearmanr(
            valid_data[feature],
            valid_data[
                CORRELATION_TARGET
            ]
        )


        spearman_results.append({

            "feature": feature,

            "spearman_correlation":
                correlation,

            "p_value":
                p_value

        })


spearman_df = pd.DataFrame(
    spearman_results
)


spearman_df = (
    spearman_df
    .sort_values(
        "spearman_correlation",
        key=abs,
        ascending=False
    )
)


print("\n" + "=" * 80)
print("SPEARMAN CORRELATION WITH FUTURE BASIS WEIGHT")
print("=" * 80)

print(
    spearman_df.to_string(
        index=False
    )
)


# ============================================================
# 28. MUTUAL INFORMATION
# ============================================================
#
# Mutual Information can detect nonlinear dependencies.
# ============================================================

mi_data = df[
    PROCESS_VARIABLES
    + [CORRELATION_TARGET]
].dropna()


X_mi = mi_data[
    PROCESS_VARIABLES
]

y_mi = mi_data[
    CORRELATION_TARGET
]


mi_scores = mutual_info_regression(
    X_mi,
    y_mi,
    random_state=42
)


mi_df = pd.DataFrame({

    "feature":
        PROCESS_VARIABLES,

    "mutual_information":
        mi_scores

})


mi_df = (
    mi_df
    .sort_values(
        "mutual_information",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("MUTUAL INFORMATION")
print("=" * 80)

print(
    mi_df.to_string(
        index=False
    )
)


# ============================================================
# 29. COMBINED CORRELATION RESULTS
# ============================================================

correlation_results = (
    pearson_df[
        [
            "feature",
            "pearson_correlation"
        ]
    ]
    .merge(
        spearman_df[
            [
                "feature",
                "spearman_correlation"
            ]
        ],
        on="feature"
    )
    .merge(
        mi_df,
        on="feature"
    )
)


correlation_results.to_csv(
    RESULTS_PATH
    / "correlation_results.csv",
    index=False
)


# ============================================================
# 30. CORRELATION DURING GRADE TRANSITIONS
# ============================================================
#
# This is especially important for the Honeywell project.
#
# We compare:
#
# Normal/stable operation
#        VS
# Grade transition
#
# This can reveal relationships that become stronger
# during a grade change.
# ============================================================


transition_data = df[
    df["transition_phase"]
    == "transition"
]


stable_data = df[
    df["transition_phase"]
    == "stable"
]


transition_correlations = []


for feature in PROCESS_VARIABLES:

    # Transition correlation

    transition_valid = transition_data[
        [
            feature,
            CORRELATION_TARGET
        ]
    ].dropna()


    stable_valid = stable_data[
        [
            feature,
            CORRELATION_TARGET
        ]
    ].dropna()


    if len(transition_valid) > 2:

        transition_corr = (
            transition_valid[
                feature
            ]
            .corr(
                transition_valid[
                    CORRELATION_TARGET
                ]
            )
        )

    else:

        transition_corr = np.nan


    if len(stable_valid) > 2:

        stable_corr = (
            stable_valid[
                feature
            ]
            .corr(
                stable_valid[
                    CORRELATION_TARGET
                ]
            )
        )

    else:

        stable_corr = np.nan


    transition_correlations.append({

        "feature": feature,

        "transition_correlation":
            transition_corr,

        "stable_correlation":
            stable_corr,

        "absolute_change":
            abs(
                transition_corr
                -
                stable_corr
            )
            if (
                pd.notna(transition_corr)
                and
                pd.notna(stable_corr)
            )
            else np.nan

    })


transition_corr_df = pd.DataFrame(
    transition_correlations
)


transition_corr_df = (
    transition_corr_df
    .sort_values(
        "absolute_change",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("TRANSITION VS STABLE CORRELATIONS")
print("=" * 80)

print(
    transition_corr_df.to_string(
        index=False
    )
)


transition_corr_df.to_csv(
    RESULTS_PATH
    / "transition_vs_stable_correlations.csv",
    index=False
)


# ============================================================
# 31. FULL NUMERIC CORRELATION MATRIX
# ============================================================

correlation_columns = (
    PROCESS_VARIABLES
    + [
        "basis_weight",
        "basis_weight_setpoint",
        "future_basis_weight_5min",
        "deviation_pct",
        "stabilization_time_min"
    ]
)


correlation_matrix = (
    df[
        correlation_columns
    ]
    .corr()
)


correlation_matrix.to_csv(
    RESULTS_PATH
    / "full_correlation_matrix.csv"
)


# ============================================================
# 32. CREATE HEATMAP
# ============================================================

plt.figure(
    figsize=(14, 10)
)


sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)


plt.title(
    "Paper Process Variable Correlation Matrix"
)


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 33. BASIS WEIGHT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sns.histplot(
    df["basis_weight"],
    bins=40,
    kde=True
)


plt.axvline(
    df["basis_weight_setpoint"].mean(),
    linestyle="--",
    label="Average Setpoint"
)


plt.title(
    "Basis Weight Distribution"
)


plt.xlabel(
    "Basis Weight (GSM)"
)


plt.ylabel(
    "Frequency"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "basis_weight_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 34. DEVIATION DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sns.histplot(
    df["deviation_pct"],
    bins=40,
    kde=True
)


plt.axvline(
    2.5,
    linestyle="--",
    label="2.5% Off-Spec Threshold"
)


plt.title(
    "Basis Weight Deviation Distribution"
)


plt.xlabel(
    "Deviation (%)"
)


plt.ylabel(
    "Frequency"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "deviation_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 35. OFF-SPEC COUNT PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


sns.countplot(
    data=df,
    x="off_spec"
)


plt.title(
    "Basis Weight Off-Spec Distribution"
)


plt.xlabel(
    "Off-Spec Status (0 = Safe, 1 = Risk)"
)


plt.ylabel(
    "Number of Records"
)


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "off_spec_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 36. FUTURE BASIS WEIGHT VS SETPOINT
# ============================================================

sample_df = df.sample(
    min(3000, len(df)),
    random_state=42
)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    sample_df[
        "future_basis_weight_5min"
    ].values,
    label="Future Basis Weight"
)


plt.plot(
    sample_df[
        "basis_weight_setpoint"
    ].values,
    linestyle="--",
    label="Setpoint"
)


plt.title(
    "Future Basis Weight vs Setpoint"
)


plt.xlabel(
    "Sample"
)


plt.ylabel(
    "Basis Weight (GSM)"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "future_basis_weight_vs_setpoint.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 37. STABILIZATION TIME BY OUTCOME
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sns.boxplot(
    data=df,
    x="transition_outcome",
    y="stabilization_time_min"
)


plt.title(
    "Stabilization Time by Transition Outcome"
)


plt.xlabel(
    "Transition Outcome"
)


plt.ylabel(
    "Stabilization Time (minutes)"
)


plt.tight_layout()


plt.savefig(
    FIGURES_PATH
    / "stabilization_by_outcome.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 38. SAVE CLEANED DATA
# ============================================================
#
# IMPORTANT:
# We do NOT create the incorrect:
#
# basis_weight_deviation_percent
#
# column.
#
# We preserve the dataset's correct:
#
# deviation_pct
# ============================================================


cleaned_output = (
    PROCESSED_PATH
    / "cleaned_data.csv"
)


df.to_csv(
    cleaned_output,
    index=False
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 80)


print(
    "\nDataset:"
)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


print(
    "\nCorrect Basis Weight column:"
)

print(
    "basis_weight"
)


print(
    "\nCorrect Setpoint column:"
)

print(
    "basis_weight_setpoint"
)


print(
    "\nCorrect Deviation column:"
)

print(
    "deviation_pct"
)


print(
    "\nCorrect Off-Spec column:"
)

print(
    "off_spec"
)


print(
    "\nFuture prediction target:"
)

print(
    "future_basis_weight_5min"
)


print(
    "\nFuture classification target:"
)

print(
    "future_off_spec"
)


print(
    "\nCleaned dataset saved to:"
)

print(
    cleaned_output
)


print(
    "\nCorrelation results saved to:"
)

print(
    RESULTS_PATH
)


print(
    "\nEDA charts saved to:"
)

print(
    FIGURES_PATH
)


print("\n" + "=" * 80)
print("READY FOR STAGE 2 - FEATURE ENGINEERING")
print("=" * 80)
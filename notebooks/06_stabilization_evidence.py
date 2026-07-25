"""
06_stabilization_evidence.py

STAGE 6
Stabilization Time + Historical Evidence

Paper Grade Transition Copilot
Honeywell Project

Purpose:
1. Load cleaned process data.
2. Analyze the current grade transition.
3. Calculate historical success/failure statistics.
4. Estimate stabilization time.
5. Load Stage 5 recommendation results.
6. Determine whether a safe recommendation exists.
7. Generate historical evidence.
8. Save evidence as CSV and JSON.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

RECOMMENDATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
    / "recommendation_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
)

EVIDENCE_CSV_PATH = (
    OUTPUT_DIR
    / "stabilization_evidence.csv"
)

EVIDENCE_JSON_PATH = (
    OUTPUT_DIR
    / "stabilization_evidence.json"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

SPEC_LIMIT = 2.5

# Current process record.
# We use the first record exactly as the Stage 5 recommendation
# engine did.
CURRENT_ROW_INDEX = 0


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def make_json_safe(value):
    """
    Convert pandas/numpy values into normal Python values
    that json.dump() can serialize.

    This fixes errors such as:

    TypeError:
    Object of type int64 is not JSON serializable
    """

    # Dictionary
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    # List / tuple
    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Pandas Series
    if isinstance(value, pd.Series):
        return {
            str(key): make_json_safe(val)
            for key, val in value.to_dict().items()
        }

    # Pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return [
            make_json_safe(record)
            for record in value.to_dict(orient="records")
        ]

    # NumPy integer
    if isinstance(value, np.integer):
        return int(value)

    # NumPy floating point
    if isinstance(value, np.floating):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    # NumPy boolean
    if isinstance(value, np.bool_):
        return bool(value)

    # Pandas NA
    if pd.isna(value):
        return None

    # Normal Python value
    return value


# ============================================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. HEADER
# ============================================================

print()
print("=" * 80)
print("STAGE 6 - STABILIZATION + HISTORICAL EVIDENCE")
print("=" * 80)

print("Project root:")
print(PROJECT_ROOT)


# ============================================================
# 6. LOAD CLEANED DATA
# ============================================================

print()
print("Loading cleaned dataset:")
print(CLEANED_DATA_PATH)

if not CLEANED_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{CLEANED_DATA_PATH}"
    )

df = pd.read_csv(CLEANED_DATA_PATH)

print()
print("Dataset loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 7. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "transition_id",
    "grade_from",
    "grade_to",
    "transition_outcome",
    "basis_weight",
    "basis_weight_setpoint",
    "deviation_pct",
    "off_spec",
    "stabilization_time_min",
    "machine_speed",
    "stock_flow",
    "steam_pressure",
    "operator_action",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print()
    print("ERROR: Missing required columns:")

    for column in missing_columns:
        print(f"- {column}")

    raise ValueError(
        "Stage 6 cannot continue because required columns are missing."
    )

print()
print("All required Stage 6 columns found.")


# ============================================================
# 8. ENSURE NUMERIC COLUMNS ARE NUMERIC
# ============================================================

numeric_columns = [
    "basis_weight",
    "basis_weight_setpoint",
    "deviation_pct",
    "off_spec",
    "stabilization_time_min",
    "machine_speed",
    "stock_flow",
    "steam_pressure",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows that cannot be analyzed
df = df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# 9. CURRENT PROCESS
# ============================================================

current = df.iloc[CURRENT_ROW_INDEX]

current_transition_id = str(
    current["transition_id"]
)

current_grade_from = str(
    current["grade_from"]
)

current_grade_to = str(
    current["grade_to"]
)

current_machine_speed = float(
    current["machine_speed"]
)

current_stock_flow = float(
    current["stock_flow"]
)

current_steam_pressure = float(
    current["steam_pressure"]
)

current_basis_weight = float(
    current["basis_weight"]
)

current_setpoint = float(
    current["basis_weight_setpoint"]
)

current_deviation = float(
    current["deviation_pct"]
)

current_status = (
    "SAFE"
    if current_deviation <= SPEC_LIMIT
    else "OFF-SPEC / HIGH RISK"
)


print()
print("=" * 80)
print("CURRENT PROCESS")
print("=" * 80)

print(
    f"Transition: {current_transition_id}"
)

print(
    f"Grade: {current_grade_from} → {current_grade_to}"
)

print(
    f"Machine Speed: {current_machine_speed:.3f}"
)

print(
    f"Stock Flow: {current_stock_flow:.3f}"
)

print(
    f"Steam Pressure: {current_steam_pressure:.3f}"
)

print(
    f"Current Basis Weight: {current_basis_weight:.3f}"
)

print(
    f"Setpoint: {current_setpoint:.3f}"
)

print(
    f"Current Deviation: {current_deviation:.3f}%"
)

print(
    f"Current Status: {current_status}"
)


# ============================================================
# 10. HISTORICAL TRANSITION ANALYSIS
# ============================================================

print()
print("=" * 80)
print("HISTORICAL TRANSITION ANALYSIS")
print("=" * 80)

historical_transition_count = int(
    df["transition_id"].nunique()
)

print(
    f"Historical transition records found: "
    f"{historical_transition_count}"
)


# ============================================================
# 11. MATCH CURRENT GRADE TRANSITION
# ============================================================

matching_transition = df[
    (df["grade_from"] == current_grade_from)
    &
    (df["grade_to"] == current_grade_to)
].copy()

print()
print("=" * 80)
print("GRADE TRANSITION EVIDENCE")
print("=" * 80)

print(
    f"Grade transition: "
    f"{current_grade_from} → {current_grade_to}"
)

print(
    f"Historical observations: "
    f"{len(matching_transition)}"
)


# ============================================================
# 12. HISTORICAL SUCCESS / FAILURE
# ============================================================

if len(matching_transition) > 0:

    success_mask = (
        matching_transition["transition_outcome"]
        .astype(str)
        .str.lower()
        .eq("success")
    )

    failure_mask = (
        matching_transition["transition_outcome"]
        .astype(str)
        .str.lower()
        .eq("failure")
    )

    success_rate = (
        success_mask.mean() * 100
    )

    failure_rate = (
        failure_mask.mean() * 100
    )

    historical_off_spec_rate = (
        matching_transition["off_spec"].mean()
        * 100
    )

    average_stabilization = (
        matching_transition[
            "stabilization_time_min"
        ].mean()
    )

    median_stabilization = (
        matching_transition[
            "stabilization_time_min"
        ].median()
    )

    minimum_stabilization = (
        matching_transition[
            "stabilization_time_min"
        ].min()
    )

    maximum_stabilization = (
        matching_transition[
            "stabilization_time_min"
        ].max()
    )

else:

    success_rate = 0.0
    failure_rate = 0.0
    historical_off_spec_rate = 0.0

    average_stabilization = float(
        df["stabilization_time_min"].mean()
    )

    median_stabilization = float(
        df["stabilization_time_min"].median()
    )

    minimum_stabilization = float(
        df["stabilization_time_min"].min()
    )

    maximum_stabilization = float(
        df["stabilization_time_min"].max()
    )


print(
    f"Historical success rate: "
    f"{success_rate:.2f}%"
)

print(
    f"Historical failure rate: "
    f"{failure_rate:.2f}%"
)

print(
    f"Historical off-spec rate: "
    f"{historical_off_spec_rate:.2f}%"
)

print(
    f"Average stabilization: "
    f"{average_stabilization:.2f} min"
)

print(
    f"Median stabilization: "
    f"{median_stabilization:.2f} min"
)

print(
    f"Minimum stabilization: "
    f"{minimum_stabilization:.2f} min"
)

print(
    f"Maximum stabilization: "
    f"{maximum_stabilization:.2f} min"
)


# ============================================================
# 13. SIMILAR HISTORICAL TRANSITIONS
# ============================================================

print()
print("=" * 80)
print("SIMILAR HISTORICAL TRANSITIONS")
print("=" * 80)


transition_summary = (
    df.groupby(
        [
            "transition_id",
            "grade_from",
            "grade_to",
            "transition_outcome",
        ]
    )
    .agg(
        average_stabilization_time_min=(
            "stabilization_time_min",
            "mean",
        ),
        median_stabilization_time_min=(
            "stabilization_time_min",
            "median",
        ),
        average_deviation_pct=(
            "deviation_pct",
            "mean",
        ),
        off_spec_rate=(
            "off_spec",
            "mean",
        ),
        operator_action_rate=(
            "operator_action",
            lambda x: (
                x.astype(str)
                .str.lower()
                .ne("none")
                .mean()
            ),
        ),
        rows=(
            "transition_id",
            "size",
        ),
    )
    .reset_index()
)


# Only same grade-to-grade transitions
similar_transitions = transition_summary[
    (transition_summary["grade_from"] == current_grade_from)
    &
    (transition_summary["grade_to"] == current_grade_to)
].copy()


# Rank by stabilization time
similar_transitions = (
    similar_transitions
    .sort_values(
        by=[
            "average_stabilization_time_min",
            "average_deviation_pct",
        ]
    )
)


if len(similar_transitions) > 0:

    display_columns = [
        "transition_id",
        "grade_from",
        "grade_to",
        "transition_outcome",
        "average_stabilization_time_min",
        "median_stabilization_time_min",
        "average_deviation_pct",
        "off_spec_rate",
        "operator_action_rate",
        "rows",
    ]

    print(
        similar_transitions[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No matching historical grade transitions found."
    )


# ============================================================
# 14. LOAD STAGE 5 RECOMMENDATION RESULTS
# ============================================================

print()
print("=" * 80)
print("STAGE 5 RECOMMENDATION EVIDENCE")
print("=" * 80)

if RECOMMENDATION_PATH.exists():

    print(
        "Loaded candidate recommendations:"
    )

    print(
        RECOMMENDATION_PATH
    )

    candidates = pd.read_csv(
        RECOMMENDATION_PATH
    )

    print(
        f"Candidate rows: {len(candidates)}"
    )

else:

    print(
        "Stage 5 candidate file was not found."
    )

    candidates = pd.DataFrame()


# ============================================================
# 15. SAFE CANDIDATES
# ============================================================

if not candidates.empty:

    if "predicted_deviation_pct" in candidates.columns:

        safe_candidates = candidates[
            candidates[
                "predicted_deviation_pct"
            ]
            <= SPEC_LIMIT
        ].copy()

    else:

        safe_candidates = pd.DataFrame()

else:

    safe_candidates = pd.DataFrame()


print()
print(
    f"Safe candidates from Stage 5: "
    f"{len(safe_candidates)}"
)


# ============================================================
# 16. BEST CANDIDATE
# ============================================================

best_candidate = None

if not candidates.empty:

    if "predicted_deviation_pct" in candidates.columns:

        candidates_sorted = (
            candidates
            .sort_values(
                by=[
                    "predicted_deviation_pct",
                ]
            )
        )

        best_candidate = (
            candidates_sorted
            .iloc[0]
            .to_dict()
        )


# ============================================================
# 17. STABILIZATION TIME ESTIMATION
# ============================================================

print()
print("=" * 80)
print("STABILIZATION TIME ESTIMATION")
print("=" * 80)


if len(matching_transition) > 0:

    estimated_stabilization = float(
        matching_transition[
            "stabilization_time_min"
        ].mean()
    )

    estimation_source = (
        "historical average for matching "
        "grade transition"
    )

    historical_evidence_strength = "HIGH"

elif len(similar_transitions) > 0:

    estimated_stabilization = float(
        similar_transitions[
            "average_stabilization_time_min"
        ].mean()
    )

    estimation_source = (
        "historical average from similar "
        "grade transitions"
    )

    historical_evidence_strength = "MEDIUM"

else:

    estimated_stabilization = float(
        df["stabilization_time_min"].mean()
    )

    estimation_source = (
        "overall historical process average"
    )

    historical_evidence_strength = "LOW"


# Stabilization risk classification
if estimated_stabilization <= 35:

    stabilization_risk = "LOW"

elif estimated_stabilization <= 55:

    stabilization_risk = "MEDIUM"

else:

    stabilization_risk = "HIGH"


print(
    f"Estimated stabilization time: "
    f"{estimated_stabilization:.2f} minutes"
)

print(
    f"Estimation source: "
    f"{estimation_source}"
)

print(
    f"Stabilization risk: "
    f"{stabilization_risk}"
)

print(
    f"Historical evidence strength: "
    f"{historical_evidence_strength}"
)


# ============================================================
# 18. COPILOT DECISION
# ============================================================

print()
print("=" * 80)
print("STAGE 6 COPILOT DECISION")
print("=" * 80)


if len(safe_candidates) > 0:

    # Best safe candidate:
    safe_ranked = (
        safe_candidates
        .sort_values(
            by=[
                "predicted_deviation_pct",
                "change_penalty",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )

    recommended_candidate = (
        safe_ranked
        .iloc[0]
    )

    recommendation_status = (
        "SAFE RECOMMENDATION FOUND"
    )

    print(
        "SAFE RECOMMENDATION FOUND"
    )

    print()
    print(
        f"Machine Speed: "
        f"{float(recommended_candidate['machine_speed']):.3f}"
    )

    print(
        f"Stock Flow: "
        f"{float(recommended_candidate['stock_flow']):.3f}"
    )

    print(
        f"Steam Pressure: "
        f"{float(recommended_candidate['steam_pressure']):.3f}"
    )

    print(
        f"Predicted Basis Weight: "
        f"{float(recommended_candidate['predicted_basis_weight']):.3f} GSM"
    )

    print(
        f"Predicted deviation: "
        f"{float(recommended_candidate['predicted_deviation_pct']):.3f}%"
    )

    print(
        f"Estimated stabilization: "
        f"{estimated_stabilization:.2f} minutes"
    )

    recommended_action = (
        "Apply recommended process settings "
        "after operator review."
    )

else:

    recommendation_status = (
        "NO SAFE RECOMMENDATION FOUND"
    )

    print(
        "NO SAFE RECOMMENDATION FOUND"
    )

    if best_candidate is not None:

        best_predicted_bw = float(
            best_candidate[
                "predicted_basis_weight"
            ]
        )

        best_predicted_deviation = float(
            best_candidate[
                "predicted_deviation_pct"
            ]
        )

        print()
        print(
            "The tested controllable variables "
            "cannot bring the predicted process "
            "inside the 2.5% specification limit."
        )

        print()
        print(
            f"Best achievable predicted Basis Weight: "
            f"{best_predicted_bw:.3f} GSM"
        )

        print(
            f"Best achievable deviation: "
            f"{best_predicted_deviation:.3f}%"
        )

    recommended_action = (
        "Operator review / recipe-level or "
        "additional process-variable adjustment."
    )


# ============================================================
# 19. AI REASONING
# ============================================================

print()
print("=" * 80)
print("AI REASONING")
print("=" * 80)

reasoning = [
    (
        "Current process is outside the "
        "2.5% Basis Weight specification."
        if current_deviation > SPEC_LIMIT
        else
        "Current process is inside the "
        "2.5% Basis Weight specification."
    ),
    (
        "Current Basis Weight deviation is "
        "above the required safety threshold."
        if current_deviation > SPEC_LIMIT
        else
        "Current Basis Weight deviation is "
        "within the required safety threshold."
    ),
]

if len(safe_candidates) == 0:

    reasoning.append(
        "No tested combination of controllable "
        "Machine Speed, Stock Flow and Steam "
        "Pressure was predicted to satisfy "
        "the specification."
    )

    reasoning.append(
        "The system should therefore escalate "
        "the decision instead of recommending "
        "an unsafe setting."
    )

else:

    reasoning.append(
        "At least one tested combination of "
        "controllable variables satisfies "
        "the 2.5% specification."
    )

    reasoning.append(
        "The safest candidate should be ranked "
        "using predicted deviation and process "
        "change penalty."
    )


reasoning.append(
    f"Historical {current_grade_from} → "
    f"{current_grade_to} transitions show a "
    f"{success_rate:.1f}% success rate."
)

reasoning.append(
    f"Historical average stabilization time "
    f"is {estimated_stabilization:.1f} minutes."
)


for index, statement in enumerate(
    reasoning,
    start=1,
):

    print(
        f"{index}. {statement}"
    )


# ============================================================
# 20. BUILD EVIDENCE DATAFRAME
# ============================================================

evidence_records = []

for _, row in similar_transitions.iterrows():

    evidence_records.append(
        {
            "transition_id": row[
                "transition_id"
            ],
            "grade_from": row[
                "grade_from"
            ],
            "grade_to": row[
                "grade_to"
            ],
            "transition_outcome": row[
                "transition_outcome"
            ],
            "average_stabilization_time_min":
                float(
                    row[
                        "average_stabilization_time_min"
                    ]
                ),
            "median_stabilization_time_min":
                float(
                    row[
                        "median_stabilization_time_min"
                    ]
                ),
            "average_deviation_pct":
                float(
                    row[
                        "average_deviation_pct"
                    ]
                ),
            "off_spec_rate":
                float(
                    row[
                        "off_spec_rate"
                    ]
                ),
            "operator_action_rate":
                float(
                    row[
                        "operator_action_rate"
                    ]
                ),
            "rows":
                int(
                    row["rows"]
                ),
        }
    )


evidence_df = pd.DataFrame(
    evidence_records
)


# ============================================================
# 21. SAVE EVIDENCE CSV
# ============================================================

print()
print("=" * 80)
print("SAVED HISTORICAL EVIDENCE")
print("=" * 80)

evidence_df.to_csv(
    EVIDENCE_CSV_PATH,
    index=False
)

print(
    "Evidence CSV saved to:"
)

print(
    EVIDENCE_CSV_PATH
)


# ============================================================
# 22. CREATE JSON SUMMARY
# ============================================================

json_summary = {
    "project": "Paper Grade Transition Copilot",
    "stage": 6,

    "current_process": {
        "transition_id":
            current_transition_id,

        "grade_from":
            current_grade_from,

        "grade_to":
            current_grade_to,

        "machine_speed":
            current_machine_speed,

        "stock_flow":
            current_stock_flow,

        "steam_pressure":
            current_steam_pressure,

        "basis_weight":
            current_basis_weight,

        "basis_weight_setpoint":
            current_setpoint,

        "deviation_pct":
            current_deviation,

        "status":
            current_status,
    },

    "specification": {
        "maximum_allowed_deviation_pct":
            SPEC_LIMIT,
    },

    "historical_evidence": {
        "historical_transition_records":
            historical_transition_count,

        "matching_grade_transition_observations":
            len(matching_transition),

        "success_rate_pct":
            success_rate,

        "failure_rate_pct":
            failure_rate,

        "off_spec_rate_pct":
            historical_off_spec_rate,

        "average_stabilization_time_min":
            average_stabilization,

        "median_stabilization_time_min":
            median_stabilization,

        "minimum_stabilization_time_min":
            minimum_stabilization,

        "maximum_stabilization_time_min":
            maximum_stabilization,
    },

    "stage_5_recommendation": {
        "candidate_count":
            len(candidates),

        "safe_candidate_count":
            len(safe_candidates),

        "safe_threshold_pct":
            SPEC_LIMIT,
    },

    "stabilization": {
        "estimated_stabilization_time_min":
            estimated_stabilization,

        "estimation_source":
            estimation_source,

        "stabilization_risk":
            stabilization_risk,

        "historical_evidence_strength":
            historical_evidence_strength,
    },

    "copilot_decision": {
        "recommendation_status":
            recommendation_status,

        "recommended_action":
            recommended_action,

        "reasoning":
            reasoning,
    },
}


# Add best candidate if available
if best_candidate is not None:

    json_summary[
        "stage_5_recommendation"
    ][
        "best_candidate"
    ] = best_candidate


# ============================================================
# 23. MAKE JSON COMPLETELY SAFE
# ============================================================

json_summary = make_json_safe(
    json_summary
)


# ============================================================
# 24. SAVE JSON
# ============================================================

with open(
    EVIDENCE_JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        json_summary,
        file,
        indent=4,
        ensure_ascii=False,
    )


print()
print(
    "Evidence JSON saved to:"
)

print(
    EVIDENCE_JSON_PATH
)


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("STAGE 6 COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print("Outputs:")

print(
    f"1. Historical evidence CSV:"
)

print(
    EVIDENCE_CSV_PATH
)

print(
    f"2. Historical evidence JSON:"
)

print(
    EVIDENCE_JSON_PATH
)

print()
print(
    "Stage 6 is ready for Stage 7 - Flask/API integration."
)

print("=" * 80)
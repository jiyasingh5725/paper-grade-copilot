from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "basis_weight_xgboost.pkl"
)


# ============================================================
# PROCESS LIMITS
# ============================================================

PROCESS_LIMITS = {
    "machine_speed": {
        "min": 700,
        "max": 900,
    },

    "stock_flow": {
        "min": 50,
        "max": 80,
    },

    "filler_flow": {
        "min": 10,
        "max": 30,
    },

    "steam_pressure": {
        "min": 4.0,
        "max": 6.0,
    },

    "moisture": {
        "min": 5.5,
        "max": 7.5,
    },

    "ash": {
        "min": 6.0,
        "max": 12.0,
    },

    "caliper": {
        "min": 80,
        "max": 140,
    },
}


# ============================================================
# LOAD MODEL
# ============================================================

PIPELINE = joblib.load(
    MODEL_PATH
)


# ============================================================
# FEATURE DEFINITIONS
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
    + CATEGORICAL_FEATURES
)


# ============================================================
# HELPER: PROCESS LIMIT CHECK
# ============================================================

def is_within_limits(row):
    """
    Check whether every controllable process variable
    remains inside its allowed operating range.
    """

    variables = [
        "machine_speed",
        "stock_flow",
        "filler_flow",
        "steam_pressure",
        "moisture",
        "ash",
        "caliper",
    ]

    for variable in variables:

        value = float(
            row[variable]
        )

        minimum = PROCESS_LIMITS[
            variable
        ]["min"]

        maximum = PROCESS_LIMITS[
            variable
        ]["max"]

        if value < minimum or value > maximum:
            return False

    return True


# ============================================================
# HELPER: DERIVED FEATURES
# ============================================================

def create_derived_features(
    row,
    reference_row
):
    """
    Recalculate features that depend on changed
    process variables.
    """

    row = row.copy()

    # --------------------------------------------------------
    # Position within operating range
    # --------------------------------------------------------

    process_variables = [
        "machine_speed",
        "stock_flow",
        "filler_flow",
        "steam_pressure",
        "moisture",
        "ash",
        "caliper",
    ]

    for variable in process_variables:

        minimum = PROCESS_LIMITS[
            variable
        ]["min"]

        maximum = PROCESS_LIMITS[
            variable
        ]["max"]

        value = float(
            row[variable]
        )

        row[
            f"{variable}_position"
        ] = (
            value - minimum
        ) / (
            maximum - minimum
        )

        row[
            f"{variable}_position_from_min"
        ] = (
            value - minimum
        ) / (
            maximum - minimum
        )

        row[
            f"{variable}_position_from_max"
        ] = (
            maximum - value
        ) / (
            maximum - minimum
        )


    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    row["speed_stock_ratio"] = (
        row["machine_speed"]
        /
        max(row["stock_flow"], 1e-6)
    )

    row["steam_moisture_ratio"] = (
        row["steam_pressure"]
        /
        max(row["moisture"], 1e-6)
    )

    row["filler_stock_ratio"] = (
        row["filler_flow"]
        /
        max(row["stock_flow"], 1e-6)
    )


    # --------------------------------------------------------
    # Basis Weight error
    # --------------------------------------------------------

    row["basis_weight_error"] = (
        row["basis_weight"]
        -
        row["basis_weight_setpoint"]
    )

    if row["basis_weight_error"] > 0:
        row["basis_weight_error_direction"] = 1

    elif row["basis_weight_error"] < 0:
        row["basis_weight_error_direction"] = -1

    else:
        row["basis_weight_error_direction"] = 0


    # --------------------------------------------------------
    # Process changes
    # --------------------------------------------------------

    row["speed_change_5min"] = (
        row["machine_speed"]
        -
        reference_row["machine_speed"]
    )

    row["stock_flow_change_5min"] = (
        row["stock_flow"]
        -
        reference_row["stock_flow"]
    )

    return row


# ============================================================
# PREDICT FUTURE BASIS WEIGHT
# ============================================================

def predict_basis_weight(row):
    """
    Predict future Basis Weight using the trained XGBoost model.
    """

    input_df = pd.DataFrame(
        [row]
    )

    input_df = input_df[
        FEATURES
    ]

    prediction = PIPELINE.predict(
        input_df
    )[0]

    return float(
        prediction
    )


# ============================================================
# DEVIATION CALCULATION
# ============================================================

def calculate_deviation(
    prediction,
    setpoint
):
    """
    Calculate percentage deviation from Basis Weight setpoint.
    """

    return (
        abs(
            prediction
            -
            setpoint
        )
        /
        setpoint
        *
        100
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    current_row
):
    """
    Generate feasible candidate operating conditions.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDATION DEBUG")
    print("=" * 70)

    print("Current machine speed:",
          current_row["machine_speed"])

    print("Current stock flow:",
          current_row["stock_flow"])

    print("Current steam pressure:",
          current_row["steam_pressure"])

    print("Generated speed values:",
          speed_values)

    print("Generated stock values:",
          stock_values)

    print("Generated steam values:",
          steam_values)

    print("=" * 70)

    candidates = []

    current_speed = float(
        current_row[
            "machine_speed"
        ]
    )

    current_stock = float(
        current_row[
            "stock_flow"
        ]
    )

    current_steam = float(
        current_row[
            "steam_pressure"
        ]
    )


    speed_values = np.arange(
        max(
            700,
            current_speed - 30
        ),
        min(
            900,
            current_speed + 30
        ) + 1,
        10
    )


    stock_values = np.arange(
        max(
            50,
            current_stock - 6
        ),
        min(
            80,
            current_stock + 6
        ) + 1,
        2
    )


    steam_values = np.arange(
        max(
            4.0,
            current_steam - 0.4
        ),
        min(
            6.0,
            current_steam + 0.4
        ) + 0.001,
        0.1
    )


    for speed in speed_values:

        for stock in stock_values:

            for steam in steam_values:

                candidate = current_row.copy()

                candidate[
                    "machine_speed"
                ] = round(
                    float(speed),
                    3
                )

                candidate[
                    "stock_flow"
                ] = round(
                    float(stock),
                    3
                )

                candidate[
                    "steam_pressure"
                ] = round(
                    float(steam),
                    3
                )


                candidate = (
                    create_derived_features(
                        candidate,
                        current_row
                    )
                )


                if is_within_limits(
                    candidate
                ):

                    candidates.append(
                        candidate
                    )

    return candidates


# ============================================================
# RECOMMEND BEST SETTINGS
# ============================================================

def recommend(
    current_row
):
    """
    Find the safest operating conditions
    that minimize predicted Basis Weight deviation.
    """

    setpoint = float(
        current_row[
            "basis_weight_setpoint"
        ]
    )


    current_prediction = (
        predict_basis_weight(
            current_row
        )
    )


    current_deviation = (
        calculate_deviation(
            current_prediction,
            setpoint
        )
    )


    candidates = generate_candidates(
        current_row
    )


    results = []


    for candidate in candidates:

        prediction = (
            predict_basis_weight(
                candidate
            )
        )


        deviation = (
            calculate_deviation(
                prediction,
                setpoint
            )
        )


        results.append({

            "machine_speed":
                candidate[
                    "machine_speed"
                ],

            "stock_flow":
                candidate[
                    "stock_flow"
                ],

            "steam_pressure":
                candidate[
                    "steam_pressure"
                ],

            "predicted_basis_weight":
                prediction,

            "predicted_deviation_pct":
                deviation,

            "off_spec":
                int(
                    deviation > 2.5
                ),

            "speed_change":
                candidate[
                    "machine_speed"
                ]
                -
                current_row[
                    "machine_speed"
                ],

            "stock_change":
                candidate[
                    "stock_flow"
                ]
                -
                current_row[
                    "stock_flow"
                ],

            "steam_change":
                candidate[
                    "steam_pressure"
                ]
                -
                current_row[
                    "steam_pressure"
                ],
        })


    result_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Rank by safety first.
    # --------------------------------------------------------

    result_df[
        "safe_priority"
    ] = (
        result_df[
            "off_spec"
        ]
    )


    # Prefer smaller changes when predictions
    # are similarly safe.

    result_df[
        "change_penalty"
    ] = (
        abs(
            result_df[
                "speed_change"
            ]
        ) / 30
        +
        abs(
            result_df[
                "stock_change"
            ]
        ) / 6
        +
        abs(
            result_df[
                "steam_change"
            ]
        ) / 0.4
    )


    result_df = (
        result_df
        .sort_values(
            [
                "safe_priority",
                "predicted_deviation_pct",
                "change_penalty",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best = result_df.iloc[
        0
    ]


    return {
        "current_prediction":
            current_prediction,

        "current_deviation_pct":
            current_deviation,

        "recommended_machine_speed":
            best[
                "machine_speed"
            ],

        "recommended_stock_flow":
            best[
                "stock_flow"
            ],

        "recommended_steam_pressure":
            best[
                "steam_pressure"
            ],

        "predicted_basis_weight":
            best[
                "predicted_basis_weight"
            ],

        "predicted_deviation_pct":
            best[
                "predicted_deviation_pct"
            ],

        "off_spec":
            int(
                best[
                    "off_spec"
                ]
            ),

        "speed_change":
            best[
                "speed_change"
            ],

        "stock_change":
            best[
                "stock_change"
            ],

        "steam_change":
            best[
                "steam_change"
            ],

        "candidate_count":
            len(
                result_df
            ),

        "all_candidates":
            result_df,
    }
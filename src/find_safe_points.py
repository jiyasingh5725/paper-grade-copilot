# ============================================================
# PAPER GRADE TRANSITION COPILOT
# FIND SAFE OPERATING POINTS
#
# This script:
#   1. Reuses prepare_model_features() from app.py
#   2. Reuses the existing XGBoost model
#   3. Uses the current process row from app.py
#   4. Searches the complete allowed operating range
#   5. Finds combinations within ±2.5% Basis Weight deviation
#   6. Prints the top 20 safest combinations
#   7. Saves results to CSV
#
# FILE LOCATION:
#
# paper-grade-copilot/
# └── src/
#     ├── app.py
#     ├── recommendation.py
#     └── find_safe_points.py   <-- THIS FILE
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Current folder:
# paper-grade-copilot/src/

SRC_DIR = Path(
    __file__
).resolve().parent


# Project root:
# paper-grade-copilot/

PROJECT_ROOT = SRC_DIR.parent


# Make sure Python can import app.py
sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# IMPORT EXISTING APP COMPONENTS
# ============================================================

try:

    from app import (
        prepare_model_features,
        df,
        basis_weight_model
    )

except Exception as e:

    print("\n" + "=" * 80)

    print(
        "ERROR: Could not import components from app.py"
    )

    print("=" * 80)

    print(
        "\nMake sure that app.py contains:"
    )

    print(
        "1. prepare_model_features()"
    )

    print(
        "2. df"
    )

    print(
        "3. basis_weight_model"
    )

    print(
        "\nOriginal error:"
    )

    print(e)

    sys.exit(1)


# ============================================================
# SAFETY SPECIFICATION
# ============================================================

MAX_ALLOWED_DEVIATION = 2.5


# ============================================================
# PROCESS LIMITS
# ============================================================

PROCESS_LIMITS = {

    "machine_speed": {
        "min": 700.0,
        "max": 900.0
    },

    "stock_flow": {
        "min": 50.0,
        "max": 80.0
    },

    "steam_pressure": {
        "min": 4.0,
        "max": 6.0
    }

}


# ============================================================
# SEARCH RESOLUTION
# ============================================================
#
# Smaller steps = more precise search
#
# Current search:
#
# Machine Speed:
#     700 → 900
#     step = 5
#
# Stock Flow:
#     50 → 80
#     step = 1
#
# Steam Pressure:
#     4 → 6
#     step = 0.1
#
# Total combinations:
#
# 41 × 31 × 21 = 26,691
#
# This is small enough for normal execution.
# ============================================================

MACHINE_SPEED_VALUES = np.arange(
    700.0,
    900.0 + 0.001,
    5.0
)


STOCK_FLOW_VALUES = np.arange(
    50.0,
    80.0 + 0.001,
    1.0
)


STEAM_PRESSURE_VALUES = np.arange(
    4.0,
    6.0 + 0.001,
    0.1
)


# ============================================================
# GET CURRENT PROCESS
# ============================================================

def get_current_process():

    # --------------------------------------------------------
    # Check dataframe
    # --------------------------------------------------------

    if df is None:

        print(
            "\nERROR: app.py dataframe is None."
        )

        sys.exit(1)


    if df.empty:

        print(
            "\nERROR: app.py dataframe is empty."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Use first row
    # --------------------------------------------------------

    current_row = df.iloc[
        0
    ].copy()


    return current_row


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def check_required_columns(
    row
):

    required_columns = [

        "machine_speed",

        "stock_flow",

        "steam_pressure",

        "basis_weight",

        "basis_weight_setpoint"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in row.index

    ]


    if missing_columns:

        print(
            "\nERROR: Required columns are missing:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        sys.exit(1)


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features():

    if basis_weight_model is None:

        print(
            "\nERROR: Basis Weight model is not loaded."
        )

        sys.exit(1)


    if not hasattr(
        basis_weight_model,
        "feature_names_in_"
    ):

        print(
            "\nERROR: XGBoost model does not expose "
            "feature_names_in_."
        )

        sys.exit(1)


    return list(
        basis_weight_model.feature_names_in_
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(
    row,
    model_features
):

    # --------------------------------------------------------
    # Verify all expected features exist
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in model_features

        if feature not in row.index

    ]


    if missing_features:

        raise ValueError(

            "Missing model features: "

            + ", ".join(
                missing_features
            )

        )


    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(

        [

            [
                row[feature]

                for feature in model_features

            ]

        ],

        columns=model_features

    )


    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=[
            "number"
        ]
    ).columns


    for column in numeric_columns:

        X[column] = pd.to_numeric(

            X[column],

            errors="coerce"

        )


    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    X = X.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan

    )


    # --------------------------------------------------------
    # Fill numeric missing values
    # --------------------------------------------------------

    for column in numeric_columns:

        X[column] = X[column].fillna(
            0
        )


    return X


# ============================================================
# PREDICT ONE OPERATING POINT
# ============================================================

def predict_operating_point(
    current_row,
    machine_speed,
    stock_flow,
    steam_pressure,
    model_features
):

    # --------------------------------------------------------
    # Make a copy of the current process
    # --------------------------------------------------------

    candidate = current_row.copy()


    # --------------------------------------------------------
    # Change controllable variables
    # --------------------------------------------------------

    candidate[
        "machine_speed"
    ] = float(
        machine_speed
    )


    candidate[
        "stock_flow"
    ] = float(
        stock_flow
    )


    candidate[
        "steam_pressure"
    ] = float(
        steam_pressure
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Reuse EXACTLY the same feature engineering
    # used by your Flask prediction API.
    # --------------------------------------------------------

    candidate = prepare_model_features(
        candidate
    )


    # --------------------------------------------------------
    # Build model input
    # --------------------------------------------------------

    X = create_model_input(

        candidate,

        model_features

    )


    # --------------------------------------------------------
    # Run XGBoost prediction
    # --------------------------------------------------------

    prediction = basis_weight_model.predict(
        X
    )[0]


    prediction = float(
        prediction
    )


    # --------------------------------------------------------
    # Get target
    # --------------------------------------------------------

    target = float(

        candidate[
            "basis_weight_setpoint"
        ]

    )


    # --------------------------------------------------------
    # Calculate deviation
    # --------------------------------------------------------

    if target == 0:

        deviation = 0.0

    else:

        deviation = (

            abs(
                prediction - target
            )

            /

            abs(
                target
            )

            *

            100.0

        )


    # --------------------------------------------------------
    # Safe / unsafe
    # --------------------------------------------------------

    is_safe = (

        deviation
        <=
        MAX_ALLOWED_DEVIATION

    )


    return {

        "machine_speed":
            float(machine_speed),

        "stock_flow":
            float(stock_flow),

        "steam_pressure":
            float(steam_pressure),

        "predicted_basis_weight":
            prediction,

        "target_basis_weight":
            target,

        "deviation_pct":
            deviation,

        "safe":
            is_safe

    }


# ============================================================
# CALCULATE CHANGE PENALTY
# ============================================================

def calculate_change_penalty(
    result,
    current_row
):

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


    speed_change = abs(

        result[
            "machine_speed"
        ]

        -

        current_speed

    )


    stock_change = abs(

        result[
            "stock_flow"
        ]

        -

        current_stock

    )


    steam_change = abs(

        result[
            "steam_pressure"
        ]

        -

        current_steam

    )


    # --------------------------------------------------------
    # Normalize changes
    #
    # This follows the same idea as recommendation.py
    # --------------------------------------------------------

    penalty = (

        speed_change / 30.0

        +

        stock_change / 6.0

        +

        steam_change / 0.4

    )


    return float(
        penalty
    )


# ============================================================
# SEARCH ALL COMBINATIONS
# ============================================================

def search_all_points(
    current_row,
    model_features
):

    results = []


    total_combinations = (

        len(
            MACHINE_SPEED_VALUES
        )

        *

        len(
            STOCK_FLOW_VALUES
        )

        *

        len(
            STEAM_PRESSURE_VALUES
        )

    )


    print(
        f"\nTotal combinations: "
        f"{total_combinations:,}"
    )


    print(
        "\nSearching operating space..."
    )


    completed = 0


    for machine_speed in MACHINE_SPEED_VALUES:

        for stock_flow in STOCK_FLOW_VALUES:

            for steam_pressure in STEAM_PRESSURE_VALUES:

                result = predict_operating_point(

                    current_row,

                    machine_speed,

                    stock_flow,

                    steam_pressure,

                    model_features

                )


                result[
                    "change_penalty"
                ] = calculate_change_penalty(

                    result,

                    current_row

                )


                results.append(
                    result
                )


                completed += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        percentage = (

            completed
            /
            total_combinations
            *
            100

        )


        print(
            f"Progress: "
            f"{percentage:6.2f}%"
        )


    return pd.DataFrame(
        results
    )


# ============================================================
# PRINT CURRENT PROCESS
# ============================================================

def print_current_process(
    current_row
):

    print("\n")
    print("=" * 90)
    print("CURRENT PROCESS")
    print("=" * 90)


    print(

        f"Machine Speed    : "
        f"{float(current_row['machine_speed']):.3f} m/min"

    )


    print(

        f"Stock Flow       : "
        f"{float(current_row['stock_flow']):.3f} %"

    )


    print(

        f"Steam Pressure   : "
        f"{float(current_row['steam_pressure']):.3f} bar"

    )


    print(

        f"Current BW       : "
        f"{float(current_row['basis_weight']):.3f} GSM"

    )


    print(

        f"Target BW        : "
        f"{float(current_row['basis_weight_setpoint']):.3f} GSM"

    )


# ============================================================
# PRINT TOP SAFE POINTS
# ============================================================

def print_top_safe_points(
    safe_df
):

    print("\n")
    print("=" * 90)
    print("TOP 20 SAFEST OPERATING POINTS")
    print("=" * 90)


    columns = [

        "machine_speed",

        "stock_flow",

        "steam_pressure",

        "predicted_basis_weight",

        "target_basis_weight",

        "deviation_pct",

        "change_penalty"

    ]


    top_20 = safe_df.head(
        20
    )


    if top_20.empty:

        print(
            "\nNo safe operating points found."
        )

        return


    print(

        top_20[
            columns
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.3f}"

        )

    )


# ============================================================
# PRINT BEST SAFE POINT
# ============================================================

def print_best_safe_point(
    safe_df
):

    if safe_df.empty:

        return


    best = safe_df.iloc[
        0
    ]


    print("\n")
    print("=" * 90)
    print("BEST SAFE OPERATING POINT")
    print("=" * 90)


    print(

        f"\nMachine Speed   : "
        f"{best['machine_speed']:.3f} m/min"

    )


    print(

        f"Stock Flow      : "
        f"{best['stock_flow']:.3f} %"

    )


    print(

        f"Steam Pressure  : "
        f"{best['steam_pressure']:.3f} bar"

    )


    print(

        f"Predicted BW    : "
        f"{best['predicted_basis_weight']:.3f} GSM"

    )


    print(

        f"Target BW       : "
        f"{best['target_basis_weight']:.3f} GSM"

    )


    print(

        f"Deviation       : "
        f"{best['deviation_pct']:.3f}%"

    )


    print(

        f"Change Penalty  : "
        f"{best['change_penalty']:.3f}"

    )


    print(
        "\nSTATUS: SAFE"
    )


# ============================================================
# PRINT CLOSEST UNSAFE POINTS
# ============================================================

def print_closest_unsafe_points(
    results_df
):

    closest = (

        results_df

        .sort_values(

            [

                "deviation_pct",

                "change_penalty"

            ]

        )

        .head(
            20
        )

    )


    print("\n")
    print("=" * 90)
    print("NO SAFE POINT FOUND")
    print("=" * 90)


    print(

        "\nThe model did not find a combination "
        "within the 2.5% specification."

    )


    print(

        "\nTOP 20 CLOSEST OPERATING POINTS"

    )


    columns = [

        "machine_speed",

        "stock_flow",

        "steam_pressure",

        "predicted_basis_weight",

        "target_basis_weight",

        "deviation_pct",

        "change_penalty"

    ]


    print(

        closest[
            columns
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.3f}"

        )

    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    safe_df
):

    # --------------------------------------------------------
    # Save all tested combinations
    # --------------------------------------------------------

    all_results_path = (

        PROJECT_ROOT
        /
        "safe_points_search_results.csv"

    )


    results_df.to_csv(

        all_results_path,

        index=False

    )


    # --------------------------------------------------------
    # Save safe points
    # --------------------------------------------------------

    safe_results_path = (

        PROJECT_ROOT
        /
        "safe_points.csv"

    )


    safe_df.to_csv(

        safe_results_path,

        index=False

    )


    print("\n")
    print("=" * 90)
    print("RESULT FILES")
    print("=" * 90)


    print(

        f"\nAll combinations:"
        f"\n{all_results_path}"

    )


    print(

        f"\nSafe combinations:"
        f"\n{safe_results_path}"

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 90)
    print("PAPER GRADE TRANSITION COPILOT")
    print("SAFE OPERATING POINT SEARCH")
    print("=" * 90)


    print(
        "\nSafety specification:"
    )

    print(

        f"Basis Weight deviation "
        f"must be <= {MAX_ALLOWED_DEVIATION}%"

    )


    # ========================================================
    # CURRENT PROCESS
    # ========================================================

    current_row = get_current_process()


    check_required_columns(
        current_row
    )


    print_current_process(
        current_row
    )


    # ========================================================
    # MODEL
    # ========================================================

    model_features = get_model_features()


    print("\n")
    print("=" * 90)
    print("MODEL INFORMATION")
    print("=" * 90)


    print(

        f"\nModel type:"
        f"\n{type(basis_weight_model).__name__}"

    )


    print(

        f"\nExpected model features:"
        f"\n{len(model_features)}"

    )


    # ========================================================
    # SEARCH
    # ========================================================

    results_df = search_all_points(

        current_row,

        model_features

    )


    # ========================================================
    # ADD SAFE PRIORITY
    # ========================================================

    results_df[
        "safe_priority"
    ] = results_df[
        "safe"
    ].apply(

        lambda x:
        0 if x else 1

    )


    # ========================================================
    # SAFE POINTS
    # ========================================================

    safe_df = (

        results_df[
            results_df["safe"] == True
        ]

        .sort_values(

            [

                "deviation_pct",

                "change_penalty"

            ]

        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        results_df
    )


    safe_count = len(
        safe_df
    )


    unsafe_count = (
        total -
        safe_count
    )


    print("\n")
    print("=" * 90)
    print("SEARCH SUMMARY")
    print("=" * 90)


    print(
        f"\nTotal combinations tested : "
        f"{total:,}"
    )


    print(
        f"Safe combinations          : "
        f"{safe_count:,}"
    )


    print(
        f"Unsafe combinations        : "
        f"{unsafe_count:,}"
    )


    if total > 0:

        safe_percentage = (

            safe_count
            /
            total
            *
            100

        )

    else:

        safe_percentage = 0


    print(

        f"Safe percentage            : "
        f"{safe_percentage:.2f}%"

    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if safe_df.empty:

        print_closest_unsafe_points(
            results_df
        )

    else:

        print_top_safe_points(
            safe_df
        )

        print_best_safe_point(
            safe_df
        )


    # ========================================================
    # SAVE
    # ========================================================

    save_results(

        results_df,

        safe_df

    )


    print("\n")
    print("=" * 90)
    print("DONE")
    print("=" * 90)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
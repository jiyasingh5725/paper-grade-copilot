import os
import json
import joblib
import pandas as pd

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "model_results")


BASIS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "basis_weight_xgboost.pkl"
)

OFFSPEC_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "off_spec_xgboost.pkl"
)

DATA_PATH = os.path.join(
    DATA_DIR,
    "cleaned_data.csv"
)

RECOMMENDATION_PATH = os.path.join(
    REPORT_DIR,
    "recommendation_candidates.csv"
)

EVIDENCE_PATH = os.path.join(
    REPORT_DIR,
    "stabilization_evidence.json"
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

CORS(app)


# ============================================================
# FRONTEND
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

basis_weight_model = None
off_spec_model = None
df = None

MODEL_LOAD_ERROR = None


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("PAPER GRADE TRANSITION COPILOT - FLASK API")
print("=" * 70)

print("\nProject root:")
print(BASE_DIR)

print("\nLoading trained models...")


try:

    # --------------------------------------------------------
    # Basis Weight Model
    # --------------------------------------------------------

    if not os.path.exists(BASIS_MODEL_PATH):

        raise FileNotFoundError(
            f"Basis weight model not found:\n{BASIS_MODEL_PATH}"
        )

    basis_weight_model = joblib.load(
        BASIS_MODEL_PATH
    )

    print("Basis Weight model loaded successfully.")


    # --------------------------------------------------------
    # OFF-SPEC Model
    # --------------------------------------------------------

    if not os.path.exists(OFFSPEC_MODEL_PATH):

        raise FileNotFoundError(
            f"OFF-SPEC model not found:\n{OFFSPEC_MODEL_PATH}"
        )

    off_spec_model = joblib.load(
        OFFSPEC_MODEL_PATH
    )

    print("OFF-SPEC model loaded successfully.")

    print("\nModel loading completed successfully.")


except Exception as e:

    MODEL_LOAD_ERROR = str(e)

    print("\nWARNING: Models could not be loaded.")
    print(str(e))


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

try:

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nCleaned dataset loaded.")
    print("Rows    :", len(df))
    print("Columns :", len(df.columns))


except Exception as e:

    print("\nWARNING: Dataset could not be loaded.")
    print(str(e))

    df = pd.DataFrame()


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# TRANSITION PHASE ENCODING
# ============================================================

def encode_transition_phase(value):

    """
    Converts transition phase text into a numerical value.

    Expected logical order:

    Before       -> 0
    Transition   -> 1
    Stabilization-> 2
    Stable       -> 3
    """

    if pd.isna(value):

        return 0

    phase = str(value).strip().lower()

    phase_mapping = {

        "before": 0,
        "pre-transition": 0,
        "pre_transition": 0,

        "transition": 1,

        "stabilization": 2,
        "stabilisation": 2,

        "stable": 3,

        "post-transition": 3,
        "post_transition": 3
    }

    return phase_mapping.get(
        phase,
        0
    )


# ============================================================
# FEATURE ENGINEERING FOR MODEL
# ============================================================

def prepare_model_features(row):
    """
    Recreates the engineered features used while training
    the XGBoost model.

    This is necessary because cleaned_data.csv contains
    the base features, while the trained model expects
    additional engineered features.
    """

    # --------------------------------------------------------
    # Make a copy so original dataframe is not modified
    # --------------------------------------------------------

    row = row.copy()


    # ========================================================
    # BASIC PROCESS VALUES
    # ========================================================

    machine_speed = safe_float(
        row.get("machine_speed")
    )

    stock_flow = safe_float(
        row.get("stock_flow")
    )

    filler_flow = safe_float(
        row.get("filler_flow")
    )

    steam_pressure = safe_float(
        row.get("steam_pressure")
    )

    moisture = safe_float(
        row.get("moisture")
    )

    ash = safe_float(
        row.get("ash")
    )

    caliper = safe_float(
        row.get("caliper")
    )

    basis_weight = safe_float(
        row.get("basis_weight")
    )

    setpoint = safe_float(
        row.get("basis_weight_setpoint")
    )


    # ========================================================
    # POSITION FEATURES
    # ========================================================

    process_variables = [

        "machine_speed",
        "stock_flow",
        "filler_flow",
        "steam_pressure",
        "moisture",
        "ash",
        "caliper"

    ]


    for variable in process_variables:

        value = safe_float(
            row.get(variable)
        )

        min_value = safe_float(
            row.get(
                f"{variable}_min"
            )
        )

        max_value = safe_float(
            row.get(
                f"{variable}_max"
            )
        )


        range_value = max_value - min_value


        # ----------------------------------------------------
        # Position between minimum and maximum
        # ----------------------------------------------------

        if range_value != 0:

            position = (
                value - min_value
            ) / range_value

            position_from_min = position

            position_from_max = (
                max_value - value
            ) / range_value

        else:

            position = 0.0
            position_from_min = 0.0
            position_from_max = 0.0


        row[
            f"{variable}_position"
        ] = position


        row[
            f"{variable}_position_from_min"
        ] = position_from_min


        row[
            f"{variable}_position_from_max"
        ] = position_from_max


    # ========================================================
    # GRADE CHANGE
    # ========================================================

    grade_from = str(
        row.get(
            "grade_from",
            ""
        )
    )

    grade_to = str(
        row.get(
            "grade_to",
            ""
        )
    )


    row["grade_changed"] = int(
        grade_from.strip().lower()
        !=
        grade_to.strip().lower()
    )


    # ========================================================
    # TRANSITION PHASE ENCODING
    # ========================================================

    row["transition_phase_encoded"] = encode_transition_phase(
        row.get(
            "transition_phase",
            ""
        )
    )


    # ========================================================
    # OPERATOR ADJUSTMENT FLAG
    # ========================================================

    operator_action = str(
        row.get(
            "operator_action",
            ""
        )
    ).strip().lower()


    no_action_values = {

        "",
        "none",
        "no_action",
        "no action",
        "nan",
        "null"
    }


    row["operator_adjustment_flag"] = int(
        operator_action not in no_action_values
    )


    # ========================================================
    # RATIO FEATURES
    # ========================================================

    if stock_flow != 0:

        row["speed_stock_ratio"] = (
            machine_speed /
            stock_flow
        )

        row["filler_stock_ratio"] = (
            filler_flow /
            stock_flow
        )

    else:

        row["speed_stock_ratio"] = 0.0
        row["filler_stock_ratio"] = 0.0


    if moisture != 0:

        row["steam_moisture_ratio"] = (
            steam_pressure /
            moisture
        )

    else:

        row["steam_moisture_ratio"] = 0.0


    # ========================================================
    # BASIS WEIGHT ERROR
    # ========================================================

    row["basis_weight_error"] = (
        basis_weight -
        setpoint
    )


    # ========================================================
    # BASIS WEIGHT ERROR DIRECTION
    # ========================================================

    if basis_weight > setpoint:

        row["basis_weight_error_direction"] = 1

    elif basis_weight < setpoint:

        row["basis_weight_error_direction"] = -1

    else:

        row["basis_weight_error_direction"] = 0


    # ========================================================
    # RETURN ENGINEERED ROW
    # ========================================================

    return row


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "running",

        "application":
            "Paper Grade Transition Copilot",

        "models_loaded": (
            basis_weight_model is not None
            and off_spec_model is not None
        ),

        "dataset_loaded":
            not df.empty,

        "model_error":
            MODEL_LOAD_ERROR

    })


# ============================================================
# DATASET SUMMARY API
# ============================================================

@app.route("/api/summary", methods=["GET"])
def summary():

    if df.empty:

        return jsonify({

            "success": False,

            "message":
                "Dataset is not available."

        }), 500


    response = {

        "success": True,

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "transitions":
            int(
                df["transition_id"].nunique()
                if "transition_id" in df.columns
                else 0
            )

    }


    if "future_off_spec" in df.columns:

        response["off_spec_count"] = int(
            df["future_off_spec"].sum()
        )

        response["safe_count"] = int(
            (df["future_off_spec"] == 0).sum()
        )


    return jsonify(response)


# ============================================================
# CURRENT PROCESS API
# ============================================================

@app.route("/api/current-process", methods=["GET"])
def current_process():

    if df.empty:

        return jsonify({

            "success": False,

            "message":
                "Dataset is not available."

        }), 500


    row = df.iloc[0]


    result = {

        "success": True,

        "transition_id":
            str(row.get("transition_id", "")),

        "grade_from":
            str(row.get("grade_from", "")),

        "grade_to":
            str(row.get("grade_to", "")),

        "machine_speed":
            safe_float(row.get("machine_speed")),

        "stock_flow":
            safe_float(row.get("stock_flow")),

        "filler_flow":
            safe_float(row.get("filler_flow")),

        "steam_pressure":
            safe_float(row.get("steam_pressure")),

        "moisture":
            safe_float(row.get("moisture")),

        "ash":
            safe_float(row.get("ash")),

        "caliper":
            safe_float(row.get("caliper")),

        "basis_weight":
            safe_float(row.get("basis_weight")),

        "basis_weight_setpoint":
            safe_float(
                row.get("basis_weight_setpoint")
            ),

        "future_basis_weight":
            safe_float(
                row.get("future_basis_weight_5min")
            ),

        "future_deviation_pct":
            safe_float(
                row.get("future_deviation_pct")
            ),

        "future_off_spec":
            int(
                safe_float(
                    row.get(
                        "future_off_spec",
                        0
                    )
                )
            )

    }


    return jsonify(result)


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    if basis_weight_model is None:

        return jsonify({

            "success": False,

            "message":
                "Basis Weight model is not loaded.",

            "error":
                MODEL_LOAD_ERROR

        }), 500


    if df.empty:

        return jsonify({

            "success": False,

            "message":
                "Dataset is not loaded."

        }), 500


    try:

        # ====================================================
        # GET FRONTEND DATA
        # ====================================================

        data = request.get_json(
            silent=True
        ) or {}


        print("\nReceived prediction request:")
        print(data)


        # ====================================================
        # USE FIRST ROW AS CURRENT PROCESS STATE
        # ====================================================

        row = df.iloc[0].copy()


        # ====================================================
        # OVERRIDE CONTROLLABLE VALUES
        # ====================================================

        controllable_columns = [

            "machine_speed",
            "stock_flow",
            "filler_flow",
            "steam_pressure",
            "moisture",
            "ash",
            "caliper"

        ]


        for column in controllable_columns:

            if column in data:

                row[column] = safe_float(

                    data[column],

                    safe_float(
                        row.get(column)
                    )

                )


        # ====================================================
        # REBUILD ENGINEERED FEATURES
        # ====================================================

        row = prepare_model_features(
            row
        )


        # ====================================================
        # GET MODEL FEATURES
        # ====================================================

        if hasattr(
            basis_weight_model,
            "feature_names_in_"
        ):

            model_features = list(
                basis_weight_model.feature_names_in_
            )

        else:

            return jsonify({

                "success": False,

                "message": (
                    "Model does not expose "
                    "feature_names_in_."
                )

            }), 500


        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        print("\n" + "=" * 70)
        print("MODEL FEATURES")
        print("=" * 70)

        print("\nExpected features:")
        print(model_features)

        print("\nCurrent dataframe features:")
        print(list(row.index))


        # ====================================================
        # CHECK MISSING FEATURES
        # ====================================================

        missing_features = [

            feature

            for feature in model_features

            if feature not in row.index

        ]


        print("\nMissing features:")
        print(missing_features)

        print("=" * 70)


        if missing_features:

            return jsonify({

                "success": False,

                "message": (
                    "Current process data does not "
                    "contain all model features."
                ),

                "missing_features":
                    missing_features

            }), 400


        # ====================================================
        # BUILD MODEL INPUT
        # ====================================================

        X = pd.DataFrame(

            [
                [
                    row[feature]
                    for feature in model_features
                ]
            ],

            columns=model_features

        )


        # ====================================================
        # CONVERT NUMERIC FEATURES
        # ====================================================

        numeric_features = [

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
            "basis_weight_error_direction"

        ]


        for column in numeric_features:

            if column in X.columns:

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )


        # ====================================================
        # HANDLE MISSING NUMERIC VALUES
        # ====================================================

        X = X.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )


        for column in numeric_features:

            if column in X.columns:

                X[column] = X[column].fillna(0)


        print("\n" + "=" * 70)
        print("FINAL MODEL INPUT")
        print("=" * 70)

        print(X.T)

        print("=" * 70)
        
        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = basis_weight_model.predict(
            X
        )[0]


        prediction = float(
            prediction
        )


        # ====================================================
        # SETPOINT
        # ====================================================

        setpoint = safe_float(

            row.get(
                "basis_weight_setpoint",
                0
            )

        )


        # ====================================================
        # DEVIATION
        # ====================================================

        if setpoint != 0:

            deviation = (

                abs(
                    prediction -
                    setpoint
                )
                /
                setpoint
                *
                100

            )

        else:

            deviation = 0


        # ====================================================
        # OFF-SPEC DECISION
        # ====================================================

        off_spec = int(
            deviation > 2.5
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success": True,

            "predicted_basis_weight":
                round(
                    prediction,
                    3
                ),

            "predicted_deviation_pct":
                round(
                    float(deviation),
                    3
                ),

            "off_spec":
                off_spec,

            "status": (

                "OFF-SPEC"

                if off_spec

                else

                "SAFE"

            ),

            "setpoint":
                round(
                    setpoint,
                    3
                )

        }


        print("\nPrediction result:")
        print(response)


        return jsonify(
            response
        )


    except Exception as e:

        print("\nPrediction error:")
        print(str(e))


        return jsonify({

            "success": False,

            "message":
                "Prediction failed.",

            "error":
                str(e)

        }), 500


# ============================================================
# RECOMMENDATION RESULTS API
# ============================================================

@app.route("/api/recommendations", methods=["GET"])
def recommendations():

    if not os.path.exists(
        RECOMMENDATION_PATH
    ):

        return jsonify({

            "success": False,

            "message": (
                "Recommendation results are "
                "not available. Run Stage 5 first."
            )

        }), 404


    try:

        recommendations_df = pd.read_csv(
            RECOMMENDATION_PATH
        )


        recommendations_df = recommendations_df.where(

            pd.notnull(
                recommendations_df
            ),

            None

        )


        records = recommendations_df.head(
            10
        ).to_dict(
            orient="records"
        )


        return jsonify({

            "success": True,

            "count":
                len(records),

            "recommendations":
                records

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                "Could not load recommendations.",

            "error":
                str(e)

        }), 500


# ============================================================
# STABILIZATION + HISTORICAL EVIDENCE API
# ============================================================

@app.route("/api/evidence", methods=["GET"])
def evidence():

    if not os.path.exists(
        EVIDENCE_PATH
    ):

        return jsonify({

            "success": False,

            "message": (
                "Historical evidence is not available. "
                "Run Stage 6 first."
            )

        }), 404


    try:

        with open(

            EVIDENCE_PATH,

            "r",

            encoding="utf-8"

        ) as file:

            evidence_data = json.load(
                file
            )


        return jsonify({

            "success": True,

            "evidence":
                evidence_data

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message":
                "Could not load historical evidence.",

            "error":
                str(e)

        }), 500


# ============================================================
# DASHBOARD SUMMARY API
# ============================================================

@app.route("/api/dashboard", methods=["GET"])
def dashboard():

    if df.empty:

        return jsonify({

            "success": False,

            "message":
                "Dataset not loaded."

        }), 500


    row = df.iloc[0]


    current_bw = safe_float(
        row.get("basis_weight")
    )


    setpoint = safe_float(
        row.get(
            "basis_weight_setpoint"
        )
    )


    if setpoint != 0:

        current_deviation = (

            abs(
                current_bw -
                setpoint
            )
            /
            setpoint
            *
            100

        )

    else:

        current_deviation = 0


    result = {

        "success": True,

        "current_process": {

            "transition_id":
                str(
                    row.get(
                        "transition_id",
                        ""
                    )
                ),

            "grade_from":
                str(
                    row.get(
                        "grade_from",
                        ""
                    )
                ),

            "grade_to":
                str(
                    row.get(
                        "grade_to",
                        ""
                    )
                ),

            "machine_speed":
                current_process_value(
                    row,
                    "machine_speed"
                ),

            "stock_flow":
                current_process_value(
                    row,
                    "stock_flow"
                ),

            "steam_pressure":
                current_process_value(
                    row,
                    "steam_pressure"
                ),

            "basis_weight":
                current_bw,

            "setpoint":
                setpoint,

            "deviation_pct":
                round(
                    current_deviation,
                    3
                ),

            "status": (

                "OFF-SPEC / HIGH RISK"

                if current_deviation > 2.5

                else

                "SAFE"

            )

        },

        "model_status": {

            "basis_weight_model_loaded":
                basis_weight_model is not None,

            "off_spec_model_loaded":
                off_spec_model is not None

        }

    }


    return jsonify(
        result
    )


# ============================================================
# SMALL HELPER FOR DASHBOARD
# ============================================================

def current_process_value(
    row,
    column
):

    return safe_float(
        row.get(column)
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "success": False,

        "message":
            "API endpoint not found."

    }), 404


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n" +
        "=" * 70
    )

    print(
        "Starting Flask API..."
    )

    print(
        "URL: http://127.0.0.1:5000"
    )

    print(
        "=" * 70
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
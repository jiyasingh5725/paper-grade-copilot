"""
Paper Grade Transition Copilot
Stage 7 - Prediction Service

Loads the trained XGBoost models and performs:
1. Future Basis Weight prediction
2. Future OFF-SPEC classification
"""

from pathlib import Path
import pickle
import pandas as pd


# -------------------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

REGRESSION_MODEL_PATH = MODEL_DIR / "basis_weight_xgboost.pkl"
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "off_spec_xgboost.pkl"


# -------------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------------

def load_model(model_path):
    """
    Load a trained model from a pickle file.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


def load_models():
    """
    Load both regression and classification models.
    """

    regression_model = load_model(REGRESSION_MODEL_PATH)
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)

    return regression_model, classification_model


# -------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------

def predict_future_basis_weight(model, features):
    """
    Predict future Basis Weight.

    Parameters
    ----------
    model:
        Trained regression model.

    features:
        Pandas DataFrame containing model features.

    Returns
    -------
    float
        Predicted future Basis Weight.
    """

    prediction = model.predict(features)

    return float(prediction[0])


def predict_off_spec(model, features):
    """
    Predict future OFF-SPEC status.

    Returns:
        predicted_label
        probability
    """

    prediction = model.predict(features)

    predicted_label = int(prediction[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)

        if probabilities.shape[1] >= 2:
            probability = float(probabilities[0][1])

    return predicted_label, probability


# -------------------------------------------------------------------
# DEVIATION CALCULATION
# -------------------------------------------------------------------

def calculate_deviation(predicted_basis_weight, setpoint):
    """
    Calculate absolute percentage deviation from the target setpoint.
    """

    if setpoint == 0:
        return 0.0

    deviation = (
        abs(predicted_basis_weight - setpoint)
        / abs(setpoint)
    ) * 100

    return float(deviation)


def classify_status(deviation, threshold=2.5):
    """
    Convert deviation into a human-readable process status.
    """

    if deviation <= threshold:
        return "SAFE"

    return "OFF-SPEC / HIGH RISK"


# -------------------------------------------------------------------
# COMPLETE PREDICTION
# -------------------------------------------------------------------

def run_prediction(
    regression_model,
    classification_model,
    features,
    setpoint
):
    """
    Run the complete ML prediction pipeline.
    """

    predicted_basis_weight = predict_future_basis_weight(
        regression_model,
        features
    )

    predicted_deviation = calculate_deviation(
        predicted_basis_weight,
        setpoint
    )

    predicted_off_spec, off_spec_probability = predict_off_spec(
        classification_model,
        features
    )

    status = classify_status(predicted_deviation)

    return {
        "predicted_basis_weight": predicted_basis_weight,
        "predicted_deviation_pct": predicted_deviation,
        "predicted_off_spec": predicted_off_spec,
        "off_spec_probability": off_spec_probability,
        "status": status
    }
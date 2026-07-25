"""
Paper Grade Transition Copilot
Explainability Service

Reads SHAP feature importance generated in Stage 4.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHAP_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
    / "shap_feature_importance.csv"
)


def load_shap_importance():
    """
    Load SHAP feature importance.
    """

    if not SHAP_FILE.exists():
        raise FileNotFoundError(
            f"SHAP file not found:\n{SHAP_FILE}"
        )

    return pd.read_csv(SHAP_FILE)


def get_top_features(limit=10):
    """
    Return the most influential features.
    """

    df = load_shap_importance()

    if "mean_absolute_shap" in df.columns:
        df = df.sort_values(
            "mean_absolute_shap",
            ascending=False
        )

    return df.head(limit).to_dict(
        orient="records"
    )


def generate_reasoning(
    current_basis_weight,
    setpoint,
    predicted_basis_weight,
    predicted_deviation,
    safe_recommendation_found
):
    """
    Generate human-readable AI reasoning.
    """

    reasoning = []

    if predicted_deviation <= 2.5:
        reasoning.append(
            "The predicted future Basis Weight is within "
            "the 2.5% specification limit."
        )
    else:
        reasoning.append(
            "The predicted future Basis Weight remains "
            "outside the 2.5% specification limit."
        )

    if current_basis_weight < setpoint:
        reasoning.append(
            "Current Basis Weight is below the target setpoint."
        )
    elif current_basis_weight > setpoint:
        reasoning.append(
            "Current Basis Weight is above the target setpoint."
        )
    else:
        reasoning.append(
            "Current Basis Weight is currently at the target setpoint."
        )

    if safe_recommendation_found:
        reasoning.append(
            "A safe candidate operating point was identified "
            "within the tested operating limits."
        )
    else:
        reasoning.append(
            "No safe operating combination was identified "
            "within the tested controllable-variable limits."
        )

    return reasoning
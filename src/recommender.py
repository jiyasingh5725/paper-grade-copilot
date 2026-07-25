"""
Paper Grade Transition Copilot
Recommendation Service

Uses the Stage 5 recommendation-engine output.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"

CANDIDATES_FILE = (
    RESULTS_DIR / "recommendation_candidates.csv"
)

TOP_RECOMMENDATIONS_FILE = (
    RESULTS_DIR / "top_recommendations.csv"
)


def load_candidates():
    """
    Load all Stage 5 candidate recommendations.
    """

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Recommendation candidates not found:\n"
            f"{CANDIDATES_FILE}"
        )

    return pd.read_csv(CANDIDATES_FILE)


def load_top_recommendations():
    """
    Load Stage 5 ranked recommendations.
    """

    if not TOP_RECOMMENDATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Top recommendations not found:\n"
            f"{TOP_RECOMMENDATIONS_FILE}"
        )

    return pd.read_csv(TOP_RECOMMENDATIONS_FILE)


def get_safe_candidates():
    """
    Return only candidates satisfying the 2.5% specification.
    """

    df = load_candidates()

    if "predicted_deviation_pct" not in df.columns:
        return pd.DataFrame()

    safe = df[
        df["predicted_deviation_pct"] <= 2.5
    ].copy()

    return safe


def get_recommendation_summary():
    """
    Create a recommendation summary.
    """

    candidates = load_candidates()

    safe = get_safe_candidates()

    if safe.empty:
        best_candidate = None

        if not candidates.empty:
            best_candidate = (
                candidates
                .sort_values(
                    "predicted_deviation_pct"
                )
                .iloc[0]
                .to_dict()
            )

        return {
            "safe_recommendation_found": False,
            "safe_candidate_count": 0,
            "best_candidate": best_candidate
        }

    safe = safe.sort_values(
        [
            "predicted_deviation_pct",
            "change_penalty"
        ],
        ascending=[
            True,
            True
        ]
    )

    best = safe.iloc[0].to_dict()

    return {
        "safe_recommendation_found": True,
        "safe_candidate_count": int(len(safe)),
        "best_candidate": best
    }
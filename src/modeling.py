import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score


NUMERIC_FEATURES = [
    "amount_paid",
    "amount_received",
    "hour",
    "day_of_week",
]

CATEGORICAL_FEATURES = [
    "payment_currency",
    "receiving_currency",
    "payment_format",
]


def make_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build transaction-level features without changing the input."""
    features = transactions[
        ["amount_paid", "amount_received", *CATEGORICAL_FEATURES]
    ].copy()

    features["hour"] = transactions["timestamp"].dt.hour
    features["day_of_week"] = transactions["timestamp"].dt.dayofweek

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_model() -> Pipeline:
    """Create a preprocessing and classification pipeline."""
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        (
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            CATEGORICAL_FEATURES,
        ),
    ])

    classifier = HistGradientBoostingClassifier(
        categorical_features=[4, 5, 6],
        max_iter=50,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    return Pipeline([
        ("preprocess", preprocessing),
        ("model", classifier),
    ])


def evaluate_scores(
    transactions: pd.DataFrame,
    scores: np.ndarray,
    *,
    daily_budget: int = 50,
) -> dict:
    """Evaluate ranking quality and a fixed daily review capacity."""
    scores = np.asarray(scores)

    if scores.shape != (len(transactions),) or not np.isfinite(scores).all():
        raise ValueError("Supply one finite score per transaction.")
    if daily_budget < 1:
        raise ValueError("daily_budget must be positive.")

    ranked = (
        transactions[["timestamp", "is_laundering"]]
        .reset_index(drop=True)
        .copy()
    )

    positives = int(ranked["is_laundering"].sum())
    if positives == 0:
        raise ValueError("Recall requires positive examples.")

    ranked["score"] = scores
    ranked["day"] = ranked["timestamp"].dt.normalize()
    ranked["tie_breaker"] = np.random.default_rng(42).random(len(ranked))

    alerts = (
        ranked.sort_values(
            ["day", "score", "tie_breaker"],
            ascending=[True, False, True],
        )
        .groupby("day", sort=False)
        .head(daily_budget)
    )

    caught = int(alerts["is_laundering"].sum())

    return {
        "average_precision": average_precision_score(
            ranked["is_laundering"], scores
        ),
        "alerts": len(alerts),
        "laundering_caught": caught,
        "precision_at_budget": caught / len(alerts),
        "recall_at_budget": caught / positives,
    }
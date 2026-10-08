import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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
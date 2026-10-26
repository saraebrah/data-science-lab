import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.history import HISTORY_FEATURES
from src.modeling import CATEGORICAL_FEATURES


LOG_COLUMNS = [
    "amount_paid",
    "amount_received",
    *HISTORY_FEATURES,
]

TIME_COLUMNS = ["hour", "day_of_week"]


def build_neural_preprocessor() -> ColumnTransformer:
    log_scaled = Pipeline([
        (
            "log",
            FunctionTransformer(
                np.log1p,
                feature_names_out="one-to-one",
            ),
        ),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer([
        ("log_scaled", log_scaled, LOG_COLUMNS),
        ("time_scaled", StandardScaler(), TIME_COLUMNS),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            CATEGORICAL_FEATURES,
        ),
    ], sparse_threshold=0)
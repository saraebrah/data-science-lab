import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.history import HISTORY_FEATURES
from src.modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model,
    make_features,
)


@pytest.fixture
def transactions():
    size = 300
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2022-09-03", periods=size, freq="min"),
        "amount_paid": np.arange(size, dtype=float) + 1,
        "amount_received": np.arange(size, dtype=float) + 1,
        "payment_currency": ["US Dollar", "Euro"] * (size // 2),
        "receiving_currency": ["US Dollar", "Euro"] * (size // 2),
        "payment_format": ["ACH", "Wire"] * (size // 2),
        "source_row": np.arange(size),
        "is_laundering": [0, 1] * (size // 2),
    })
    for column in HISTORY_FEATURES:
        frame[column] = np.arange(size)
    return frame


def test_history_features_are_explicit_and_inputs_are_unchanged(transactions):
    original = transactions.copy(deep=True)
    baseline = make_features(transactions)
    extended = make_features(
        transactions, extra_numeric_features=tuple(HISTORY_FEATURES)
    )

    assert baseline.columns.tolist() == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert extended.columns.tolist() == (
        NUMERIC_FEATURES + HISTORY_FEATURES + CATEGORICAL_FEATURES
    )
    assert "source_row" not in extended
    assert "is_laundering" not in extended
    pd.testing.assert_frame_equal(baseline, make_features(transactions))
    pd.testing.assert_frame_equal(transactions, original)


def test_model_recognizes_categories_after_numeric_features_are_added(transactions):
    extra = tuple(HISTORY_FEATURES)
    features = make_features(transactions, extra_numeric_features=extra)
    pipeline = build_model(extra_numeric_features=extra)
    pipeline.set_params(model__max_iter=2, model__min_samples_leaf=2)

    with threadpool_limits(limits=2):
        pipeline.fit(features, transactions["is_laundering"])

        # Reversed storage category orders and a new label must remain usable.
        unseen = features.iloc[:2].copy()
        for column in CATEGORICAL_FEATURES:
            categories = sorted(features[column].unique(), reverse=True)
            unseen[column] = pd.Categorical(unseen[column], categories=categories)
        unseen["payment_format"] = unseen["payment_format"].astype("string")
        unseen["payment_format"] = "NEW_PAYMENT_TYPE"
        scores = pipeline.predict_proba(unseen)[:, 1]

    assert pipeline.named_steps["model"].is_categorical_.tolist() == (
        [False] * 8 + [True] * 3
    )
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_default_model_settings_remain_unchanged():
    model = build_model().named_steps["model"]
    assert model.categorical_features == [4, 5, 6]
    assert model.max_iter == 50
    assert model.max_leaf_nodes == 15
    assert model.learning_rate == 0.05
    assert model.l2_regularization == 1.0
    assert model.early_stopping is False
    assert model.random_state == 42


def test_excluded_format_is_not_required_or_encoded(transactions):
    configuration = {
        "extra_numeric_features": tuple(HISTORY_FEATURES),
        "excluded_features": ("payment_format",),
    }
    features = make_features(
        transactions.drop(columns="payment_format"), **configuration
    )
    assert "payment_format" not in features
    assert features.shape[1] == 10

    pipeline = build_model(**configuration)
    pipeline.set_params(model__max_iter=2, model__min_samples_leaf=2)
    with threadpool_limits(limits=2):
        pipeline.fit(features, transactions["is_laundering"])
        scores = pipeline.predict_proba(features.iloc[:2])[:, 1]

    assert pipeline.named_steps["model"].is_categorical_.tolist() == (
        [False] * 8 + [True] * 2
    )
    assert not any(
        "payment_format" in name
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    assert np.isfinite(scores).all()


def test_unknown_feature_exclusion_is_rejected(transactions):
    with pytest.raises(ValueError, match="Unknown"):
        make_features(transactions, excluded_features=("payment_formatt",))
    with pytest.raises(ValueError, match="Unknown"):
        build_model(excluded_features=("payment_formatt",))

import pandas as pd
import pytest

from src.data import summarize_activity


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "timestamp": [
            "2022/09/02 10:00",
            "2022/09/01 09:00",
            "2022/09/01 12:00",
        ],
        "payment_format": ["Wire", "ACH", "ACH"],
    })


def test_counts_add_across_chunks(transactions):
    result = summarize_activity([
        transactions.iloc[:2],
        transactions.iloc[2:],
    ])

    assert result.loc[pd.Timestamp("2022-09-01"), "ACH"] == 2
    assert result.to_numpy().sum() == 3


def test_dates_are_sorted(transactions):
    result = summarize_activity([transactions])

    assert result.index.is_monotonic_increasing


def test_chunk_boundaries_do_not_change_results(transactions):
    whole = summarize_activity([transactions])
    split = summarize_activity([
        transactions.iloc[:1],
        transactions.iloc[1:],
    ])

    pd.testing.assert_frame_equal(whole, split)
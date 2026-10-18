import pandas as pd
import pytest

from src.history import make_history_features


@pytest.fixture
def account_events():
    rows = [
        ("2022-09-02 09:59", "1", "X", "1", "A"),
        ("2022-09-02 10:00", "1", "X", "1", "A"),
        ("2022-09-03 09:00", "1", "A", "1", "Y"),
        ("2022-09-03 10:00", "1", "A", "1", "Z"),
        ("2022-09-03 10:00", "1", "A", "1", "W"),
        ("2022-09-03 11:00", "1", "X", "1", "A"),
        ("2022-09-03 09:00", "2", "A", "2", "Y"),
        ("2022-09-03 08:00", "1", "A", "1", "A"),
    ]
    frame = pd.DataFrame(rows, columns=[
        "timestamp", "sender_bank", "sender_account",
        "receiver_bank", "receiver_account",
    ])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["source_row"] = range(len(frame))
    return frame


def test_window_boundaries_identity_and_self_transfers(account_events):
    targets = account_events.loc[account_events["source_row"].isin([3, 4])]
    original = account_events.copy(deep=True)

    result = make_history_features(account_events, targets)

    assert result["source_row"].tolist() == [3, 4]
    assert result["sender_in_count_24h"].tolist() == [2, 2]
    assert result["sender_out_count_24h"].tolist() == [2, 2]
    assert result["receiver_in_count_24h"].tolist() == [0, 0]
    assert result["receiver_out_count_24h"].tolist() == [0, 0]
    pd.testing.assert_frame_equal(account_events, original)


def test_future_events_do_not_change_earlier_features(account_events):
    targets = account_events.loc[account_events["source_row"].isin([3, 4])]
    earlier = account_events.loc[
        account_events["timestamp"] <= pd.Timestamp("2022-09-03 10:00")
    ]

    pd.testing.assert_frame_equal(
        make_history_features(account_events, targets),
        make_history_features(earlier, targets),
    )


def test_input_order_does_not_change_counts(account_events):
    targets = account_events.loc[account_events["source_row"].isin([3, 4])]
    expected = make_history_features(account_events, targets)
    actual = make_history_features(
        account_events.sample(frac=1, random_state=42), targets.iloc[::-1]
    )

    assert actual["source_row"].tolist() == [4, 3]
    pd.testing.assert_frame_equal(
        actual.set_index("source_row").sort_index(),
        expected.set_index("source_row").sort_index(),
    )


def test_duplicate_source_rows_are_rejected(account_events):
    duplicated = pd.concat([account_events, account_events.iloc[:1]])

    with pytest.raises(ValueError, match="unique"):
        make_history_features(duplicated, account_events)

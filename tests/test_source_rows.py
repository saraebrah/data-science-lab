import pandas as pd
import pytest

from src.data import TRANSACTION_COLUMNS, read_source_rows


@pytest.fixture
def csv_path(tmp_path):
    rows = [
        ["2022/09/07 10:00", "001", f"A{i}", "002", f"B{i}",
         10, "Euro", 10, "Euro", "ACH", 0]
        for i in range(5)
    ]
    path = tmp_path / "transactions.csv"
    pd.DataFrame(rows, columns=TRANSACTION_COLUMNS).to_csv(path, index=False)
    return path


def test_source_positions_survive_chunking_and_filtering(csv_path):
    result = read_source_rows(
        csv_path,
        source_rows=[4, 1],
        columns=["sender_bank", "sender_account"],
        chunksize=2,
    ).set_index("source_row")

    assert set(result.index) == {1, 4}
    assert result.loc[1, "sender_account"] == "A1"
    assert result.loc[4, "sender_account"] == "A4"
    assert result.loc[1, "sender_bank"] == "001"
    assert len(result) == 2


def test_missing_source_rows_are_reported(csv_path):
    with pytest.raises(ValueError, match="not found"):
        read_source_rows(
            csv_path,
            source_rows=[99],
            columns=["sender_account"],
            chunksize=2,
        )

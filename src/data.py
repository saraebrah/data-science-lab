from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
import pandas as pd


TRANSACTION_COLUMNS = [
    "timestamp",
    "sender_bank",
    "sender_account",
    "receiver_bank",
    "receiver_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def iter_transaction_chunks(
    path: Path,
    *,
    usecols: list[str],
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Read selected transaction columns in manageable chunks."""
    identifier_columns = {
        "sender_bank",
        "sender_account",
        "receiver_bank",
        "receiver_account",
    }

    dtypes = {
        column: "string"
        for column in usecols
        if column in identifier_columns
    }

    with pd.read_csv(
        path,
        header=0,
        names=TRANSACTION_COLUMNS,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            yield chunk


def summarize_activity(
    chunks: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Count transactions by date and payment format across chunks."""
    counts = Counter()

    for chunk in chunks:
        timestamps = pd.to_datetime(
            chunk["timestamp"],
            format="%Y/%m/%d %H:%M",
            errors="raise",
        )

        dates = timestamps.dt.normalize()
        formats = chunk["payment_format"].fillna("<missing>")

        counts.update(zip(dates, formats))

    if not counts:
        raise ValueError("No transactions were supplied.")

    records = [
        {"date": date, "payment_format": payment_format, "count": count}
        for (date, payment_format), count in counts.items()
    ]

    summary = pd.DataFrame(records).pivot(
        index="date",
        columns="payment_format",
        values="count",
    )

    return summary.fillna(0).astype("int64").sort_index()


def select_period(
    transactions: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Select start <= timestamp < end; timestamps must already be parsed."""
    mask = (
        (transactions["timestamp"] >= start)
        & (transactions["timestamp"] < end)
    )
    return transactions.loc[mask].copy()


def attach_account_endpoints(
    transactions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Restore account endpoints using original CSV row identifiers."""
    endpoint_columns = [
        "sender_bank",
        "sender_account",
        "receiver_bank",
        "receiver_account",
    ]

    lookup = history[
        ["source_row", "timestamp", *endpoint_columns]
    ]

    joined = transactions.merge(
        lookup,
        on="source_row",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("", "_history"),
    )

    if not joined["timestamp"].equals(joined["timestamp_history"]):
        raise ValueError("History timestamps do not match prepared records.")

    if joined[endpoint_columns].isna().any().any():
        raise ValueError("Some transactions have missing account endpoints.")

    if not joined["source_row"].equals(
        transactions["source_row"].reset_index(drop=True)
    ):
        raise ValueError("Joining history changed transaction order.")

    return joined.drop(columns="timestamp_history")
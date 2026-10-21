"""Account activity counts using strictly earlier transaction timestamps."""

import numpy as np
import pandas as pd


HISTORY_FEATURES = [
    "sender_in_count_24h",
    "receiver_in_count_24h",
    "sender_out_count_24h",
    "receiver_out_count_24h",
]


def _counts_before(index: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    """Look up cumulative activity strictly before each query timestamp."""
    matched = pd.merge_asof(
        queries.sort_values("timestamp"),
        index,
        on="timestamp",
        by=["bank", "account"],
        direction="backward",
        allow_exact_matches=False,
    )

    return (
        matched.sort_values("_position")["cumulative_count"]
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )


def make_history_features(
    history: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Count endpoint activity in [timestamp - 24h, timestamp).

    Inputs need parsed timestamps, bank/account endpoint identifiers, and unique
    source_row identifiers. Callers supply complete observed context for the
    window; a zero means no matching events in the supplied history. The output
    follows target row order and includes source_row for safe joins. Labels and
    transaction amounts are not read. Self-transfers count once in each direction.
    """
    if not history["source_row"].is_unique or not targets["source_row"].is_unique:
        raise ValueError("SourcOkay, go to the next piece, which is index.e-row identifiers must be unique.")

    result = pd.DataFrame({"source_row": targets["source_row"].to_numpy()})

    for direction, event_role in [("in", "receiver"), ("out", "sender")]:
        events = history[
            ["timestamp", f"{event_role}_bank", f"{event_role}_account"]
        ].rename(columns={
            f"{event_role}_bank": "bank",
            f"{event_role}_account": "account",
        })

        if events.isna().any().any():
            raise ValueError("History timestamps and endpoints must be present.")

        # Aggregate simultaneous events before calculating cumulative counts.
        index = (
            events.groupby(["bank", "account", "timestamp"], sort=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values("timestamp")
        )
        index["cumulative_count"] = index.groupby(
            ["bank", "account"], sort=False
        )["count"].cumsum()
        index = index[["timestamp", "bank", "account", "cumulative_count"]]

        for role in ["sender", "receiver"]:
            queries = targets[
                ["timestamp", f"{role}_bank", f"{role}_account"]
            ].rename(columns={
                f"{role}_bank": "bank",
                f"{role}_account": "account",
            })

            if queries.isna().any().any():
                raise ValueError("Target timestamps and endpoints must be present.")

            queries["_position"] = np.arange(len(queries))
            before_now = _counts_before(index, queries)
            earlier_queries = queries.assign(
                timestamp=queries["timestamp"] - pd.Timedelta("24h")
            )
            before_window = _counts_before(index, earlier_queries)

            result[f"{role}_{direction}_count_24h"] = before_now - before_window

    return result

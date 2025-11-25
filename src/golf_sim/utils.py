from __future__ import annotations

from typing import Iterable, Set

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    df_name: str = "DataFrame",
) -> None:
    """
    Validate that a pandas DataFrame contains a set of required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : Iterable[str]
        Collection of column names that must be present in ``df``.
    df_name : str, optional
        Human-readable name for the DataFrame, used in error messages.
        Default is "DataFrame".

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    required_set: Set[str] = set(required)
    missing = required_set - set(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"{df_name} is missing required columns: {missing_sorted}")

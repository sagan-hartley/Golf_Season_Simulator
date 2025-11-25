"""
Player profile utilities for the golf season simulator.

This module provides:
    * Synthetic player generation (MeanScore / StdDev profiles).
    * CSV loading helpers for real-world player data.
    * Validation checks using the shared utilities module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from golf_sim.utils import validate_required_columns


PathLike = Union[str, Path]


def create_synthetic_players(
    num_players: int = 150,
    mean_score_start: float = 70.0,
    mean_score_end: float = 72.0,
    stddev_start: float = 2.0,
    stddev_end: float = 2.0015,
) -> pd.DataFrame:
    """
    Create a synthetic set of player scoring profiles.

    Players are assigned:
        * A Player ID from 1..num_players.
        * A mean 18-hole score, linearly spaced between mean_score_start
          and mean_score_end.
        * A standard deviation (score volatility), linearly spaced between
          stddev_start and stddev_end.

    These parameters produce a deterministic and smooth distribution of
    skill levels for simulation purposes.

    Parameters
    ----------
    num_players : int, optional
        Number of players to generate. Default is 150.
    mean_score_start : float, optional
        Mean score (strokes) for the top player (Player 1).
    mean_score_end : float, optional
        Mean score for the final player (Player num_players).
    stddev_start : float, optional
        Score standard deviation for Player 1.
    stddev_end : float, optional
        Score standard deviation for Player num_players.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "Player" (int)
        - "MeanScore" (float)
        - "StdDev" (float)

    Raises
    ------
    ValueError
        If num_players <= 0.
    """
    if num_players <= 0:
        raise ValueError("num_players must be a positive integer.")

    players = pd.DataFrame(
        {
            "Player": np.arange(1, num_players + 1, dtype=int),
            "MeanScore": np.linspace(mean_score_start, mean_score_end, num_players),
            "StdDev": np.linspace(stddev_start, stddev_end, num_players),
        }
    )
    return players


def load_players_csv(
    path: PathLike,
    player_col: str = "Player",
    mean_col: str = "MeanScore",
    stddev_col: str = "StdDev",
    df_name: Optional[str] = "players_csv",
) -> pd.DataFrame:
    """
    Load player scoring profiles from a CSV file.

    The CSV must contain at least three columns specifying the player's
    expected scoring behavior. Column names can be overridden if the CSV
    uses different headers.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    player_col : str, optional
        Column name containing a unique player identifier.
    mean_col : str, optional
        Column containing the mean 18-hole score for each player.
    stddev_col : str, optional
        Column containing score standard deviation for each player.
    df_name : str, optional
        Human-readable name for error messages. Default is "players_csv".

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - "Player"
        - "MeanScore"
        - "StdDev"
        plus any additional columns present in the CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If required columns are missing from the CSV.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Player CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate that the required columns exist in the CSV
    validate_required_columns(
        df,
        required=[player_col, mean_col, stddev_col],
        df_name=df_name,
    )

    # Rename to standardized simulator column names
    df = df.rename(
        columns={
            player_col: "Player",
            mean_col: "MeanScore",
            stddev_col: "StdDev",
        }
    )

    # Ensure proper ordering by Player ID
    df = df.sort_values("Player").reset_index(drop=True)
    return df

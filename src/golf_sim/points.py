"""Point distribution utilities for the golf season simulator.

This module provides:
    * Default elevated and regular event point distributions.
    * Helper functions to generate per-field arrays for a given number of players.
    * CSV loaders for custom point distributions.

Typical usage example:

    from golf_sim.points import (
        get_regular_point_distribution,
        get_elevated_point_distribution,
        load_point_distribution_csv,
    )

    # Points for 150-player field (regular event)
    point_distribution_reg = get_regular_point_distribution(num_players=150)

    # Custom distribution loaded from CSV
    custom_points = load_point_distribution_csv("data/real/points_2025.csv", num_players=150)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Default elevated event point distribution (rank 1..N)
# ---------------------------------------------------------------------------

DEFAULT_ELEVATED_POINTS: List[float] = [
    700,
    420,
    275,
    190,
    155,
    140,
    125,
    120,
    112,
    105,
    99.5316159250586,
    92.4222147875544,
    85.3128136500502,
    81.0471729675477,
    78.203412512546,
    75.3596520575443,
    72.5158916025427,
    69.672131147541,
    66.8283706925393,
    63.9846102375376,
    61.140849782536,
    58.2970893275343,
    55.4533288725326,
    52.6095684175309,
    50.4767480762797,
    48.3439277350284,
    46.2111073937772,
    44.0782870525259,
    41.9454667112747,
    39.8126463700234,
    37.6798260287722,
    35.5470056875209,
    33.4141853462697,
    31.2813650050184,
    29.8594847775176,
    28.4376045500167,
    27.0157243225159,
    25.5938440950151,
    24.1719638675142,
    22.7500836400134,
    21.3282034125125,
    19.9063231850117,
    18.4844429575109,
    17.06256273001,
    15.6406825025092,
    14.9297423887588,
    14.2188022750084,
    13.5078621612579,
    12.7969220475075,
    12.0859819337571,
    11.3750418200067,
    10.6641017062563,
    9.95316159250586,
    9.24222147875544,
    8.53128136500502,
    8.24690531950485,
    7.96252927400468,
    7.67815322850452,
    7.39377718300435,
    7.10940113750418,
    6.82502509200401,
    6.54064904650385,
    6.25627300100368,
    5.97189695550351,
    5.68752091000334,
    5.40314486450317,
    5.11876881900301,
    4.83439277350284,
    4.55001672800267,
    4.26564068250251,
]

# ---------------------------------------------------------------------------
# Default regular event point distribution (rank 1..N)
# ---------------------------------------------------------------------------

DEFAULT_REGULAR_POINTS: List[float] = [
    500,
    300,
    190,
    135,
    110,
    100,
    90,
    85,
    80,
    75,
    70.2576112412178,
    65.2392104382737,
    60.2208096353295,
    57.2097691535631,
    55.2024088323854,
    53.1950485112078,
    51.1876881900301,
    49.1803278688525,
    47.1729675476748,
    45.1656072264972,
    43.1582469053195,
    41.1508865841419,
    39.1435262629642,
    37.1361659417866,
    35.6306457009033,
    34.1251254600201,
    32.6196052191368,
    31.1140849782536,
    29.6085647373704,
    28.1030444964871,
    26.5975242556039,
    25.0920040147206,
    23.5864837738374,
    22.0809635329542,
    21.0772833723653,
    20.0736032117765,
    19.0699230511877,
    18.0662428905989,
    17.06256273001,
    16.0588825694212,
    15.0552024088324,
    14.0515222482436,
    13.0478420876547,
    12.0441619270659,
    11.0404817664771,
    10.5386416861827,
    10.0368016058883,
    9.53496152559384,
    9.03312144529943,
    8.53128136500502,
    8.02944128471061,
    7.52760120441619,
    7.02576112412178,
    6.52392104382737,
    6.02208096353295,
    5.82134493141519,
    5.62060889929742,
    5.41987286717966,
    5.21913683506189,
    5.01840080294413,
    4.81766477082636,
    4.6169287387086,
    4.41619270659083,
    4.21545667447307,
    4.0147206423553,
    3.81398461023754,
    3.61324857811977,
    3.412512546002,
    3.21177651388424,
    3.01104048176647,
]


def _expand_points_to_num_players(
    base_points: Sequence[float],
    num_players: int,
) -> np.ndarray:
    """Expand a base point list to a vector of length `num_players`.

    The typical usage is to take the PGA-style distribution defined for a
    limited number of ranks (e.g., 1..70) and pad it with zeros so that every
    player in a larger field gets an assigned value (usually zero).

    Args:
        base_points:
            Base point values ordered by rank (index 0 is rank 1).
        num_players:
            Total number of players in the field.

    Returns:
        NumPy array of length `num_players` where element `i` corresponds to
        rank `i + 1`.

    Raises:
        ValueError:
            If num_players is not positive.
    """
    if num_players <= 0:
        raise ValueError("num_players must be a positive integer.")

    base_len = len(base_points)
    if base_len >= num_players:
        return np.asarray(base_points[:num_players], dtype=float)

    # Pad with zeros for ranks beyond the base list.
    padded = np.zeros(num_players, dtype=float)
    padded[:base_len] = np.asarray(base_points, dtype=float)
    return padded


def get_regular_point_distribution(
    num_players: int,
    base_points: Sequence[float] | None = None,
) -> np.ndarray:
    """Get a regular event point distribution for a given field size.

    By default, this uses the hard-coded PGA-like regular event distribution
    in :data:`DEFAULT_REGULAR_POINTS` and pads with zeros beyond that range.

    Args:
        num_players:
            Number of players in the tournament field.
        base_points:
            Optional custom base point sequence ordered by rank. If provided,
            it overrides :data:`DEFAULT_REGULAR_POINTS`.

    Returns:
        NumPy array of shape (num_players,) where index 0 corresponds to
        rank 1, index 1 to rank 2, and so on.
    """
    if base_points is None:
        base_points = DEFAULT_REGULAR_POINTS
    return _expand_points_to_num_players(base_points, num_players)


def get_elevated_point_distribution(
    num_players: int,
    base_points: Sequence[float] | None = None,
) -> np.ndarray:
    """Get an elevated event point distribution for a given field size.

    By default, this uses the hard-coded PGA-like elevated event distribution
    in :data:`DEFAULT_ELEVATED_POINTS` and pads with zeros beyond that range.

    Args:
        num_players:
            Number of players in the tournament field.
        base_points:
            Optional custom base point sequence ordered by rank. If provided,
            it overrides :data:`DEFAULT_ELEVATED_POINTS`.

    Returns:
        NumPy array of shape (num_players,) where index 0 corresponds to
        rank 1, index 1 to rank 2, and so on.
    """
    if base_points is None:
        base_points = DEFAULT_ELEVATED_POINTS
    return _expand_points_to_num_players(base_points, num_players)


def load_point_distribution_csv(
    path: PathLike,
    rank_col: str = "Rank",
    points_col: str = "Points",
    num_players: int | None = None,
) -> np.ndarray:
    """Load a point distribution from a CSV file.

    The CSV is expected to contain at least two columns:
        * A rank column (e.g., "Rank") with 1-based rank indices.
        * A points column (e.g., "Points") with numeric values.

    The rows do not need to be sorted; they will be ordered by ascending rank.

    Args:
        path:
            Path to the CSV file.
        rank_col:
            Name of the column containing integer ranks (1-based).
        points_col:
            Name of the column containing point values.
        num_players:
            Optional total number of players. If provided, the resulting
            array is padded with zeros up to length num_players. If omitted,
            the length is equal to the maximum rank present in the CSV.

    Returns:
        NumPy array of shape (N,) where index 0 corresponds to rank 1.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist.
        ValueError:
            If required columns are missing or ranks are invalid.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Point distribution CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in (rank_col, points_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file {csv_path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[rank_col, points_col]].dropna()
    df = df.sort_values(by=rank_col)

    ranks = df[rank_col].to_numpy(dtype=int)
    points = df[points_col].to_numpy(dtype=float)

    if (ranks <= 0).any():
        raise ValueError("All ranks must be positive integers (1-based).")

    max_rank = int(ranks.max())
    length = num_players if num_players is not None else max_rank

    if length <= 0:
        raise ValueError("Resulting distribution length must be positive.")

    dist = np.zeros(length, dtype=float)

    for r, p in zip(ranks, points, strict=False):
        idx = r - 1
        if idx < length:
            dist[idx] = p
        # If idx >= length, silently drop that rank; caller chose a smaller field.

    return dist
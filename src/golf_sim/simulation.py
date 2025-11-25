from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from players import create_synthetic_players
from points import (
    create_default_regular_distribution,
    create_default_elevated_distribution,
)
from utils import validate_required_columns


def simulate_seasons(
    num_players: int = 150,
    num_tournaments_reg: int = 1,
    num_tournaments_elev: int = 0,
    elev_cutoff: int = 70,
    num_seasons: int = 1000,
    players_df: Optional[pd.DataFrame] = None,
    point_distribution_reg: Optional[np.ndarray] = None,
    point_distribution_elev: Optional[np.ndarray] = None,
    # Round / cut configuration (regular events)
    reg_total_rounds: int = 4,
    reg_cut_round: Optional[int] = 2,
    reg_cut_size: int = 70,
    # Round / cut configuration (elevated events)
    elev_total_rounds: int = 4,
    elev_cut_round: Optional[int] = None,
    elev_cut_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple simulated PGA-like seasons.

    This function simulates tournament scores, points, and season standings for a field
    of players over multiple seasons. It supports both regular and elevated events and
    configurable round / cut logic.

    Parameters
    ----------
    num_players : int, optional
        Total number of players in the full field. Only used if ``players_df`` is
        not supplied. Default is 150.
    num_tournaments_reg : int, optional
        Number of regular tournaments per season. Default is 1.
    num_tournaments_elev : int, optional
        Number of elevated tournaments per season. Default is 0.
    elev_cutoff : int, optional
        Maximum player ID eligible for elevated events (e.g. top 70 players by ID).
        Default is 70.
    num_seasons : int, optional
        Number of independent seasons to simulate. Default is 1000.
    players_df : pd.DataFrame, optional
        Optional DataFrame with columns ``["Player", "MeanScore", "StdDev"]``.
        If ``None``, synthetic players will be generated.
    point_distribution_reg : np.ndarray, optional
        Optional 1D array of points awarded for a regular event by finishing rank.
        Index 0 corresponds to rank 1. If ``None``, a default distribution is used.
    point_distribution_elev : np.ndarray, optional
        Optional 1D array of points awarded for an elevated event by finishing rank.
        Index 0 corresponds to rank 1. If ``None``, a default distribution is used.
    reg_total_rounds : int, optional
        Number of rounds in a regular event. Must be >= 1. Default is 4.
    reg_cut_round : int or None, optional
        Round at which to apply the cut in a regular event (e.g. 2 for a 36-hole cut).
        If ``None``, no cut is applied. Must be < ``reg_total_rounds`` if not ``None``.
        Default is 2.
    reg_cut_size : int, optional
        Number of players to make the cut in regular events. Only used if
        ``reg_cut_round`` is not ``None``. Default is 70.
    elev_total_rounds : int, optional
        Number of rounds in an elevated event. Must be >= 1. Default is 4.
    elev_cut_round : int or None, optional
        Round at which to apply the cut in elevated events. If ``None``, no cut is
        applied. Default is ``None``.
    elev_cut_size : int or None, optional
        Number of players to make the cut in elevated events. Only used if
        ``elev_cut_round`` is not ``None``. Default is ``None``.
    random_seed : int or None, optional
        Base random seed for reproducibility. Each season uses
        ``random_seed + season_idx``. If ``None``, the season index is used.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with the following keys:

        - ``"sim_results"`` : Final standings for each season.
        - ``"win_results"`` : Number of wins per player per season.
        - ``"season_results"`` : Per-tournament, per-player results for all seasons.
        - ``"agg_sim"`` : Aggregated counts of how often each player finishes in the
          top N for several N values.

    Raises
    ------
    ValueError
        If the players DataFrame is missing required columns or if configuration
        parameters are inconsistent (e.g. invalid cut round).
    """
    # Basic validation on round / cut configuration
    if reg_total_rounds < 1:
        raise ValueError("reg_total_rounds must be at least 1.")
    if elev_total_rounds < 1:
        raise ValueError("elev_total_rounds must be at least 1.")
    if reg_cut_round is not None and not (1 <= reg_cut_round < reg_total_rounds):
        raise ValueError(
            "reg_cut_round must be between 1 and reg_total_rounds - 1, "
            "or None for no cut."
        )
    if elev_cut_round is not None and not (1 <= elev_cut_round < elev_total_rounds):
        raise ValueError(
            "elev_cut_round must be between 1 and elev_total_rounds - 1, "
            "or None for no cut."
        )

    # ----------------------------------
    # Player setup
    # ----------------------------------
    if players_df is None:
        players = create_synthetic_players(num_players=num_players)
    else:
        players = players_df.copy()

    validate_required_columns(
        players,
        required={"Player", "MeanScore", "StdDev"},
        df_name="players_df",
    )

    players = players.sort_values("Player").reset_index(drop=True)

    # Field for elevated events (only players up to elev_cutoff)
    players_elev = players[players["Player"] <= elev_cutoff].copy()

    # ----------------------------------
    # Point distributions
    # ----------------------------------
    if point_distribution_reg is None:
        point_distribution_reg = create_default_regular_distribution(num_players)
    if point_distribution_elev is None:
        point_distribution_elev = create_default_elevated_distribution(elev_cutoff)

    max_idx_reg = len(point_distribution_reg)
    max_idx_elev = len(point_distribution_elev)

    # ----------------------------------
    # Accumulators for all seasons
    # ----------------------------------
    total_season_results: List[pd.DataFrame] = []
    sim_results: List[pd.DataFrame] = []
    win_results: List[pd.DataFrame] = []

    tournament_counter = 1

    # ----------------------------------
    # Season loop
    # ----------------------------------
    for season_idx in range(num_seasons):
        if random_seed is not None:
            np.random.seed(random_seed + season_idx)
        else:
            np.random.seed(season_idx)

        season_results: List[pd.DataFrame] = []

        # -----------------------
        # Elevated tournaments
        # -----------------------
        for _ in range(num_tournaments_elev):
            # Simulate round 1..elev_total_rounds
            elev_field = players_elev.copy()

            # Optionally apply a cut after elev_cut_round
            if elev_cut_round is not None and elev_cut_size is not None:
                # Pre-cut rounds
                pre_cut_rounds = list(range(1, elev_cut_round + 1))
                for r in pre_cut_rounds:
                    elev_field[f"Score{r}"] = np.random.normal(
                        elev_field["MeanScore"],
                        elev_field["StdDev"],
                    )
                pre_cut_cols = [f"Score{r}" for r in pre_cut_rounds]
                elev_field["pre_cut_scores"] = elev_field[pre_cut_cols].sum(axis=1)
                elev_field["Rank_precut"] = elev_field["pre_cut_scores"].rank(
                    method="min"
                )

                # Apply cut
                elev_field = elev_field.sort_values("Rank_precut").head(elev_cut_size)

                # Remaining rounds
                remaining_rounds = list(range(elev_cut_round + 1, elev_total_rounds + 1))
                for r in remaining_rounds:
                    elev_field[f"Score{r}"] = np.random.normal(
                        elev_field["MeanScore"],
                        elev_field["StdDev"],
                    )
                score_cols = [f"Score{r}" for r in range(1, elev_total_rounds + 1)]
            else:
                # No cut: all rounds for full elevated field
                for r in range(1, elev_total_rounds + 1):
                    elev_field[f"Score{r}"] = np.random.normal(
                        elev_field["MeanScore"],
                        elev_field["StdDev"],
                    )
                score_cols = [f"Score{r}" for r in range(1, elev_total_rounds + 1)]

            elev_field["Score"] = elev_field[score_cols].sum(axis=1)
            elev_field["Rank"] = elev_field["Score"].rank(method="min")

            elev_field["Points"] = elev_field["Rank"].apply(
                lambda r: point_distribution_elev[int(r) - 1]
                if 1 <= int(r) <= max_idx_elev
                else 0.0
            )

            temp = elev_field[["Player", "Score", "Rank", "Points"]].copy()
            temp["Tournament"] = tournament_counter
            season_results.append(temp)
            tournament_counter += 1

        # -----------------------
        # Regular tournaments
        # -----------------------
        for _ in range(num_tournaments_reg):
            reg_field = players.copy()

            # If we have a cut, simulate pre-cut rounds first
            if reg_cut_round is not None:
                pre_cut_rounds = list(range(1, reg_cut_round + 1))
                for r in pre_cut_rounds:
                    reg_field[f"Score{r}"] = np.random.normal(
                        reg_field["MeanScore"],
                        reg_field["StdDev"],
                    )
                pre_cut_cols = [f"Score{r}" for r in pre_cut_rounds]
                reg_field["pre_cut_scores"] = reg_field[pre_cut_cols].sum(axis=1)
                reg_field["Rank_precut"] = reg_field["pre_cut_scores"].rank(
                    method="min"
                )

                # Apply cut: keep top reg_cut_size
                reg_field = reg_field.sort_values("Rank_precut").head(reg_cut_size)

                # Remaining rounds
                remaining_rounds = list(range(reg_cut_round + 1, reg_total_rounds + 1))
                for r in remaining_rounds:
                    reg_field[f"Score{r}"] = np.random.normal(
                        reg_field["MeanScore"],
                        reg_field["StdDev"],
                    )

                score_cols = [f"Score{r}" for r in range(1, reg_total_rounds + 1)]
            else:
                # No cut: simulate all rounds for all players
                for r in range(1, reg_total_rounds + 1):
                    reg_field[f"Score{r}"] = np.random.normal(
                        reg_field["MeanScore"],
                        reg_field["StdDev"],
                    )
                score_cols = [f"Score{r}" for r in range(1, reg_total_rounds + 1)]

            reg_field["Score"] = reg_field[score_cols].sum(axis=1)
            reg_field["Rank"] = reg_field["Score"].rank(method="min")

            reg_field["Points"] = reg_field["Rank"].apply(
                lambda r: point_distribution_reg[int(r) - 1]
                if 1 <= int(r) <= max_idx_reg
                else 0.0
            )

            temp = reg_field[["Player", "Score", "Rank", "Points"]].copy()
            temp["Tournament"] = tournament_counter
            season_results.append(temp)
            tournament_counter += 1

        # -----------------------
        # Build season-level table
        # -----------------------
        season_df = pd.concat(season_results, ignore_index=True)
        season_df["CumulativePoints"] = season_df.groupby("Player")["Points"].cumsum()

        season_df["CumulativeRank"] = season_df.groupby("Tournament")[
            "CumulativePoints"
        ].rank(ascending=False, method="min")

        # Final season standings (by total points)
        final_standings = (
            season_df.groupby("Player", as_index=False)["Points"]
            .sum()
            .sort_values("Points", ascending=False)
            .reset_index(drop=True)
        )
        final_standings["FinalRank"] = final_standings.index + 1

        # Add top-N indicators
        for num_guy in [5, 10, 30, 50, 70, 100]:
            col_name = f"season_top_{num_guy}_finish"
            final_standings[col_name] = np.where(
                final_standings["FinalRank"] <= num_guy, 1, 0
            )

        final_standings["sim_num"] = season_idx

        # Wins per season
        wins_df = (
            season_df[season_df["Rank"] == 1]
            .groupby("Player")
            .count()[["Tournament"]]
            .sort_values(by="Tournament", ascending=False)
        )
        wins_df = wins_df.rename(columns={"Tournament": "Wins"})
        wins_df["sim_num"] = season_idx

        season_df["sim_num"] = season_idx

        total_season_results.append(season_df)
        sim_results.append(final_standings)
        win_results.append(wins_df)

    # -----------------------
    # Aggregate across seasons
    # -----------------------
    sim_df = pd.concat(sim_results, ignore_index=True)
    wins_agg = pd.concat(win_results, ignore_index=False)
    seasons_agg = pd.concat(total_season_results, ignore_index=True)

    top_cols = [c for c in sim_df.columns if "season_top_" in c.lower()]
    agg_sim = sim_df.groupby("Player")[top_cols].sum().reset_index()

    return {
        "sim_results": sim_df,
        "win_results": wins_agg,
        "season_results": seasons_agg,
        "agg_sim": agg_sim,
    }

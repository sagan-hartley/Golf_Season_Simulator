"""
Command-line wrapper for running the golf season simulator.

This script ties together:
    * Synthetic or CSV-based player profiles.
    * Default or CSV-based point distributions.
    * The core simulation engine in `simulation.py`.

It outputs:
    - agg_sim.csv               (aggregated season-top-finish counts per player)
    - sim_results.csv           (final standings for each simulation)
    - win_results.csv           (wins per player per simulation)
    - total_season_results.csv  (per-tournament, per-player results)

Optionally, it can also generate plots summarizing the results.

Usage (from project root, with src on PYTHONPATH):

    python -m golf_sim.run_simulation

or

    python src/golf_sim/run_simulation.py

With plots:

    python -m golf_sim.run_simulation --make-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

from golf_sim.players import create_synthetic_players, load_players_csv
from golf_sim.points import (
    get_default_regular_points,
    get_default_elevated_points,
    load_points_csv,
)
from golf_sim.simulation import simulate_many_seasons, RoundConfig


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the season simulator.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the PGA-style golf season simulator."
    )

    # Core simulation scale
    parser.add_argument(
        "--num-seasons",
        type=int,
        default=1000,
        help="Number of Monte Carlo seasons to simulate. Default is 1000.",
    )
    parser.add_argument(
        "--num-tournaments-reg",
        type=int,
        default=1,
        help="Number of regular events per season. Default is 1.",
    )
    parser.add_argument(
        "--num-tournaments-elev",
        type=int,
        default=0,
        help="Number of elevated events per season. Default is 0.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=150,
        help=(
            "Number of players when using synthetic players. "
            "Ignored if --players-csv is provided."
        ),
    )
    parser.add_argument(
        "--elev-cutoff",
        type=int,
        default=70,
        help="Number of players eligible for elevated events. Default is 70.",
    )

    # Player / points data sources
    parser.add_argument(
        "--players-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV with player scoring profiles. "
            "If omitted, synthetic players are generated."
        ),
    )
    parser.add_argument(
        "--points-reg-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV with regular event point distribution. "
            "If omitted, a default distribution is used."
        ),
    )
    parser.add_argument(
        "--points-elev-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV with elevated event point distribution. "
            "If omitted, a default distribution is used."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory where CSV outputs will be written. Default is data/output.",
    )

    # Round / cut config (basic knobs; defaults mimic your friend’s script)
    parser.add_argument(
        "--rounds-per-tournament",
        type=int,
        default=4,
        help="Total rounds per tournament (usually 4). Default is 4.",
    )
    parser.add_argument(
        "--cut-enabled",
        action="store_true",
        help="Enable a cut after a given round (default: disabled).",
    )
    parser.add_argument(
        "--cut-after-round",
        type=int,
        default=2,
        help="Round number after which the cut is applied. Default is 2.",
    )
    parser.add_argument(
        "--cut-size",
        type=int,
        default=70,
        help="Number of players to survive the cut. Default is 70.",
    )

    # Plotting options
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate PNG plots into the output directory.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively (requires matplotlib GUI backend).",
    )

    return parser.parse_args()


def get_players(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load or generate player profiles.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    pd.DataFrame
        Player profiles with columns: Player, MeanScore, StdDev.
    """
    if args.players_csv:
        print(f"Loading players from CSV: {args.players_csv}")
        players_df = load_players_csv(args.players_csv)
    else:
        print(f"Generating {args.num_players} synthetic players...")
        players_df = create_synthetic_players(num_players=args.num_players)
    return players_df


def get_point_distributions(
    args: argparse.Namespace, num_players: int
) -> Tuple[List[float], List[float]]:
    """
    Load or generate point distributions for regular and elevated events.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    num_players : int
        Number of players, used for validation of point distribution length.

    Returns
    -------
    (list[float], list[float])
        Tuple of (point_distribution_reg, point_distribution_elev).
    """
    # Regular events
    if args.points_reg_csv:
        print(f"Loading regular event points from CSV: {args.points_reg_csv}")
        points_reg = load_points_csv(
            args.points_reg_csv,
            num_players=num_players,
            df_name="points_regular_csv",
        )
    else:
        points_reg = get_default_regular_points(num_players=num_players)

    # Elevated events
    if args.points_elev_csv:
        print(f"Loading elevated event points from CSV: {args.points_elev_csv}")
        points_elev = load_points_csv(
            args.points_elev_csv,
            num_players=num_players,
            df_name="points_elevated_csv",
        )
    else:
        points_elev = get_default_elevated_points(num_players=num_players)

    return points_reg, points_elev


def generate_plots(
    agg_sim: pd.DataFrame,
    wins_agg: pd.DataFrame,
    seasons_agg: pd.DataFrame,
    output_dir: Path,
    num_seasons: int,
    show_plots: bool = False,
) -> None:
    """
    Generate summary plots for the simulation results.

    This function is intentionally conservative: if matplotlib is not
    available, it will simply print a warning and do nothing.

    Plots generated (if possible):
        1. Elevation probability (Top-70 season finish) by player.
        2. Total wins by player (top 20).

    Parameters
    ----------
    agg_sim : pd.DataFrame
        Aggregated per-player statistics across all simulations.
    wins_agg : pd.DataFrame
        Wins per player per simulation.
    seasons_agg : pd.DataFrame
        Full per-tournament results across all simulations.
        (Currently unused, but kept for future plot ideas.)
    output_dir : pathlib.Path
        Directory where PNG files will be written.
    num_seasons : int
        Number of simulated seasons (for converting counts to probabilities).
    show_plots : bool, optional
        If True, display plots interactively. Default is False.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; skipping plot generation. "
            "Install matplotlib to enable plotting."
        )
        return

    # 1. Elevation probability (Top-70 season finish) by player
    if "season_top_70_finish" in agg_sim.columns and num_seasons > 0:
        elevation_df = agg_sim[["Player", "season_top_70_finish"]].copy()
        elevation_df["elev_prob"] = elevation_df["season_top_70_finish"] / num_seasons

        elevation_df = elevation_df.sort_values("Player")

        plt.figure(figsize=(10, 5))
        plt.plot(elevation_df["Player"], elevation_df["elev_prob"])
        plt.xlabel("Player ID")
        plt.ylabel("Probability of Top-70 Season Finish")
        plt.title("Elevation Probability by Player (Top-70 Season Finish)")
        plt.tight_layout()

        elev_path = output_dir / "elevation_probability_by_player.png"
        plt.savefig(elev_path)
        print(f"Saved elevation probability plot to: {elev_path}")
        if show_plots:
            plt.show()
        plt.close()

    else:
        print(
            "Column 'season_top_70_finish' not found in agg_sim; "
            "skipping elevation probability plot."
        )

    # 2. Wins distribution: total wins per player across all simulations
    if "Wins" in wins_agg.columns:
        wins_total = (
            wins_agg.groupby("Player")["Wins"]
            .sum()
            .reset_index()
            .sort_values("Wins", ascending=False)
        )

        top_n = 20
        wins_top = wins_total.head(top_n)

        plt.figure(figsize=(10, 5))
        plt.bar(wins_top["Player"].astype(str), wins_top["Wins"])
        plt.xlabel("Player ID")
        plt.ylabel("Total Wins Across All Simulations")
        plt.title(f"Top {top_n} Players by Total Wins")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        wins_path = output_dir / "top_players_by_wins.png"
        plt.savefig(wins_path)
        print(f"Saved wins distribution plot to: {wins_path}")
        if show_plots:
            plt.show()
        plt.close()
    else:
        print("Column 'Wins' not found in wins_agg; skipping wins distribution plot.")


def main() -> None:
    """
    Entry point for running the golf season simulator.

    This function:
        * Parses CLI arguments.
        * Loads/generates players and point distributions.
        * Configures round / cut behavior.
        * Runs the Monte Carlo simulation.
        * Writes CSV outputs.
        * Optionally generates plots summarizing the results.
    """
    args = parse_args()

    # 1. Players
    players_df = get_players(args)
    num_players = len(players_df)

    # 2. Points
    point_distribution_reg, point_distribution_elev = get_point_distributions(
        args, num_players=num_players
    )

    # 3. Round / cut configuration
    round_config = RoundConfig(
        rounds_per_tournament=args.rounds_per_tournament,
        cut_enabled=args.cut_enabled,
        cut_after_round=args.cut_after_round,
        cut_size=args.cut_size,
    )

    # 4. Run simulations
    print(
        f"Running simulation: {args.num_seasons} seasons, "
        f"{args.num_tournaments_reg} regular events, "
        f"{args.num_tournaments_elev} elevated events..."
    )

    (
        agg_sim,      # aggregated season-top-X finish counts
        sim_df,       # final standings per sim
        wins_agg,     # wins per player per sim
        seasons_agg,  # per-tournament details
    ) = simulate_many_seasons(
        players=players_df,
        point_distribution_reg=point_distribution_reg,
        point_distribution_elev=point_distribution_elev,
        num_tournaments_reg=args.num_tournaments_reg,
        num_tournaments_elev=args.num_tournaments_elev,
        num_seasons=args.num_seasons,
        elev_cutoff=args.elev_cutoff,
        round_config=round_config,
        random_seed=None,  # or set a global seed here if you want reproducibility
    )

    # 5. Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_path = output_dir / "agg_sim.csv"
    sim_path = output_dir / "sim_results.csv"
    wins_path = output_dir / "win_results.csv"
    seasons_path = output_dir / "total_season_results.csv"

    agg_sim.to_csv(agg_path, index=False)
    sim_df.to_csv(sim_path, index=False)
    wins_agg.to_csv(wins_path, index=False)
    seasons_agg.to_csv(seasons_path, index=False)

    print("\nSimulation complete. Files written to:")
    print(f"  - {agg_path}")
    print(f"  - {sim_path}")
    print(f"  - {wins_path}")
    print(f"  - {seasons_path}")

    # 6. Small human-readable summary (text)
    if "season_top_70_finish" in agg_sim.columns:
        preview = (
            agg_sim.sort_values("season_top_70_finish", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        print("\nTop 10 players by number of top-70 season finishes (across all sims):")
        print(preview)
    else:
        print(
            "\nTip: Add season_top_X_finish columns inside simulation.py "
            "if you want aggregated top-X stats per player."
        )

    # 7. Optional plots
    if args.make_plots:
        print("\nGenerating plots...")
        generate_plots(
            agg_sim=agg_sim,
            wins_agg=wins_agg,
            seasons_agg=seasons_agg,
            output_dir=output_dir,
            num_seasons=args.num_seasons,
            show_plots=args.show_plots,
        )


if __name__ == "__main__":
    main()

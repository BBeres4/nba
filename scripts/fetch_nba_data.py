# scripts/fetch_nba_data.py
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Seasons 
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def fetch_season(season: str) -> pd.DataFrame:
    # PerGame keeps stats comparable and smaller dataset
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",  # "Base" boxscore stats
    )
    df = endpoint.get_data_frames()[0]
    df["season"] = season
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Map NBA API column names to your app's column names
    df = df.rename(
        columns={
            "PLAYER_NAME": "player",
            "TEAM_ABBREVIATION": "team",
            "GP": "games",
            "MIN": "minutes",
            "FGA": "fga",
            "FGM": "fgm",
            "FTA": "fta",
            "FTM": "ftm",
            "PTS": "pts",
            "AST": "ast",
            "REB": "reb",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
        }
    )

    if "position" not in df.columns:
        df["position"] = "UNK"

    keep = [
        "season",
        "player",
        "team",
        "position",
        "games",
        "minutes",
        "fga",
        "fgm",
        "fta",
        "ftm",
        "pts",
        "ast",
        "reb",
        "stl",
        "blk",
        "tov",
    ]

    # Keep only columns that exist (safety)
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    numeric_cols = [
        "games",
        "minutes",
        "fga",
        "fgm",
        "fta",
        "ftm",
        "pts",
        "ast",
        "reb",
        "stl",
        "blk",
        "tov",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["player"] = df["player"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()

    return df


def main() -> None:
    all_seasons: list[pd.DataFrame] = []

    for s in SEASONS:
        print(f"Fetching season {s}...")
        season_df = fetch_season(s)
        all_seasons.append(season_df)

        # NBA Stats endpoint can rate limit; sleep a bit
        time.sleep(1.25)

    df = pd.concat(all_seasons, ignore_index=True)
    df = normalize_columns(df)

    out_path = DATA_DIR / "player_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved real NBA data to: {out_path}")


if __name__ == "__main__":
    main()

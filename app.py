import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Paths (repo-relative)
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"


def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file at {CSV_PATH}. Ensure data/player_stats.csv exists."
        )

    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip().lower() for col in df.columns]

    numeric_cols = [
        "season",
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
        "home_games",
        "away_games",
        "wins",
        "losses",
    ]

    # Coerce numeric columns (missing columns won't error if you guard)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure sane minimums (avoid division by zero)
    if "minutes" in df.columns:
        df["minutes"] = df["minutes"].clip(lower=1)
    if "games" in df.columns:
        df["games"] = df["games"].clip(lower=1)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper().str.strip()
    if "position" in df.columns:
        df["position"] = df["position"].astype(str).str.upper().str.strip()
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()

    # Make season a clean Python int for Streamlit widgets
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)

    return df


def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TS% denominator
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["ts_pct"] = np.where(denom > 0, df["pts"] / denom, 0)

    # Usage rate (simple proxy)
    df["usage_rate"] = np.where(
        df["minutes"] > 0,
        ((df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]) * 100,
        0,
    )

    # Simple PER-like metric (not official PER)
    df["per"] = (
        (df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 + df["stl"] * 2 + df["blk"] * 2)
        / df["games"]
        - df["tov"] * 0.5
    )

    df["ts_pct"] = pd.to_numeric(df["ts_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df["usage_rate"] = pd.to_numeric(df["usage_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df["per"] = pd.to_numeric(df["per"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def ensure_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If DB exists and is newer than CSV, keep it
    if DB_PATH.exists() and CSV_PATH.exists():
        if DB_PATH.stat().st_mtime >= CSV_PATH.stat().st_mtime:
            return

    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


def load_filtered_data(season: int | None, team: str | None, player: str | None) -> pd.DataFrame:
    query = "SELECT * FROM player_stats WHERE 1=1"
    params: list[object] = []

    if season is not None:
        query += " AND season = ?"
        params.append(int(season))

    if team:
        query += " AND team = ?"
        params.append(team)

    if player:
        query += " AND player = ?"
        params.append(player)

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def comparison_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    return df.set_index("player")[metrics]


def most_improved(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "season"])
    grouped = df.groupby("player")["per"].agg(["first", "last"])
    grouped["delta"] = grouped["last"] - grouped["first"]
    return grouped.sort_values("delta", ascending=False).head(10)


def most_efficient(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["games"] >= 40].sort_values("ts_pct", ascending=False).head(10)


# ------------------- Streamlit App -------------------

st.set_page_config(page_title="NBA Player Performance Dashboard", layout="wide")

try:
    ensure_database()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Make sure the data/ folder exists and includes player_stats.csv "
        "(and that you pulled the latest GitHub changes)."
    )
    st.stop()

st.title("NBA Player Performance Analytics Dashboard")

with sqlite3.connect(DB_PATH) as conn:
    base = pd.read_sql_query("SELECT * FROM player_stats", conn)

# Ensure season is int in-memory (helps widgets)
if "season" in base.columns:
    base["season"] = pd.to_numeric(base["season"], errors="coerce").fillna(0).astype(int)

seasons = sorted(base["season"].unique().tolist()) if "season" in base.columns else []
teams = sorted(base["team"].dropna().astype(str).unique().tolist()) if "team" in base.columns else []
players = sorted(ba


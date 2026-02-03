import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
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
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["minutes"] = df["minutes"].clip(lower=1)
    df["games"] = df["games"].clip(lower=1)
    df["team"] = df["team"].str.upper()
    df["position"] = df["position"].str.upper()
    return df


def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_pct"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]))
    df["usage_rate"] = (
        (df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]
    ) * 100
    df["per"] = (
        df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 + df["stl"] * 2 + df["blk"] * 2
    ) / df["games"] - df["tov"] * 0.5
    df["ts_pct"] = df["ts_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def ensure_database() -> None:
    if DB_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


def load_filtered_data(
    season: int | None, team: str | None, player: str | None
) -> pd.DataFrame:
    query = "SELECT * FROM player_stats WHERE 1=1"
    params: list[object] = []
    if season is not None:
        query += " AND season = ?"
        params.append(season)
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


st.set_page_config(page_title="NBA Player Performance Dashboard", layout="wide")
try:
    ensure_database()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "If you cloned the repo, make sure the data/ folder exists and includes "
        "player_stats.csv. You can also open the CSV in Excel to verify it downloaded "
        "correctly."
    )
    st.stop()

st.title("NBA Player Performance Analytics Dashboard")

with sqlite3.connect(DB_PATH) as conn:
    base = pd.read_sql_query("SELECT * FROM player_stats", conn)

seasons = sorted(base["season"].unique())
teams = sorted(base["team"].unique())
players = sorted(base["player"].unique())

st.sidebar.header("Filters")
season_filter = st.sidebar.selectbox("Season", [None] + seasons, format_func=lambda x: x or "All")
team_filter = st.sidebar.selectbox("Team", [None] + teams, format_func=lambda x: x or "All")
player_filter = st.sidebar.selectbox("Player", [None] + players, format_func=lambda x: x or "All")

filtered = load_filtered_data(season_filter, team_filter, player_filter)

col1, col2, col3 = st.columns(3)
col1.metric("Players", filtered["player"].nunique())
col2.metric("Total Points", f"{filtered['pts'].sum():,.0f}")
col3.metric("Avg TS%", f"{filtered['ts_pct'].mean():.3f}")

st.subheader("Performance Overview")
overview = (
    filtered.groupby(["season", "player"], as_index=False)[
        ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    ]
    .mean()
    .sort_values("per", ascending=False)
)
st.dataframe(overview, use_container_width=True)

st.subheader("Advanced Stats Distribution")
chart_data = filtered[["player", "ts_pct", "usage_rate", "per"]].set_index("player")
st.bar_chart(chart_data)

st.subheader("Compare Two Players")
compare_left, compare_right = st.columns(2)
player_a = compare_left.selectbox("Player A", players, index=0)
player_b = compare_right.selectbox("Player B", players, index=1)

comparison = filtered[filtered["player"].isin([player_a, player_b])]
comparison_summary = (
    comparison.groupby("player", as_index=False)[
        ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    ]
    .mean()
)
st.dataframe(comparison_metrics(comparison_summary), use_container_width=True)

improved_col, efficient_col = st.columns(2)
with improved_col:
    st.subheader("Most Improved (PER)")
    st.dataframe(most_improved(base), use_container_width=True)
with efficient_col:
    st.subheader("Most Efficient (TS%)")
    st.dataframe(most_efficient(base), use_container_width=True)

st.caption("Data source: Synthetic dataset stored in SQLite for demo purposes.")

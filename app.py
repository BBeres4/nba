import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Paths (repo-relative) ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"


# ---------- Data loading & prep ----------
def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file at {CSV_PATH}. Ensure data/player_stats.csv exists."
        )

    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip().lower() for col in df.columns]

    # Required columns check (helps debugging if CSV columns changed)
    required = {"season", "player", "team", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Keep denominators safe
    df["minutes"] = df["minutes"].clip(lower=1)
    df["games"] = df["games"].clip(lower=1)

    # Clean strings
    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()

    # Make sure season is plain int (fixes Streamlit widget issues)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)

    return df


def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TS% = PTS / (2*(FGA + 0.44*FTA)) with safe denominator
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["ts_pct"] = np.where(denom > 0, df["pts"] / denom, 0)

    # Simple usage proxy
    df["usage_rate"] = np.where(
        df["minutes"] > 0,
        ((df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]) * 100,
        0,
    )

    # Simple PER-like score (not official PER)
    df["per"] = (
        (df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 + df["stl"] * 2 + df["blk"] * 2)
        / df["games"]
        - df["tov"] * 0.5
    )

    # Clean any weird values
    for col in ["ts_pct", "usage_rate", "per"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

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


# ---------- Queries / analytics ----------
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


# ---------- Streamlit UI ----------
st.set_page_config(page_title="NBA Player Performance Dashboard", layout="wide")

try:
    ensure_database()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info(
        "Fix: confirm your repo has data/player_stats.csv (and that it has columns like "
        "'season', 'player', 'team', 'position')."
    )
    st.stop()

st.title("NBA Player Performance Analytics Dashboard")

with sqlite3.connect(DB_PATH) as conn:
    base = pd.read_sql_query("SELECT * FROM player_stats", conn)

# Ensure season is int in memory too
base["season"] = pd.to_numeric(base["season"], errors="coerce").fillna(0).astype(int)
base["team"] = base["team"].astype(str)
base["player"] = base["player"].astype(str)

seasons = sorted(base["season"].unique().tolist())
teams = sorted(base["team"].dropna().unique().tolist())
players = sorted(base["player"].dropna().unique().tolist())

st.sidebar.header("Filters")

# ✅ FIXED: format_func always returns a string
season_filter = st.sidebar.selectbox(
    "Season",
    [None] + seasons,
    format_func=lambda x: "All" if x is None else str(int(x)),
)

team_filter = st.sidebar.selectbox(
    "Team",
    [None] + teams,
    format_func=lambda x: "All" if x is None else str(x),
)

player_filter = st.sidebar.selectbox(
    "Player",
    [None] + players,
    format_func=lambda x: "All" if x is None else str(x),
)

filtered = load_filtered_data(season_filter, team_filter, player_filter)

col1, col2, col3 = st.columns(3)
col1.metric("Players", int(filtered["player"].nunique()) if not filtered.empty else 0)
col2.metric("Total Points", f"{filtered['pts'].sum():,.0f}" if not filtered.empty else "0")
col3.metric("Avg TS%", f"{filtered['ts_pct'].mean():.3f}" if not filtered.empty else "0.000")

st.subheader("Performance Overview")
if filtered.empty:
    st.warning("No rows match your filters. Try selecting 'All' for one or more filters.")
else:
    overview = (
        filtered.groupby(["season", "player"], as_index=False)[
            ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
        ]
        .mean()
        .sort_values("per", ascending=False)
    )
    st.dataframe(overview, use_container_width=True)

st.subheader("Advanced Stats Distribution")
if filtered.empty:
    st.info("Chart will appear when your filters return data.")
else:
    chart_data = filtered[["player", "ts_pct", "usage_rate", "per"]].set_index("player")
    st.bar_chart(chart_data)

st.subheader("Compare Two Players")
if len(players) < 2:
    st.info("Need at least two players in the dataset to compare.")
else:
    compare_left, compare_right = st.columns(2)
    player_a = compare_left.selectbox("Player A", players, index=0)
    player_b = compare_right.selectbox("Player B", players, index=1)

    comparison = filtered[filtered["player"].isin([player_a, player_b])]
    if comparison.empty:
        st.warning("No comparison data for those players with current filters. Try 'All' filters.")
    else:
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

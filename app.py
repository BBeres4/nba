import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"


def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CSV_PATH}\n\n"
            "Run:\n"
            "  python scripts/fetch_nba_data.py\n"
            "to download real NBA data into data/player_stats.csv"
        )

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns for your app
    required = {"season", "player", "team", "games", "minutes", "pts", "ast", "reb", "fga", "fta", "tov"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

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

    df["games"] = df["games"].clip(lower=1)
    df["minutes"] = df["minutes"].clip(lower=1)

    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["player"] = df["player"].astype(str).str.strip()
    if "position" in df.columns:
        df["position"] = df["position"].astype(str).str.upper().str.strip()
    else:
        df["position"] = "UNK"

    # keep season as string like "2023-24" (don’t force to int now)
    df["season"] = df["season"].astype(str).str.strip()

    return df


def calculate_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["ts_pct"] = np.where(denom > 0, df["pts"] / denom, 0)

    df["usage_rate"] = np.where(
        df["minutes"] > 0,
        ((df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]) * 100,
        0,
    )

    df["per"] = (
        (df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 + df["stl"] * 2 + df["blk"] * 2)
        / df["games"]
        - df["tov"] * 0.5
    )

    for c in ["ts_pct", "usage_rate", "per"]:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    return df


def ensure_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Rebuild DB if missing, or if CSV updated
    if DB_PATH.exists() and CSV_PATH.exists():
        if DB_PATH.stat().st_mtime >= CSV_PATH.stat().st_mtime:
            return

    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


@st.cache_data(show_spinner=False)
def read_base() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM player_stats", conn)


st.set_page_config(page_title="NBA Player Performance Dashboard (Real Data)", layout="wide")

try:
    ensure_database()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

base = read_base()

st.title("NBA Player Performance Analytics Dashboard (Real Data)")
st.caption("Data source: NBA Stats API via nba_api")

# Filters
seasons = sorted(base["season"].unique().tolist())
teams = sorted(base["team"].unique().tolist())
players = sorted(base["player"].unique().tolist())

st.sidebar.header("Filters")
season_filter = st.sidebar.selectbox("Season", ["All"] + seasons)
team_filter = st.sidebar.selectbox("Team", ["All"] + teams)
player_filter = st.sidebar.selectbox("Player", ["All"] + players)

filtered = base.copy()
if season_filter != "All":
    filtered = filtered[filtered["season"] == season_filter]
if team_filter != "All":
    filtered = filtered[filtered["team"] == team_filter]
if player_filter != "All":
    filtered = filtered[filtered["player"] == player_filter]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", int(filtered["player"].nunique()) if not filtered.empty else 0)
c2.metric("Total Points", f"{filtered['pts'].sum():,.0f}" if not filtered.empty else "0")
c3.metric("Avg TS%", f"{filtered['ts_pct'].mean():.3f}" if not filtered.empty else "0.000")
c4.metric("Avg PER", f"{filtered['per'].mean():.2f}" if not filtered.empty else "0.00")

st.subheader("Data Preview")
st.dataframe(filtered.sort_values("per", ascending=False), use_container_width=True)

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="NBA Player Performance Analytics",
    page_icon="🏀",
    layout="wide",
)

# -------------------- PATHS --------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"

st.caption(f"Running file: {Path(__file__).resolve()}")

# -------------------- DATA PIPELINE --------------------
def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    numeric_cols = [
        "season", "games", "minutes", "fga", "fgm", "fta", "ftm",
        "pts", "ast", "reb", "stl", "blk", "tov",
        "home_games", "away_games", "wins", "losses",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["season"] = df["season"].astype(int)
    df["games"] = df["games"].clip(lower=1)
    df["minutes"] = df["minutes"].clip(lower=1)

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()

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


def rebuild_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


@st.cache_data(show_spinner=False)
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM player_stats", conn)
    df["season"] = df["season"].astype(int)
    return df


# -------------------- INIT --------------------
if not DB_PATH.exists():
    rebuild_database()

if st.sidebar.button("🔁 Rebuild database"):
    rebuild_database()
    st.cache_data.clear()
    st.rerun()

base = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("Filters")

seasons = sorted(base["season"].unique())
teams = sorted(base["team"].unique())
players = sorted(base["player"].unique())

season_label = st.sidebar.selectbox("Season", ["All"] + [str(s) for s in seasons])
season_filter = None if season_label == "All" else int(season_label)

team_label = st.sidebar.selectbox("Team", ["All"] + teams)
team_filter = None if team_label == "All" else team_label

player_search = st.sidebar.text_input("Search Player")
player_list = [p for p in players if player_search.lower() in p.lower()]
player_label = st.sidebar.selectbox("Player", ["All"] + player_list)
player_filter = None if player_label == "All" else player_label

filtered = base.copy()
if season_filter is not None:
    filtered = filtered[filtered["season"] == season_filter]
if team_filter is not None:
    filtered = filtered[filtered["team"] == team_filter]
if player_filter is not None:
    filtered = filtered[filtered["player"] == player_filter]

# -------------------- HEADER --------------------
st.title("NBA Player Performance Analytics Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Players", filtered["player"].nunique())
k2.metric("Total Points", f"{filtered['pts'].sum():,.0f}")
k3.metric("Avg TS%", f"{filtered['ts_pct'].mean():.3f}")
k4.metric("Avg PER", f"{filtered['per'].mean():.2f}")

st.download_button(
    "⬇️ Download Filtered CSV",
    data=filtered.to_csv(index=False),
    file_name="nba_filtered.csv",
)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🧑 Player Explorer", "⚔️ Compare", "🧠 Insights"]
)

# -------------------- OVERVIEW --------------------
with tab1:
    st.subheader("Top Players by PER")
    st.dataframe(
        filtered.sort_values("per", ascending=False).head(15),
        use_container_width=True,
    )

    st.subheader("Stat Distribution")
    metric = st.selectbox("Metric", ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"])
    bins = pd.cut(filtered[metric], bins=16)
    hist = bins.value_counts().sort_index()
    hist.index = hist.index.astype(str)  # 🔥 FIXED
    st.bar_chart(hist)

# -------------------- PLAYER EXPLORER --------------------
with tab2:
    if player_filter is None:
        st.info("Select a player from the sidebar.")
    else:
        pdata = base[base["player"] == player_filter].sort_values("season")
        latest = pdata.iloc[-1]

        a, b, c = st.columns(3)
        a.metric("Latest PER", f"{latest['per']:.2f}")
        b.metric("Latest TS%", f"{latest['ts_pct']:.3f}")
        c.metric("Latest Usage", f"{latest['usage_rate']:.1f}")

        st.subheader("Trends Over Time")
        st.line_chart(pdata.set_index("season")[["per", "ts_pct", "usage_rate"]])

        st.subheader("Season Table")
        st.dataframe(pdata, use_container_width=True)

# -------------------- COMPARE --------------------
with tab3:
    p1 = st.selectbox("Player A", players)
    p2 = st.selectbox("Player B", players, index=1)

    comp = base[base["player"].isin([p1, p2])]
    summary = comp.groupby("player")[["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]].mean()
    st.dataframe(summary, use_container_width=True)

# -------------------- INSIGHTS --------------------
with tab4:
    st.subheader("Correlation Matrix")
    corr = filtered[["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]].corr()
    st.dataframe(corr, use_container_width=True)

st.caption("Built with Streamlit • pandas • SQLite")



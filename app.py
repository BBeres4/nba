import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------
st.set_page_config(page_title="NBA Analytics Dashboard v2", page_icon="🏀", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"

st.caption(f"Running: {Path(__file__).resolve()}")  # proves which file is running


# -------------------- DATA PIPELINE --------------------
def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"season", "player", "team", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    numeric_cols = [
        "season", "games", "minutes", "fga", "fgm", "fta", "ftm",
        "pts", "ast", "reb", "stl", "blk", "tov",
        "home_games", "away_games", "wins", "losses",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
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


def rebuild_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


@st.cache_data(show_spinner=False)
def read_base() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        base = pd.read_sql_query("SELECT * FROM player_stats", conn)

    base["season"] = pd.to_numeric(base["season"], errors="coerce").fillna(0).astype(int)
    base["team"] = base["team"].astype(str)
    base["player"] = base["player"].astype(str)
    base["position"] = base["position"].astype(str)
    return base


def filter_df(base: pd.DataFrame, season: int | None, team: str | None, player: str | None) -> pd.DataFrame:
    df = base
    if season is not None:
        df = df[df["season"] == int(season)]
    if team:
        df = df[df["team"] == team]
    if player:
        df = df[df["player"] == player]
    return df.copy()


def leaderboard(df: pd.DataFrame, metric: str, top_n: int, min_games: int) -> pd.DataFrame:
    d = df[df["games"] >= min_games].copy()
    d = d.sort_values(metric, ascending=False).head(top_n)
    cols = ["season", "player", "team", "position", "games", "pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    cols = [c for c in cols if c in d.columns]
    return d[cols]


# -------------------- FIRST RUN SAFETY --------------------
try:
    if not DB_PATH.exists():
        rebuild_database()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Sidebar DB controls
st.sidebar.header("Controls")
if st.sidebar.button("🔁 Rebuild database from CSV"):
    rebuild_database()
    st.cache_data.clear()
    st.rerun()

base = read_base()

# -------------------- SIDEBAR FILTERS (string-only, always safe) --------------------
st.sidebar.header("Filters")

seasons = sorted(base["season"].unique().tolist())
teams = sorted(base["team"].dropna().unique().tolist())
players = sorted(base["player"].dropna().unique().tolist())

season_label = st.sidebar.selectbox("Season", ["All"] + [str(int(s)) for s in seasons])
season_filter = None if season_label == "All" else int(season_label)

team_label = st.sidebar.selectbox("Team", ["All"] + [str(t) for t in teams])
team_filter = None if team_label == "All" else team_label

player_search = st.sidebar.text_input("Search player", "")
player_list = [p for p in players if player_search.lower() in p.lower()]
player_label = st.sidebar.selectbox("Player", ["All"] + player_list)
player_filter = None if player_label == "All" else player_label

min_games = st.sidebar.slider("Min games (leaderboards)", 0, 82, 40, 1)

filtered = filter_df(base, season_filter, team_filter, player_filter)

# -------------------- HEADER + KPIs --------------------
st.title("🏀 NBA Player Performance Analytics (v2)")
st.caption("SQLite-backed • Filters • Leaderboards • Trends • Insights")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Players", int(filtered["player"].nunique()) if not filtered.empty else 0)
k2.metric("Total Points", f"{filtered['pts'].sum():,.0f}" if not filtered.empty else "0")
k3.metric("Avg TS%", f"{filtered['ts_pct'].mean():.3f}" if not filtered.empty else "0.000")
k4.metric("Avg PER", f"{filtered['per'].mean():.2f}" if not filtered.empty else "0.00")

st.download_button(
    "⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="nba_filtered.csv",
    mime="text/csv",
)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧑‍💻 Player Explorer", "⚔️ Compare", "🧠 Insights"])

with tab1:
    c1, c2 = st.columns([1.2, 0.8], gap="large")
    with c1:
        st.subheader("Top PER (Leaderboard)")
        if filtered.empty:
            st.warning("No data with current filters.")
        else:
            st.dataframe(leaderboard(filtered, "per", 15, min_games), use_container_width=True, height=420)

    with c2:
        st.subheader("Pick a leaderboard metric")
        metric = st.selectbox("Metric", ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"])
        if filtered.empty:
            st.info("Adjust filters to see results.")
        else:
            lb = leaderboard(filtered, metric, 10, min_games)
            st.dataframe(lb[["player", "team", "games", metric]], use_container_width=True, height=420)

    st.subheader("Distributions")
    if filtered.empty:
        st.info("No rows to chart.")
    else:
        m = st.selectbox("Distribution metric", ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"], index=0)
        bins = pd.cut(filtered[m], bins=16)
        hist = bins.value_counts().sort_index()
        st.bar_chart(hist)

with tab2:
    st.subheader("Player Profile")
    if player_filter is None:
        st.info("Choose a player from the sidebar to see trends + profile.")
    else:
        pdata = base[base["player"] == player_filter].sort_values("season")
        if pdata.empty:
            st.warning("No rows for that player.")
        else:
            latest = pdata.iloc[-1]
            a, b, c = st.columns(3)
            a.markdown(f"### {player_filter}")
            a.write(f"**Team:** {latest['team']}")
            a.write(f"**Position:** {latest['position']}")
            a.write(f"**Seasons:** {pdata['season'].nunique()}")

            b.metric("Latest PER", f"{latest['per']:.2f}")
            b.metric("Latest TS%", f"{latest['ts_pct']:.3f}")
            b.metric("Latest Usage", f"{latest['usage_rate']:.1f}")

            c.metric("Latest PTS", f"{latest['pts']:.1f}")
            c.metric("Latest REB", f"{latest['reb']:.1f}")
            c.metric("Latest AST", f"{latest['ast']:.1f}")

            st.divider()
            st.subheader("Trends over seasons")
            trend_cols = st.multiselect(
                "Trend metrics",
                ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"],
                default=["per", "ts_pct", "usage_rate"],
            )
            if trend_cols:
                st.line_chart(pdata.set_index("season")[trend_cols])

            st.subheader("Season table")
            cols = ["season", "team", "position", "games", "pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
            st.dataframe(pdata[cols], use_container_width=True)

with tab3:
    st.subheader("Compare Two Players")
    if len(players) < 2:
        st.info("Need at least 2 players in dataset.")
    else:
        left, right = st.columns(2)
        p1 = left.selectbox("Player A", players, index=0)
        p2 = right.selectbox("Player B", players, index=1)

        comp_base = base.copy()
        if season_filter is not None:
            comp_base = comp_base[comp_base["season"] == season_filter]
        if team_filter is not None:
            comp_base = comp_base[comp_base["team"] == team_filter]

        comp = comp_base[comp_base["player"].isin([p1, p2])]
        if comp.empty:
            st.warning("No rows match those players with current filters. Try Season=All and Team=All.")
        else:
            summary = comp.groupby("player", as_index=False)[["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]].mean()
            st.dataframe(summary.set_index("player"), use_container_width=True)

            st.subheader("PER over seasons")
            per_trend = (
                base[base["player"].isin([p1, p2])]
                .groupby(["season", "player"], as_index=False)["per"]
                .mean()
                .pivot(index="season", columns="player", values="per")
            )
            st.line_chart(per_trend)

with tab4:
    st.subheader("Correlation & Quick Insights")
    if filtered.empty:
        st.info("Adjust filters to generate insights.")
    else:
        cols = ["pts", "reb", "ast", "ts_pct", "usage_rate", "per", "tov", "minutes", "fga", "fta"]
        cols = [c for c in cols if c in filtered.columns]
        corr = filtered[cols].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)

        # Tiny auto-insight
        if "per" in corr.index and "ts_pct" in corr.columns:
            st.write(f"PER ↔ TS% correlation: **{corr.loc['per','ts_pct']:+.2f}**")
        if "per" in corr.index and "usage_rate" in corr.columns:
            st.write(f"PER ↔ Usage correlation: **{corr.loc['per','usage_rate']:+.2f}**")

st.caption("v2 file: dashboard_v2.py • CSV → SQLite → Streamlit")


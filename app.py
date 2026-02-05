import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="NBA Player Performance Analytics",
    page_icon="🏀",
    layout="wide",
)

# -------------------- Paths --------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "nba_stats.sqlite"
CSV_PATH = DATA_DIR / "player_stats.csv"


# -------------------- Data Prep --------------------
def load_raw_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file at {CSV_PATH}. Ensure data/player_stats.csv exists."
        )

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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

    for col in ["ts_pct", "usage_rate", "per"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    return df


def ensure_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep DB if it's newer than CSV
    if DB_PATH.exists() and CSV_PATH.exists():
        if DB_PATH.stat().st_mtime >= CSV_PATH.stat().st_mtime:
            return

    df = calculate_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("player_stats", conn, if_exists="replace", index=False)


@st.cache_data(show_spinner=False)
def read_base() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        base = pd.read_sql_query("SELECT * FROM player_stats", conn)
    # Normalize types for UI reliability
    base["season"] = pd.to_numeric(base["season"], errors="coerce").fillna(0).astype(int)
    base["team"] = base["team"].astype(str)
    base["player"] = base["player"].astype(str)
    base["position"] = base["position"].astype(str)
    return base


def load_filtered_data(base: pd.DataFrame, season: int | None, team: str | None, player: str | None) -> pd.DataFrame:
    df = base
    if season is not None:
        df = df[df["season"] == int(season)]
    if team:
        df = df[df["team"] == team]
    if player:
        df = df[df["player"] == player]
    return df.copy()


def percentile(series: pd.Series, value: float) -> float:
    if series.empty:
        return 0.0
    return float((series <= value).mean() * 100)


# -------------------- UI Helpers --------------------
def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", int(df["player"].nunique()) if not df.empty else 0)
    c2.metric("Total Points", f"{df['pts'].sum():,.0f}" if not df.empty else "0")
    c3.metric("Avg TS%", f"{df['ts_pct'].mean():.3f}" if not df.empty else "0.000")
    c4.metric("Avg PER", f"{df['per'].mean():.2f}" if not df.empty else "0.00")


def leaderboard(df: pd.DataFrame, metric: str, n: int = 10, min_games: int = 40) -> pd.DataFrame:
    d = df[df["games"] >= min_games].copy()
    d = d.sort_values(metric, ascending=False).head(n)
    cols = ["season", "player", "team", "position", "games", "pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    cols = [c for c in cols if c in d.columns]
    return d[cols]


# -------------------- Start App --------------------
try:
    ensure_database()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Make sure your repo contains data/player_stats.csv with the required columns.")
    st.stop()

base = read_base()

# -------------------- Sidebar --------------------
st.sidebar.title("🏀 Filters")

seasons = sorted(base["season"].unique().tolist())
teams = sorted(base["team"].dropna().unique().tolist())
players = sorted(base["player"].dropna().unique().tolist())

# Use string-only dropdowns (always safe)
season_label = st.sidebar.selectbox("Season", ["All"] + [str(int(s)) for s in seasons])
season_filter = None if season_label == "All" else int(season_label)

team_label = st.sidebar.selectbox("Team", ["All"] + [str(t) for t in teams])
team_filter = None if team_label == "All" else team_label

player_query = st.sidebar.text_input("Search Player", value="")
player_pick = st.sidebar.selectbox(
    "Player",
    ["All"] + [p for p in players if player_query.lower() in p.lower()],
)
player_filter = None if player_pick == "All" else player_pick

st.sidebar.divider()
min_games = st.sidebar.slider("Minimum Games (leaderboards)", 0, 82, 40, 1)
st.sidebar.caption("Tip: Use search + filters to drill down fast.")

filtered = load_filtered_data(base, season_filter, team_filter, player_filter)

# -------------------- Header --------------------
st.title("NBA Player Performance Analytics Dashboard")
st.caption("Portfolio-style Streamlit analytics app • SQLite-backed • Advanced metrics + insights")

kpi_row(filtered)

# Download button
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv_bytes,
    file_name="nba_filtered.csv",
    mime="text/csv",
)

# -------------------- Tabs --------------------
tab_overview, tab_players, tab_compare, tab_insights = st.tabs(
    ["📊 Overview", "🧑‍💻 Player Explorer", "⚔️ Comparisons", "🧠 Insights"]
)

# -------------------- Overview Tab --------------------
with tab_overview:
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.subheader("Top Performers (PER)")
        if filtered.empty:
            st.warning("No data matches your filters.")
        else:
            top = leaderboard(filtered, "per", n=15, min_games=min_games)
            st.dataframe(top, use_container_width=True, height=420)

    with right:
        st.subheader("Leaderboards")
        if filtered.empty:
            st.info("Adjust filters to see leaderboards.")
        else:
            m = st.selectbox("Metric", ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"])
            lb = leaderboard(filtered, m, n=10, min_games=min_games)
            st.dataframe(lb[["player", "team", "games", m]], use_container_width=True, height=420)

    st.subheader("Distribution")
    if filtered.empty:
        st.info("No rows to chart.")
    else:
        metric = st.selectbox("Distribution Metric", ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"], index=0)
        # Streamlit histogram-style chart using bar_chart on binned data
        bins = pd.cut(filtered[metric], bins=15)
        hist = bins.value_counts().sort_index()
        st.bar_chart(hist)

# -------------------- Player Explorer Tab --------------------
with tab_players:
    st.subheader("Player Snapshot")
    if player_filter is None:
        st.info("Pick a player in the sidebar to see a detailed profile.")
    else:
        pdata = base[base["player"] == player_filter].sort_values("season")
        latest = pdata.iloc[-1] if not pdata.empty else None

        if pdata.empty:
            st.warning("No records found for that player.")
        else:
            a, b, c = st.columns([1, 1, 1])
            with a:
                st.markdown(f"### {player_filter}")
                st.write(f"**Team:** {latest['team']}  \n**Position:** {latest['position']}")
                st.write(f"**Seasons in dataset:** {pdata['season'].nunique()}")

            with b:
                st.metric("Latest Season PER", f"{latest['per']:.2f}")
                st.metric("Latest Season TS%", f"{latest['ts_pct']:.3f}")
                st.metric("Latest Season Usage", f"{latest['usage_rate']:.1f}")

            with c:
                # Percentiles within current season filter if available, else across all data
                ref = base if season_filter is None else base[base["season"] == season_filter]
                per_p = percentile(ref["per"], float(latest["per"]))
                ts_p = percentile(ref["ts_pct"], float(latest["ts_pct"]))
                usg_p = percentile(ref["usage_rate"], float(latest["usage_rate"]))
                st.metric("PER Percentile", f"{per_p:.0f}th")
                st.metric("TS% Percentile", f"{ts_p:.0f}th")
                st.metric("Usage Percentile", f"{usg_p:.0f}th")

            st.divider()
            st.subheader("Trends Over Time")
            trend_cols = st.multiselect(
                "Select trend metrics",
                ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"],
                default=["per", "ts_pct", "usage_rate"],
            )
            if trend_cols:
                trend_df = pdata[["season"] + trend_cols].set_index("season")
                st.line_chart(trend_df)

            st.subheader("Season-by-season table")
            show_cols = ["season", "team", "position", "games", "pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
            st.dataframe(pdata[show_cols], use_container_width=True)

# -------------------- Comparisons Tab --------------------
with tab_compare:
    st.subheader("Compare Two Players")
    if len(players) < 2:
        st.info("Need at least two players to compare.")
    else:
        c1, c2 = st.columns(2)
        pA = c1.selectbox("Player A", players, index=0)
        pB = c2.selectbox("Player B", players, index=1)

        comp_base = base
        if season_filter is not None:
            comp_base = comp_base[comp_base["season"] == season_filter]
        if team_filter is not None:
            comp_base = comp_base[comp_base["team"] == team_filter]

        comp = comp_base[comp_base["player"].isin([pA, pB])].copy()
        if comp.empty:
            st.warning("No comparison rows with current filters. Try Season=All / Team=All.")
        else:
            summary = (
                comp.groupby("player", as_index=False)[
                    ["pts", "reb", "ast", "ts_pct", "usage_rate", "per", "games"]
                ]
                .mean()
            )

            st.dataframe(summary.set_index("player"), use_container_width=True)

            st.subheader("Scatter: Usage vs Efficiency")
            # scatter via Streamlit: use dataframe with x/y columns; st.scatter_chart will pick numeric columns
            sc = comp[["player", "usage_rate", "ts_pct", "per"]].copy()
            sc["player_tag"] = sc["player"]
            sc = sc.set_index("player_tag")
            st.scatter_chart(sc[["usage_rate", "ts_pct"]])

            st.subheader("PER over seasons")
            per_trend = (
                base[base["player"].isin([pA, pB])]
                .groupby(["season", "player"], as_index=False)["per"]
                .mean()
                .pivot(index="season", columns="player", values="per")
            )
            st.line_chart(per_trend)

# -------------------- Insights Tab --------------------
with tab_insights:
    st.subheader("Relationships & Quick Insights")

    if filtered.empty:
        st.info("Adjust filters to generate insights.")
    else:
        insight_cols = ["pts", "reb", "ast", "ts_pct", "usage_rate", "per", "tov", "fga", "fta", "minutes"]
        insight_cols = [c for c in insight_cols if c in filtered.columns]

        corr = filtered[insight_cols].corr(numeric_only=True)

        st.write("**Correlation Matrix** (filtered data)")
        st.dataframe(corr, use_container_width=True)

        st.subheader("Key takeaways (auto)")
        # Simple heuristic insights
        per_ts = corr.loc["per", "ts_pct"] if "per" in corr.index and "ts_pct" in corr.columns else 0
        per_usg = corr.loc["per", "usage_rate"] if "per" in corr.index and "usage_rate" in corr.columns else 0
        per_tov = corr.loc["per", "tov"] if "per" in corr.index and "tov" in corr.columns else 0

        bullets = []
        bullets.append(f"- PER correlation with TS%: **{per_ts:+.2f}** (higher = stronger relationship)")
        bullets.append(f"- PER correlation with Usage: **{per_usg:+.2f}**")
        if "tov" in corr.columns:
            bullets.append(f"- PER correlation with Turnovers: **{per_tov:+.2f}** (negative is usually good)")

        st.markdown("\n".join(bullets))

        st.subheader("Most Improved (PER) across seasons")
        tmp = base.sort_values(["player", "season"])
        g = tmp.groupby("player")["per"].agg(["first", "last"])
        g["delta"] = g["last"] - g["first"]
        st.dataframe(g.sort_values("delta", ascending=False).head(15), use_container_width=True)

# -------------------- Footer --------------------
st.caption("Built with Streamlit + pandas + SQLite • Synthetic dataset for demo.")

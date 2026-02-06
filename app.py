import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="NBA Player Analytics (Real Data)",
    page_icon="🏀",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "player_stats.csv"
DB_PATH = DATA_DIR / "nba_stats.sqlite"

st.caption(f"Running: {Path(__file__).resolve()}")

# -------------------- DATA --------------------
def load_raw_data():
    if not CSV_PATH.exists():
        st.error("Missing real NBA data. Run: python scripts/fetch_nba_data.py")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower().strip() for c in df.columns]

    numeric = [
        "games","minutes","pts","ast","reb","stl","blk",
        "tov","fga","fgm","fta","ftm"
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["games"] = df["games"].clip(lower=1)
    df["minutes"] = df["minutes"].clip(lower=1)
    df["team"] = df["team"].str.upper()
    df["player"] = df["player"].str.strip()
    df["season"] = df["season"].astype(str)

    return df


def add_advanced_stats(df):
    df = df.copy()
    df["ts_pct"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]))
    df["ts_pct"] = df["ts_pct"].replace([np.inf, -np.inf], 0).fillna(0)

    df["usage_rate"] = (
        (df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]
    ) * 100

    df["per"] = (
        (df["pts"]
         + df["reb"] * 1.2
         + df["ast"] * 1.5
         + df["stl"] * 2
         + df["blk"] * 2)
        / df["games"]
        - df["tov"] * 0.5
    )
    return df


def ensure_db():
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        return
    df = add_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("players", conn, if_exists="replace", index=False)


@st.cache_data
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM players", conn)


ensure_db()
base = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("Filters")

seasons = sorted(base["season"].unique())
teams = sorted(base["team"].unique())
players = sorted(base["player"].unique())

season = st.sidebar.selectbox("Season", ["All"] + seasons)
team = st.sidebar.selectbox("Team", ["All"] + teams)
player = st.sidebar.selectbox("Player", ["All"] + players)
min_games = st.sidebar.slider("Min Games", 0, 82, 40)

df = base.copy()
if season != "All":
    df = df[df["season"] == season]
if team != "All":
    df = df[df["team"] == team]
if player != "All":
    df = df[df["player"] == player]

# -------------------- HEADER --------------------
st.title("🏀 NBA Player Performance Dashboard (Real Data)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", df["player"].nunique())
c2.metric("Total PTS", f"{df['pts'].sum():,.0f}")
c3.metric("Avg TS%", f"{df['ts_pct'].mean():.3f}")
c4.metric("Avg PER", f"{df['per'].mean():.2f}")

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🧑 Player", "🏟 Team Hub", "🏁 Rankings", "🧩 Archetypes", "🧠 Insights"]
)

# -------------------- OVERVIEW --------------------
with tab1:
    st.subheader("Top Players by PER")
    st.dataframe(
        df[df["games"] >= min_games]
        .sort_values("per", ascending=False)
        .head(20),
        use_container_width=True,
    )

    st.subheader("Metric Distribution")
    metric = st.selectbox("Metric", ["per","ts_pct","usage_rate","pts","ast","reb"])
    bins = pd.cut(df[metric], 16)
    hist = bins.value_counts().sort_index()
    hist.index = hist.index.astype(str)
    st.bar_chart(hist)

# -------------------- PLAYER --------------------
with tab2:
    if player == "All":
        st.info("Select a player to explore.")
    else:
        pdata = base[base["player"] == player].sort_values("season")
        latest = pdata.iloc[-1]

        a,b,c = st.columns(3)
        a.metric("PER", f"{latest['per']:.2f}")
        b.metric("TS%", f"{latest['ts_pct']:.3f}")
        c.metric("Usage", f"{latest['usage_rate']:.1f}")

        st.line_chart(
            pdata.set_index("season")[["per","ts_pct","usage_rate"]]
        )

# -------------------- TEAM HUB --------------------
with tab3:
    team_pick = st.selectbox("Team", teams)
    tdf = base[base["team"] == team_pick]

    st.metric("Players", tdf["player"].nunique())
    st.metric("Avg PER", f"{tdf['per'].mean():.2f}")

    roster = (
        tdf[tdf["games"] >= min_games]
        .groupby("player", as_index=False)[["pts","reb","ast","per"]]
        .mean()
        .sort_values("per", ascending=False)
    )
    st.dataframe(roster, use_container_width=True)

# -------------------- RANKINGS --------------------
with tab4:
    st.subheader("Custom Ranking")

    w_per = st.slider("PER weight", 0.0, 3.0, 1.5)
    w_ts = st.slider("TS% weight", 0.0, 3.0, 1.0)
    w_pts = st.slider("PTS weight", 0.0, 3.0, 0.5)

    ref = base[base["games"] >= min_games]
    z = (ref[["per","ts_pct","pts"]] - ref[["per","ts_pct","pts"]].mean()) / ref[["per","ts_pct","pts"]].std()
    score = z["per"]*w_per + z["ts_pct"]*w_ts + z["pts"]*w_pts

    out = ref[["player","team","season"]].copy()
    out["score"] = score

    st.dataframe(
        out.groupby("player",as_index=False)["score"]
        .mean()
        .sort_values("score",ascending=False)
        .head(25),
        use_container_width=True
    )

# -------------------- ARCHETYPES --------------------
with tab5:
    st.subheader("Player Archetypes (KMeans)")

    features = ["pts","reb","ast","ts_pct","usage_rate","per"]
    ref = base[base["games"] >= min_games]
    X = ref.groupby("player")[features].mean()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = st.slider("Number of archetypes", 4, 10, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    X["archetype"] = labels.astype(str)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)
    X["pca1"] = Z[:,0]
    X["pca2"] = Z[:,1]

    chart = (
        alt.Chart(X.reset_index())
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x="pca1",
            y="pca2",
            color="archetype",
            tooltip=["player","archetype"]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(X.reset_index().sort_values("archetype"), use_container_width=True)

# -------------------- INSIGHTS --------------------
with tab6:
    st.subheader("League Correlations")
    corr = df[["pts","reb","ast","ts_pct","usage_rate","per"]].corr()
    st.dataframe(corr, use_container_width=True)

st.caption("Real NBA data via NBA Stats API • Streamlit • scikit-learn")


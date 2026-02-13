import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================== CONFIG ====================
st.set_page_config(
    page_title="NBA Player Analytics",
    page_icon="🏀",
    layout="wide",
)

# Simple CSS to make the app feel like a dashboard (cards, spacing, nicer metrics)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1300px; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .muted { opacity: 0.75; font-size: 0.95rem; }

      /* Make metric cards look like dashboard tiles */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px 16px;
        border-radius: 16px;
      }

      .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
      }

      button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== PATHS ====================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "player_stats.csv"
DB_PATH = DATA_DIR / "nba_stats.sqlite"

# ==================== DATA ====================
def load_raw_data() -> pd.DataFrame:
    """Load the CSV and clean it so the app can rely on consistent columns."""
    if not CSV_PATH.exists():
        st.error("Missing NBA data. Run: python scripts/fetch_nba_data.py")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower().strip() for c in df.columns]

    numeric_cols = [
        "games", "minutes", "pts", "ast", "reb", "stl", "blk",
        "tov", "fga", "fgm", "fta", "ftm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Keep rate stats safe (avoid divide-by-zero)
    if "games" in df.columns:
        df["games"] = df["games"].clip(lower=1)
    if "minutes" in df.columns:
        df["minutes"] = df["minutes"].clip(lower=1)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper()
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)

    return df


def add_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra “helper” stats that are easier to compare than raw box score numbers.
    These are not perfect — they’re meant for exploration and learning.
    """
    df = df.copy()

    # TS% (True Shooting %) = how efficiently someone scores (includes free throws)
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["ts_pct"] = (df["pts"] / denom).replace([np.inf, -np.inf], 0).fillna(0)

    # Usage rate proxy = how involved a player is (more shots/turnovers = more usage)
    df["usage_rate"] = ((df["fga"] + 0.44 * df["fta"] + df["tov"]) / df["minutes"]) * 100

    # PER-like score (demo metric; NOT official NBA PER)
    # Think of it as: “overall box score impact per game”
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


def ensure_db() -> None:
    """Create the SQLite DB once if it doesn't exist (faster app reruns)."""
    DATA_DIR.mkdir(exist_ok=True)
    if DB_PATH.exists():
        return

    df = add_advanced_stats(load_raw_data())
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("players", conn, if_exists="replace", index=False)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Read data from SQLite. Cached so Streamlit reruns stay fast."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM players", conn)


ensure_db()
base = load_data()

# ==================== SIDEBAR ====================
st.sidebar.header("Filters")
st.sidebar.caption("Use these filters to narrow what you’re looking at.")

seasons = sorted(base["season"].unique())
teams = sorted(base["team"].unique())
players = sorted(base["player"].unique())

with st.sidebar.expander("Season / Team / Player", expanded=True):
    season = st.selectbox("Season", ["All"] + seasons)
    team = st.selectbox("Team", ["All"] + teams)
    player = st.selectbox("Player", ["All"] + players)

with st.sidebar.expander("Minimums", expanded=True):
    min_games = st.slider("Min Games", 0, 82, 40)
    st.caption("Higher minimums remove small sample sizes (more reliable).")

# ==================== FILTER DATA ====================
df = base.copy()
if season != "All":
    df = df[df["season"] == season]
if team != "All":
    df = df[df["team"] == team]
if player != "All":
    df = df[df["player"] == player]
df = df[df["games"] >= min_games]

# ==================== HEADER ====================
st.title("🏀 NBA Player Analytics")

st.markdown(
    '<div class="muted">A beginner-friendly dashboard to explore player performance, player “types”, and simple trends.</div>',
    unsafe_allow_html=True,
)
st.write("")

# “What do these mean?” helper box
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**Quick guide to the main stats:**")
st.markdown(
    """
- **PTS** = points scored per game (scoring volume)  
- **TS%** = scoring efficiency (how many points per shot attempt)  
- **Usage** = how involved a player is in the offense  
- **PER (demo)** = an “overall impact” score built from box score stats (not official NBA PER)
"""
)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# Summary metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", int(df["player"].nunique()))
c2.metric("Total PTS", f"{df['pts'].sum():,.0f}")
c3.metric("Avg TS%", f"{df['ts_pct'].mean():.3f}")
c4.metric("Avg PER (demo)", f"{df['per'].mean():.2f}")
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🧑 Player", "🏟 Team Hub", "🏁 Rankings", "🧩 Archetypes", "🧠 Insights"]
)

# ==================== OVERVIEW ====================
with tab1:
    st.subheader("Overview")
    st.markdown(
        '<div class="muted">Start here: a simple top list + a distribution chart to understand what’s “normal”.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown("**Top Players by Overall Impact (PER demo)**")
    st.caption("This is a quick “who pops out?” list. Use the filters in the sidebar to narrow it down.")

    top = (
        df.sort_values("per", ascending=False)
        .loc[:, ["player", "team", "season", "games", "pts", "reb", "ast", "ts_pct", "usage_rate", "per"]]
        .head(20)
    )

    st.dataframe(
        top.style.format({
            "pts": "{:.1f}",
            "reb": "{:.1f}",
            "ast": "{:.1f}",
            "ts_pct": "{:.3f}",
            "usage_rate": "{:.1f}",
            "per": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.write("")
    st.markdown("**How are players distributed?**")
    metric = st.selectbox(
        "Pick a metric to see its distribution",
        ["per", "ts_pct", "usage_rate", "pts", "ast", "reb"],
        format_func=lambda m: {
            "per": "PER (Overall Impact)",
            "ts_pct": "TS% (Efficiency)",
            "usage_rate": "Usage (Involvement)",
            "pts": "PTS (Scoring)",
            "ast": "AST (Assists)",
            "reb": "REB (Rebounds)",
        }[m],
    )

    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=24), title=""),
            y=alt.Y("count():Q", title="Number of players"),
            tooltip=["count():Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(hist, use_container_width=True)

    st.markdown(
        '<div class="muted">Tip: If you see a long tail, it means a small number of players are extreme outliers.</div>',
        unsafe_allow_html=True,
    )

# ==================== PLAYER ====================
with tab2:
    st.subheader("Player Snapshot")
    st.markdown(
        '<div class="muted">Pick a player in the sidebar to see their latest season + how their stats changed over time.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if player == "All":
        st.info("Select a player from the sidebar to view details.")
    else:
        pdata = base[base["player"] == player].sort_values("season")
        latest = pdata.iloc[-1]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {player}")
        st.markdown(f'<div class="muted">{latest["team"]} • Season {latest["season"]}</div>', unsafe_allow_html=True)

        a, b, c, d = st.columns(4)
        a.metric("PER (demo)", f"{latest['per']:.2f}")
        b.metric("TS% (efficiency)", f"{latest['ts_pct']:.3f}")
        c.metric("Usage (involvement)", f"{latest['usage_rate']:.1f}")
        d.metric("Games", f"{latest['games']:.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Trend chart: see whether the player improved, declined, or stayed consistent across seasons.")

        trend = pdata[["season", "per", "ts_pct", "usage_rate"]].melt(
            "season", var_name="metric", value_name="value"
        )

        metric_names = {
            "per": "PER (Overall Impact)",
            "ts_pct": "TS% (Efficiency)",
            "usage_rate": "Usage (Involvement)",
        }
        trend["metric"] = trend["metric"].map(metric_names)

        line = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("season:N", title="Season"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["season", "metric", "value"],
            )
            .properties(height=340)
        )
        st.altair_chart(line, use_container_width=True)

# ==================== TEAM HUB ====================
with tab3:
    st.subheader("Team Hub")
    st.markdown(
        '<div class="muted">Pick a team to see a simple roster summary (average per player).</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    team_pick = st.selectbox("Team", teams)
    tdf = base[(base["team"] == team_pick) & (base["games"] >= min_games)]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("Players", int(tdf["player"].nunique()))
    b.metric("Avg PER (demo)", f"{tdf['per'].mean():.2f}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    st.caption("Roster table = each player’s average stats (filtered by minimum games).")

    roster = (
        tdf.groupby("player", as_index=False)[["pts", "reb", "ast", "per"]]
        .mean()
        .sort_values("per", ascending=False)
        .round(2)
    )
    roster = roster.rename(columns={"pts": "PTS", "reb": "REB", "ast": "AST", "per": "PER (demo)"})
    st.dataframe(roster, use_container_width=True)

# ==================== RANKINGS ====================
with tab4:
    st.subheader("Custom Ranking")
    st.markdown(
        '<div class="muted">This is a “build your own definition of best player” tool.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**How it works (simple):**")
    st.markdown(
        """
- You choose how important each metric is (weights).  
- We compare players to the league average, then combine the stats into one score.  
- Higher score = better according to *your* definition.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # Weights (user-controlled)
    w_per = st.slider("PER weight (overall impact)", 0.0, 3.0, 1.5)
    st.caption("Higher = you care more about overall box-score impact.")

    w_ts = st.slider("TS% weight (efficiency)", 0.0, 3.0, 1.0)
    st.caption("Higher = you care more about efficient scoring.")

    w_pts = st.slider("PTS weight (scoring volume)", 0.0, 3.0, 0.5)
    st.caption("Higher = you care more about players who score a lot.")

    ref = base[base["games"] >= min_games]
    stats = ["per", "ts_pct", "pts"]

    # Z-scores normalize scales
    z = (ref[stats] - ref[stats].mean()) / ref[stats].std()
    score = z["per"] * w_per + z["ts_pct"] * w_ts + z["pts"] * w_pts

    out = ref[["player", "team", "season"]].copy()
    out["score"] = score

    st.subheader("Top Players by Your Score")
    st.caption("This list updates instantly when you change the weights.")

    st.dataframe(
        out.groupby("player", as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=False)
        .head(25),
        use_container_width=True,
    )

# ==================== ARCHETYPES ====================
with tab5:
    st.subheader("Player Archetypes (10 Types)")
    st.markdown(
        '<div class="muted">Archetypes are “player types”. We use clustering to group players with similar stat profiles.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**How to use this tab:**")
    st.markdown(
        """
- The **map** groups similar players close together.  
- The **color** shows the player archetype.  
- Use the dropdown to see who is in each archetype.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    st.info(
        "Note: This dataset doesn’t include 3-point tracking or advanced defense metrics, "
        "so some archetype names are best-effort."
    )

    features = ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    ref = base[base["games"] >= min_games].copy()
    X = ref.groupby("player")[features].mean()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = 10
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    X["cluster"] = km.fit_predict(Xs).astype(int)

    z = pd.DataFrame(scaler.transform(X[features]), index=X.index, columns=features)
    cluster_z = z.join(X["cluster"]).groupby("cluster")[features].mean()

    remaining = set(cluster_z.index.tolist())
    cluster_to_role = {}

    def pick_cluster(score_fn):
        best_c = None
        best_s = None
        for c in remaining:
            s = score_fn(cluster_z.loc[c])
            if best_s is None or s > best_s:
                best_s = s
                best_c = c
        remaining.remove(best_c)
        return best_c

    c = pick_cluster(lambda p: p["usage_rate"] + p["pts"] + p["ast"])
    cluster_to_role[c] = "Primary Creator"

    c = pick_cluster(lambda p: p["ast"] + 0.25 * p["usage_rate"])
    cluster_to_role[c] = "Playmaker"

    c = pick_cluster(lambda p: p["pts"] + 0.25 * p["usage_rate"])
    cluster_to_role[c] = "Shot Creator"

    c = pick_cluster(lambda p: p["pts"] + p["ts_pct"] - 0.6 * p["usage_rate"])
    cluster_to_role[c] = "Off-Ball Scorer"

    c = pick_cluster(lambda p: 1.5 * p["ts_pct"] - 0.5 * p["usage_rate"] + 0.2 * p["pts"])
    cluster_to_role[c] = "Efficient Finisher"

    c = pick_cluster(lambda p: 1.5 * p["reb"] + 0.2 * p["per"] - 0.2 * p["pts"])
    cluster_to_role[c] = "Rebounder / Hustle Big"

    c = pick_cluster(lambda p: 1.2 * p["per"] - 0.6 * p["pts"] - 0.4 * p["usage_rate"])
    cluster_to_role[c] = "Defensive Anchor"

    c = pick_cluster(lambda p: -0.8 * p["usage_rate"] + 0.8 * p["ts_pct"] + 0.6 * p["per"])
    cluster_to_role[c] = "3&D Wing"

    c = pick_cluster(lambda p: 0.6 * p["pts"] + 0.6 * p["reb"] + 0.6 * p["ast"] + 0.6 * p["per"])
    cluster_to_role[c] = "Two-Way Wing"

    last_c = remaining.pop()
    cluster_to_role[last_c] = "Energy / Bench Role"

    X["role"] = X["cluster"].map(cluster_to_role)

    Z2 = PCA(n_components=2, random_state=42).fit_transform(Xs)
    X["pca1"], X["pca2"] = Z2[:, 0], Z2[:, 1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Players clustered", int(X.shape[0]))
    m2.metric("Archetypes", 10)
    m3.metric("Unique labels", int(X["role"].nunique()))
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    st.caption("Map tip: closer dots = more similar stat profiles. Hover a dot to see the player name.")

    chart = (
        alt.Chart(X.reset_index())
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("pca1:Q", title="Style Map (left/right)"),
            y=alt.Y("pca2:Q", title="Style Map (up/down)"),
            color=alt.Color("role:N", title="Archetype"),
            tooltip=["player:N", "role:N"],
        )
        .interactive()
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Archetype Profiles (Average Stats)")
    st.caption("This table shows the average stats for each archetype (so you can compare types).")

    role_profile = (
        X.reset_index()
        .groupby("role")[features]
        .mean()
        .round(2)
        .sort_values("per", ascending=False)
    )
    st.dataframe(role_profile, use_container_width=True)

    st.subheader("Players in Each Archetype")
    st.caption("Pick an archetype to see which players belong to it.")

    role_pick = st.selectbox("Pick an archetype", sorted(X["role"].unique()), key="role_pick")
    players_df = X.reset_index()
    players_in_role = (
        players_df.loc[players_df["role"] == role_pick, ["player"] + features]
        .sort_values("per", ascending=False)
        .head(30)
        .round(2)
    )
    st.dataframe(players_in_role, use_container_width=True)

# ==================== INSIGHTS ====================
with tab6:
    st.markdown("### Insights")
    st.markdown(
        '<div class="muted">This tab shows simple patterns (what stats move together). It’s a starting point for learning.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    cols = ["pts", "reb", "ast", "ts_pct", "usage_rate", "per"]
    pretty = {
        "pts": "PTS (Scoring)",
        "reb": "REB (Rebounds)",
        "ast": "AST (Assists)",
        "ts_pct": "TS% (Efficiency)",
        "usage_rate": "Usage (Involvement)",
        "per": "PER (Overall Impact)",
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**🗺 Heatmap**")
        st.markdown('<div class="muted">Darker = stronger link.</div>', unsafe_allow_html=True)
    with s2:
        st.markdown("**💡 Takeaways**")
        st.markdown('<div class="muted">Top relationships explained.</div>', unsafe_allow_html=True)
    with s3:
        st.markdown("**🔎 Scatter**")
        st.markdown('<div class="muted">Pick two stats and see players.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Players in view", int(df["player"].nunique()))
    b.metric("Avg PTS", f"{df['pts'].mean():.1f}")
    c.metric("Avg TS%", f"{df['ts_pct'].mean():.3f}")
    d.metric("Avg PER (demo)", f"{df['per'].mean():.2f}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    st.subheader("Correlation Heatmap")
    st.caption("Correlation ranges from -1 to +1. Near 0 means there isn’t a clear pattern.")

    corr = df[cols].corr()
    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="stat_y", value_name="corr")
        .rename(columns={"index": "stat_x"})
    )

    heat = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("stat_x:N", title="", sort=cols),
            y=alt.Y("stat_y:N", title="", sort=cols),
            color=alt.Color("corr:Q", title="Correlation", scale=alt.Scale(domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("stat_x:N", title="Stat A"),
                alt.Tooltip("stat_y:N", title="Stat B"),
                alt.Tooltip("corr:Q", title="Correlation", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(heat, use_container_width=True)

    st.subheader("Key Takeaways")
    st.caption("These are the strongest relationships in the current view.")

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.loc[cols[i], cols[j]])))

    pairs_df = (
        pd.DataFrame(pairs, columns=["stat_a", "stat_b", "corr"])
        .assign(abs_corr=lambda d: d["corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(6)
        .drop(columns="abs_corr")
    )

    def strength_label(v: float) -> str:
        av = abs(v)
        if av >= 0.70:
            return "Strong"
        if av >= 0.40:
            return "Moderate"
        return "Weak"

    takeaways = pairs_df.copy()
    takeaways["Relationship"] = takeaways.apply(
        lambda r: f"{pretty[r['stat_a']]} ↔ {pretty[r['stat_b']]}",
        axis=1,
    )
    takeaways["Insight"] = takeaways.apply(
        lambda r: f"{strength_label(r['corr'])} link — these usually move together."
        if r["corr"] > 0
        else f"{strength_label(r['corr'])} link — one tends to rise as the other falls.",
        axis=1,
    )
    takeaways = takeaways[["Relationship", "corr", "Insight"]]

    st.dataframe(
        takeaways.style.format({"corr": "{:.2f}"}),
        use_container_width=True,
    )

    st.subheader("Explore a Relationship")
    st.caption("Pick two stats. Each dot is a player-season row. Hover a dot to see who it is.")

    left, right = st.columns(2)
    with left:
        x_stat = st.selectbox("X-axis", cols, index=0, format_func=lambda x: pretty[x])
    with right:
        y_stat = st.selectbox("Y-axis", cols, index=5, format_func=lambda x: pretty[x])

    scatter = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=70)
        .encode(
            x=alt.X(f"{x_stat}:Q", title=pretty[x_stat]),
            y=alt.Y(f"{y_stat}:Q", title=pretty[y_stat]),
            tooltip=[
                "player:N",
                "team:N",
                "season:N",
                alt.Tooltip(f"{x_stat}:Q", title=pretty[x_stat], format=".2f"),
                alt.Tooltip(f"{y_stat}:Q", title=pretty[y_stat], format=".2f"),
            ],
        )
        .interactive()
        .properties(height=360)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        **Simple interpretation:**
        - **Positive** correlation → players high in one stat are often high in the other  
        - **Negative** correlation → players high in one stat are often lower in the other  
        - **Near zero** → no strong pattern here  

        Correlation shows a pattern, not a reason.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

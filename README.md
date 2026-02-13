# 🏀 NBA Player Analytics (Streamlit)

A beginner-friendly Streamlit dashboard that lets you explore NBA player performance with simple filters, easy visualizations, custom rankings, player “archetypes” (player types), and basic insights.

---

## What this app does (in plain English)

This project takes NBA player data and helps you answer questions like:

- Who are the top performers (by an overall impact score)?
- How efficient is a player at scoring?
- Which players are similar “types” (e.g., creators, playmakers, finishers)?
- If you change what you care about (impact vs efficiency vs points), who ranks highest?
- Which stats tend to move together (patterns, not cause-and-effect)?

---

## Pages / Tabs

### 📊 Overview
- Shows the **Top Players** by a demo “overall impact” score.
- Shows a **distribution chart** so you can see what’s normal vs rare.

### 🧑 Player
- Pick a player and see their latest season snapshot:
  - **PER (demo)**, **TS%**, **Usage**, and **Games**
- View a simple **trend chart** across seasons.

### 🏟 Team Hub
- Pick a team to see:
  - number of players in the dataset for that team
  - average impact score
  - roster table (player averages)

### 🏁 Rankings
Create your own ranking by choosing how important each stat is:
- **PER weight** (overall impact)
- **TS% weight** (efficiency)
- **PTS weight** (scoring volume)

The app standardizes stats first so they’re comparable, then combines them into one score.

### 🧩 Archetypes
Groups players into **10 player types** using clustering:
- Primary Creator
- Playmaker
- Shot Creator
- Off-Ball Scorer
- Efficient Finisher
- Rebounder / Hustle Big
- Defensive Anchor (best-effort)
- 3&D Wing (best-effort)
- Two-Way Wing
- Energy / Bench Role

It also includes:
- a 2D “style map” (PCA) where closer players = more similar
- archetype profile averages
- a list of players inside each archetype

### 🧠 Insights
A “patterns” page using correlation:
- Heatmap showing relationships between stats
- “Key Takeaways” list (strongest relationships)
- Scatter plot to explore any two stats interactively

---

## Stats explained (for non-basketball watchers)

- **PTS (Points):** how many points a player scores per game  
- **REB (Rebounds):** how many missed shots a player grabs (possession control)  
- **AST (Assists):** passes that directly lead to a made shot  
- **TS% (True Shooting %):** scoring efficiency (includes free throws)  
- **Usage:** how involved a player is in the offense (more shots/turnovers = more usage)  
- **PER (demo):** a custom “overall impact” score built from box score stats  
  - ⚠️ This is **not the official NBA PER formula** — it’s a demo metric for this project.

---

## Data source

This app expects a CSV file.


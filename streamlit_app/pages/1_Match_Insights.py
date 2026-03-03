"""
streamlit_app/pages/1_📊_Match_Insights.py
-------------------------------------------
Per-match deep dive: scorecard, run rate progression, top performers.
"""

import os, sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

st.set_page_config(page_title="Match Insights", page_icon="📊", layout="wide")
st.title("📊 Match Insights — Ball-by-Ball Analysis")
st.caption("Drill into any IPL 2022 fixture for a deep-dive scorecard, run rate charts, and player impact.")


@st.cache_data
def load_data():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    return balls, matches


balls, matches = load_data()

# Match selector
match_label = matches.apply(
    lambda r: f"#{r['id']} — {r.get('team1','?')} vs {r.get('team2','?')} @ {r.get('venue','?')}", axis=1
)
selected_label = st.selectbox("Select a Match", match_label.tolist())
match_id = int(selected_label.split("—")[0].replace("#", "").strip())

match_info = matches[matches["id"] == match_id].iloc[0]
match_balls = balls[balls["id"] == match_id].copy()
match_balls["delivery_num"] = match_balls.groupby("innings").cumcount() + 1
match_balls["over_num"] = ((match_balls["delivery_num"] - 1) // 6) + 1
match_balls["cumulative_runs"] = match_balls.groupby("innings")["batsman_run"].cumsum()

# Header
c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    st.metric("🏏 Team 1", match_info.get("team1", "N/A"))
with c2:
    st.markdown("### VS")
with c3:
    st.metric("🏏 Team 2", match_info.get("team2", "N/A"))

winner_col = "winningteam"
if winner_col in match_info:
    st.success(f"🏆 **Winner:** {match_info[winner_col]}")
if "venue" in match_info:
    st.caption(f"📍 Venue: {match_info['venue']}")

st.divider()

# Scorecard
bat_col = [c for c in match_balls.columns if "battingteam" in c or "batting_team" in c]
if bat_col:
    innings_tabs = st.tabs([f"Innings: {t}" for t in match_balls[bat_col[0]].unique()])
    for i, team in enumerate(match_balls[bat_col[0]].unique()):
        with innings_tabs[i]:
            tb = match_balls[match_balls[bat_col[0]] == team]
            total_runs = int(tb["batsman_run"].sum() + (tb["total_run"].sum() - tb["batsman_run"].sum()))
            wickets = int(tb["iswicketdelivery"].sum())
            legal = len(tb[tb.get("extra_type", pd.Series(dtype=str)).isna()]) if "extra_type" in tb.columns else len(tb)
            overs = f"{legal // 6}.{legal % 6}"

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Runs", total_runs)
            m2.metric("Wickets", wickets)
            m3.metric("Overs", overs)

            # Batter scorecard
            batter_sc = tb.groupby("batter").agg(
                Runs=("batsman_run", "sum"),
                Balls=("batsman_run", "count"),
                Fours=("batsman_run", lambda x: (x == 4).sum()),
                Sixes=("batsman_run", lambda x: (x == 6).sum()),
            ).reset_index()
            batter_sc["SR"] = (batter_sc["Runs"] / batter_sc["Balls"] * 100).round(1)
            batter_sc = batter_sc.sort_values("Runs", ascending=False)
            st.dataframe(batter_sc, use_container_width=True, hide_index=True)

# Run Rate Progression
st.markdown("### 📈 Run Rate Progression")
innings_list = match_balls["innings"].unique() if "innings" in match_balls.columns else []
fig = go.Figure()
colors = ["#e94560", "#0f3460"]
for idx, inn in enumerate(innings_list):
    inn_data = match_balls[match_balls["innings"] == inn].copy()
    over_runs = inn_data.groupby("over_num")["batsman_run"].sum().reset_index()
    over_runs["cumulative"] = over_runs["batsman_run"].cumsum()
    fig.add_trace(go.Scatter(
        x=over_runs["over_num"], y=over_runs["cumulative"],
        mode="lines+markers", name=f"Innings {inn}",
        line=dict(color=colors[idx % 2], width=2.5),
        fill="tozeroy", fillcolor=f"rgba({','.join(str(int(c,16)) for c in [colors[idx%2][1:3],colors[idx%2][3:5],colors[idx%2][5:7]])},.1)" if len(colors[idx%2])==7 else None
    ))
fig.update_layout(
    template="plotly_dark", height=350,
    xaxis_title="Over", yaxis_title="Cumulative Runs",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# Top performers
st.markdown("### 🌟 Match Top Performers")
p1, p2 = st.columns(2)
with p1:
    top_bat = match_balls.groupby("batter")["batsman_run"].sum().sort_values(ascending=False).head(5)
    fig_b = px.bar(top_bat.reset_index(), x="batter", y="batsman_run",
                   title="Top Batters", template="plotly_dark",
                   color="batsman_run", color_continuous_scale="Reds",
                   labels={"batter": "Player", "batsman_run": "Runs"})
    fig_b.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_b, use_container_width=True)
with p2:
    wkts = match_balls[match_balls["iswicketdelivery"] == 1]
    top_bowl = wkts.groupby("bowler")["iswicketdelivery"].count().sort_values(ascending=False).head(5)
    fig_w = px.bar(top_bowl.reset_index(), x="bowler", y="iswicketdelivery",
                   title="Top Bowlers", template="plotly_dark",
                   color="iswicketdelivery", color_continuous_scale="Blues",
                   labels={"bowler": "Player", "iswicketdelivery": "Wickets"})
    fig_w.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_w, use_container_width=True)

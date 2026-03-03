"""
streamlit_app/pages/3_📈_Player_Analytics.py
---------------------------------------------
Player deep-dive: career stats, radar chart, phase analysis, head-to-head comparison.
"""

import os, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

st.set_page_config(page_title="Player Analytics", page_icon="📈", layout="wide")
st.title("📈 Player Analytics")
st.caption("Career profile, form tracker, phase breakdown, and head-to-head comparison.")


@st.cache_data
def load_data():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    balls["delivery_num"] = balls.groupby(["id", "innings"]).cumcount() + 1
    balls["over_num"] = ((balls["delivery_num"] - 1) // 6) + 1
    balls["phase"] = pd.cut(balls["over_num"], bins=[0, 6, 15, 20],
                             labels=["Powerplay", "Middle", "Death"])
    return balls, matches


balls, matches = load_data()
all_players = sorted(balls["batter"].dropna().unique().tolist())

# ── Player Selection ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Player Search")
player = st.sidebar.selectbox("Select Primary Player", all_players)
compare_mode = st.sidebar.checkbox("Enable Head-to-Head Comparison")
player2 = None
if compare_mode:
    player2 = st.sidebar.selectbox("Select Player 2", [p for p in all_players if p != player])


def get_batting_stats(p):
    d = balls[balls["batter"] == p]
    runs = int(d["batsman_run"].sum())
    bf = len(d)
    sr = round(runs / bf * 100, 2) if bf else 0
    fours = int((d["batsman_run"] == 4).sum())
    sixes = int((d["batsman_run"] == 6).sum())
    matches_played = d["id"].nunique()
    avg = round(runs / max(1, d["id"].nunique()), 2)
    return {"Player": p, "Runs": runs, "Balls": bf, "SR": sr,
            "4s": fours, "6s": sixes, "Matches": matches_played, "Avg": avg}


def get_bowling_stats(p):
    d = balls[balls["bowler"] == p]
    wickets = int(d["iswicketdelivery"].sum())
    runs_con = int(d["total_run"].sum())
    lb = len(d)
    eco = round(runs_con / (lb / 6), 2) if lb else 0
    dots = int((d["batsman_run"] == 0).sum())
    return {"Wickets": wickets, "Economy": eco, "Runs Conceded": runs_con, "Dot Balls": dots}


def radar_chart(bat_stats: dict, bowl_stats: dict, player_name: str):
    """Build a normalized radar chart for the player."""
    # Normalize to 0-100 scale with assumed maximums
    maxima = {"Strike Rate": 200, "Boundary %": 40, "Avg Runs/Match": 60,
              "Wickets": 20, "Dot Ball %": 60, "Economy (inv.)": 15}

    bf = max(bat_stats["Balls"], 1)
    boundary_pct = (bat_stats["4s"] + bat_stats["6s"]) / bf * 100
    dot_pct = bowl_stats["Dot Balls"] / bf * 100 if bf else 0
    eco_inv = max(0, 15 - bowl_stats["Economy"])  # lower economy = higher score

    values = [
        min(bat_stats["SR"] / maxima["Strike Rate"] * 100, 100),
        min(boundary_pct / maxima["Boundary %"] * 100, 100),
        min(bat_stats["Avg"] / maxima["Avg Runs/Match"] * 100, 100),
        min(bowl_stats["Wickets"] / maxima["Wickets"] * 100, 100),
        min(dot_pct / maxima["Dot Ball %"] * 100, 100),
        min(eco_inv / maxima["Economy (inv.)"] * 100, 100),
    ]
    categories = ["Strike Rate", "Boundary %", "Avg Runs/Match", "Wickets", "Dot Ball %", "Economy"]
    values += [values[0]]  # close the loop
    categories += [categories[0]]

    fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill="toself",
                                    line=dict(color="#e94560", width=2),
                                    fillcolor="rgba(233,69,96,0.2)"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=False, template="plotly_dark", height=380,
                      title=f"Performance Radar — {player_name}")
    return fig


# ── Main display ───────────────────────────────────────────────────────────────
players_to_show = [player] if not compare_mode else [player, player2]

for p in players_to_show:
    st.markdown(f"## 🏏 {p}")
    bat = get_batting_stats(p)
    bowl = get_bowling_stats(p)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val in zip([k1, k2, k3, k4, k5],
                                ["Runs", "Strike Rate", "Avg", "Wickets", "Economy"],
                                [bat["Runs"], bat["SR"], bat["Avg"],
                                 bowl["Wickets"], bowl["Economy"]]):
        col.metric(label, val)

    col_radar, col_phase = st.columns([1, 1])

    with col_radar:
        st.plotly_chart(radar_chart(bat, bowl, p), use_container_width=True)

    with col_phase:
        st.markdown("#### 📊 Runs by Phase")
        phase_data = balls[balls["batter"] == p].groupby("phase")["batsman_run"].sum().reset_index()
        phase_data.columns = ["Phase", "Runs"]
        if not phase_data.empty:
            fig_phase = px.bar(phase_data, x="Phase", y="Runs", color="Phase",
                               color_discrete_map={"Powerplay": "#e94560", "Middle": "#0f3460", "Death": "#533483"},
                               template="plotly_dark", height=360)
            fig_phase.update_layout(showlegend=False)
            st.plotly_chart(fig_phase, use_container_width=True)
        else:
            st.info("No batting data for phase breakdown.")

    # Match-by-match form tracker
    st.markdown("#### 📉 Form Tracker (Runs per Match)")
    form_data = balls[balls["batter"] == p].groupby("id")["batsman_run"].sum().reset_index()
    form_data.columns = ["Match ID", "Runs"]
    if not form_data.empty:
        form_data["Rolling Avg (5)"] = form_data["Runs"].rolling(5, min_periods=1).mean().round(1)
        fig_form = go.Figure()
        fig_form.add_trace(go.Bar(x=form_data["Match ID"], y=form_data["Runs"],
                                  name="Runs", marker_color="#0f3460"))
        fig_form.add_trace(go.Scatter(x=form_data["Match ID"], y=form_data["Rolling Avg (5)"],
                                      name="5-match rolling avg", line=dict(color="#e94560", width=2.5)))
        fig_form.update_layout(template="plotly_dark", height=300, xaxis_title="Match ID",
                                yaxis_title="Runs", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_form, use_container_width=True)

    st.divider()

# Head-to-head comparison table
if compare_mode and player2:
    st.markdown("## ⚖️ Head-to-Head Comparison")
    bat1, bat2 = get_batting_stats(player), get_batting_stats(player2)
    bowl1, bowl2 = get_bowling_stats(player), get_bowling_stats(player2)

    comparison = pd.DataFrame({
        "Metric": ["Runs", "Balls Faced", "Strike Rate", "Avg", "4s", "6s",
                   "Wickets", "Economy", "Dot Balls"],
        player: [bat1["Runs"], bat1["Balls"], bat1["SR"], bat1["Avg"], bat1["4s"], bat1["6s"],
                 bowl1["Wickets"], bowl1["Economy"], bowl1["Dot Balls"]],
        player2: [bat2["Runs"], bat2["Balls"], bat2["SR"], bat2["Avg"], bat2["4s"], bat2["6s"],
                  bowl2["Wickets"], bowl2["Economy"], bowl2["Dot Balls"]],
    })
    st.dataframe(comparison.set_index("Metric"), use_container_width=True)

    fig_comp = px.bar(comparison, x="Metric", y=[player, player2], barmode="group",
                      template="plotly_dark", color_discrete_sequence=["#e94560", "#0f3460"])
    fig_comp.update_layout(height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

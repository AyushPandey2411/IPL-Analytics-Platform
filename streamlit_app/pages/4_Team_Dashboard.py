"""
streamlit_app/pages/4_🏆_Team_Dashboard.py
-------------------------------------------
Team-level performance: phase analysis, toss impact, season form.
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

st.set_page_config(page_title="Team Dashboard", page_icon="🏆", layout="wide")
st.title("🏆 Team Performance Dashboard")
st.caption("Phase-by-phase breakdown, toss analysis, head-to-head records, and win trends.")


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
teams = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))

# ── Team selector ──────────────────────────────────────────────────────────────
selected_team = st.selectbox("🏏 Select Team", teams)
bat_col = [c for c in balls.columns if "battingteam" in c or "batting_team" in c]
team_balls = balls[balls[bat_col[0]] == selected_team] if bat_col else pd.DataFrame()

# ── Season record ──────────────────────────────────────────────────────────────
team_matches = matches[(matches["team1"] == selected_team) | (matches["team2"] == selected_team)]
wins = (team_matches["winningteam"] == selected_team).sum()
played = len(team_matches)
losses = played - wins

m1, m2, m3, m4 = st.columns(4)
m1.metric("Matches Played", played)
m2.metric("Wins", int(wins))
m3.metric("Losses", int(losses))
m4.metric("Win Rate", f"{wins/played*100:.1f}%" if played else "N/A")

st.divider()

tabs = st.tabs(["📊 Phase Analysis", "🪙 Toss Impact", "🆚 Head-to-Head", "📅 Season Timeline"])

with tabs[0]:
    st.markdown("### Batting Performance by Phase")
    if not team_balls.empty:
        phase_stats = team_balls.groupby("phase").agg(
            Runs=("batsman_run", "sum"),
            Wickets=("iswicketdelivery", "sum"),
            Balls=("batsman_run", "count"),
        ).reset_index()
        phase_stats["Run Rate"] = (phase_stats["Runs"] / (phase_stats["Balls"] / 6)).round(2)
        phase_stats["Boundary Count"] = team_balls[team_balls["batsman_run"].isin([4, 6])].groupby("phase").size().values[:3] if len(phase_stats) >= 1 else 0

        c_r, c_w = st.columns(2)
        with c_r:
            fig = px.bar(phase_stats, x="phase", y="Runs", color="phase",
                         color_discrete_map={"Powerplay": "#e94560", "Middle": "#0f3460", "Death": "#533483"},
                         title=f"{selected_team} — Runs by Phase", template="plotly_dark")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c_w:
            fig2 = px.bar(phase_stats, x="phase", y="Wickets", color="phase",
                          color_discrete_map={"Powerplay": "#e94560", "Middle": "#0f3460", "Death": "#533483"},
                          title=f"{selected_team} — Wickets Lost by Phase", template="plotly_dark")
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(phase_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No ball-by-ball data available for this team.")

with tabs[1]:
    st.markdown("### Toss Impact Analysis")
    toss_col = "tosswinner" if "tosswinner" in matches.columns else None
    if toss_col:
        team_toss = team_matches.copy()
        team_toss["won_toss"] = team_toss[toss_col] == selected_team
        team_toss["won_match"] = team_toss["winningteam"] == selected_team

        toss_analysis = team_toss.groupby("won_toss")["won_match"].agg(["sum", "count"]).reset_index()
        toss_analysis.columns = ["Won Toss", "Match Wins", "Total Matches"]
        toss_analysis["Win %"] = (toss_analysis["Match Wins"] / toss_analysis["Total Matches"] * 100).round(1)
        toss_analysis["Won Toss"] = toss_analysis["Won Toss"].map({True: "Won Toss", False: "Lost Toss"})
        st.dataframe(toss_analysis, use_container_width=True, hide_index=True)

        fig_toss = px.bar(toss_analysis, x="Won Toss", y="Win %",
                          color="Won Toss", text="Win %",
                          template="plotly_dark", title="Win Rate: When Winning vs Losing Toss",
                          color_discrete_sequence=["#e94560", "#0f3460"])
        fig_toss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_toss.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_toss, use_container_width=True)

        # Toss decision breakdown
        if "tossdecision" in matches.columns:
            st.markdown("#### Toss Decision Distribution")
            decision_data = team_toss[team_toss["won_toss"]]["tossdecision"].value_counts().reset_index()
            decision_data.columns = ["Decision", "Count"]
            fig_dec = px.pie(decision_data, names="Decision", values="Count", hole=0.4,
                             template="plotly_dark", color_discrete_sequence=["#e94560", "#0f3460"])
            st.plotly_chart(fig_dec, use_container_width=True)
    else:
        st.info("Toss data not available.")

with tabs[2]:
    st.markdown("### Head-to-Head vs All Teams")
    h2h_rows = []
    for opp in [t for t in teams if t != selected_team]:
        h2h = team_matches[
            ((team_matches["team1"] == selected_team) & (team_matches["team2"] == opp)) |
            ((team_matches["team1"] == opp) & (team_matches["team2"] == selected_team))
        ]
        if not h2h.empty:
            w = (h2h["winningteam"] == selected_team).sum()
            h2h_rows.append({"Opponent": opp, "Played": len(h2h), "Won": int(w),
                              "Lost": int(len(h2h) - w), "Win %": round(w / len(h2h) * 100, 1)})

    if h2h_rows:
        h2h_df = pd.DataFrame(h2h_rows).sort_values("Win %", ascending=False)
        fig_h2h = px.bar(h2h_df, x="Opponent", y="Win %", color="Win %",
                         color_continuous_scale="RdYlGn", template="plotly_dark",
                         title=f"{selected_team} Win % vs Each Opponent", text="Win %")
        fig_h2h.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_h2h.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig_h2h, use_container_width=True)
        st.dataframe(h2h_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown("### Season Match Timeline")
    timeline = team_matches.copy()
    timeline["Result"] = timeline["winningteam"].apply(
        lambda w: "Win" if w == selected_team else "Loss"
    )
    timeline["Match #"] = range(1, len(timeline) + 1)

    fig_t = px.scatter(timeline, x="Match #", y="Result",
                       color="Result", color_discrete_map={"Win": "#27ae60", "Loss": "#e94560"},
                       title=f"{selected_team} — Season Result Timeline", template="plotly_dark",
                       hover_data=["team1", "team2", "venue"] if "venue" in timeline.columns else ["team1", "team2"])
    fig_t.update_traces(marker=dict(size=14))
    fig_t.update_layout(height=300, yaxis_title="")
    st.plotly_chart(fig_t, use_container_width=True)

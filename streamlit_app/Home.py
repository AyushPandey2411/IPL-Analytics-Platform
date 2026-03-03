"""
streamlit_app/🏠_Home.py
-------------------------
Season overview: KPIs, leaderboards, top performers.
Run with: streamlit run streamlit_app/🏠_Home.py
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #e94560;
        border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin: 20px 0 12px 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")

    # Phase engineering
    balls["delivery_num"] = balls.groupby(["id", "innings"]).cumcount() + 1
    balls["over_num"] = ((balls["delivery_num"] - 1) // 6) + 1
    balls["phase"] = pd.cut(balls["over_num"], bins=[0, 6, 15, 20],
                             labels=["Powerplay", "Middle", "Death"])
    return balls, matches


@st.cache_data
def load_player_stats():
    path = os.path.join(DATA_DIR, "player_stats.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


balls, matches = load_data()
player_stats = load_player_stats()

# ── Header ─────────────────────────────────────────────────────────────────────
logo_path = os.path.join(ROOT, "assets", "ipl_image.png")
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with col_title:
    st.markdown("# 🏏 IPL 2022 Analytics Platform")
    st.caption("End-to-end cricket intelligence: ETL pipeline · ML predictions · Fantasy optimization · Executive dashboards")

st.divider()

# ── KPI Metrics ────────────────────────────────────────────────────────────────
total_matches = len(matches)
total_runs = int(balls["batsman_run"].sum())
total_wickets = int(balls["iswicketdelivery"].sum())
total_sixes = int((balls["batsman_run"] == 6).sum())
total_fours = int((balls["batsman_run"] == 4).sum())
unique_players = balls["batter"].nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    (c1, total_matches, "Matches Played"),
    (c2, f"{total_runs:,}", "Total Runs"),
    (c3, total_wickets, "Total Wickets"),
    (c4, total_sixes, "Sixes Hit"),
    (c5, total_fours, "Fours Hit"),
    (c6, unique_players, "Unique Players"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboards", "📊 Team Rankings", "📈 Season Trends", "🗓️ Match Log"])

with tab1:
    c_bat, c_bowl = st.columns(2)

    with c_bat:
        st.markdown('<div class="section-header">Top 10 Run Scorers</div>', unsafe_allow_html=True)
        top_bat = (
            balls.groupby("batter")["batsman_run"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_bat.columns = ["Player", "Runs"]
        fig = px.bar(top_bat, x="Runs", y="Player", orientation="h",
                     color="Runs", color_continuous_scale="Reds",
                     text="Runs", template="plotly_dark")
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"},
                          height=400, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c_bowl:
        st.markdown('<div class="section-header">Top 10 Wicket Takers</div>', unsafe_allow_html=True)
        top_bowl = (
            balls[balls["iswicketdelivery"] == 1].groupby("bowler")["iswicketdelivery"].count()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_bowl.columns = ["Player", "Wickets"]
        fig2 = px.bar(top_bowl, x="Wickets", y="Player", orientation="h",
                      color="Wickets", color_continuous_scale="Blues",
                      text="Wickets", template="plotly_dark")
        fig2.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"},
                           height=400, margin=dict(l=0, r=0, t=0, b=0))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Team Win Table</div>', unsafe_allow_html=True)
    wins = matches["winningteam"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    played_1 = matches["team1"].value_counts()
    played_2 = matches["team2"].value_counts()
    played = (played_1.add(played_2, fill_value=0)).reset_index()
    played.columns = ["Team", "Played"]
    team_table = wins.merge(played, on="Team")
    team_table["Losses"] = team_table["Played"] - team_table["Wins"]
    team_table["Win %"] = (team_table["Wins"] / team_table["Played"] * 100).round(1)
    team_table = team_table.sort_values("Win %", ascending=False)
    st.dataframe(team_table, use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-header">Runs by Phase Across Season</div>', unsafe_allow_html=True)
    phase_runs = balls.groupby("phase")["batsman_run"].sum().reset_index()
    phase_runs.columns = ["Phase", "Runs"]
    fig3 = px.pie(phase_runs, values="Runs", names="Phase", hole=0.45,
                  color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
                  template="plotly_dark")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">All Matches — IPL 2022</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["id", "team1", "team2", "venue", "winningteam"] if c in matches.columns]
    st.dataframe(matches[display_cols], use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("📊 IPL Analytics Platform · Built with Python, Streamlit, FastAPI, XGBoost & Power BI · Ayush Pandey")

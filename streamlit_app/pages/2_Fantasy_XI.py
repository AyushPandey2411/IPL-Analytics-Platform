"""
streamlit_app/pages/2_🤖_Fantasy_XI.py
----------------------------------------
ML-powered Fantasy XI generator with Captain/Vice-Captain picks,
role breakdown, and predicted points visualization.
"""

import os, sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Fantasy XI", page_icon="🤖", layout="wide")
st.title("🤖 ML-Powered Fantasy XI Generator")
st.caption("Select two teams to generate an optimized Fantasy XI with ML-predicted points, role balance, and Captain/VC recommendations.")


@st.cache_data
def load_data():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    return balls, matches


balls_raw, matches = load_data()
teams = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))

# ── Team Selection ─────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    team1 = st.selectbox("🅰️ Select Team 1", teams, index=0)
with c2:
    team2 = st.selectbox("🆚 Select Team 2", [t for t in teams if t != team1], index=0)

generate = st.button("⚡ Generate Fantasy XI", type="primary", use_container_width=True)

if generate:
    with st.spinner("Running ML model..."):

        # ── Build player stats ─────────────────────────────────────────────────
        balls = balls_raw.copy()

        # Batting stats
        batting = balls.groupby("batter").agg(
            runs=("batsman_run", "sum"),
            balls_faced=("batsman_run", "count"),
            fours=("batsman_run", lambda x: (x == 4).sum()),
            sixes=("batsman_run", lambda x: (x == 6).sum()),
            matches_bat=("id", "nunique"),
        ).reset_index().rename(columns={"batter": "player"})
        batting["strike_rate"] = (batting["runs"] / batting["balls_faced"] * 100).round(2)

        # Bowling stats
        bowling = balls[balls["iswicketdelivery"] == 1].groupby("bowler").agg(
            wickets=("iswicketdelivery", "sum"),
        ).reset_index().rename(columns={"bowler": "player"})
        runs_con = balls.groupby("bowler")["total_run"].sum().reset_index()
        runs_con.columns = ["player", "runs_conceded"]
        balls_bowled = balls.groupby("bowler")["total_run"].count().reset_index()
        balls_bowled.columns = ["player", "balls_bowled"]
        dot_balls = balls[balls["batsman_run"] == 0].groupby("bowler").size().reset_index()
        dot_balls.columns = ["player", "dot_balls"]

        bowling = bowling.merge(runs_con, on="player", how="left")\
                         .merge(balls_bowled, on="player", how="left")\
                         .merge(dot_balls, on="player", how="left").fillna(0)
        bowling["economy"] = (bowling["runs_conceded"] / (bowling["balls_bowled"] / 6).replace(0, 1)).round(2)

        # Merge
        df = pd.merge(batting, bowling, on="player", how="outer").fillna(0)

        # Fantasy points (Dream11-style)
        df["fantasy_points"] = (
            df["runs"]
            + df["wickets"] * 25
            + (df["runs"] // 50) * 8
            + (df["runs"] // 100) * 16
            + df["fours"] * 1
            + df["sixes"] * 2
            + df["dot_balls"] * 0.5
        ).round(1)

        # Role classification
        def classify(row):
            is_bat = row["runs"] > 200 or row["strike_rate"] > 130
            is_bowl = row["wickets"] > 5 or (0 < row["economy"] < 8)
            if is_bat and is_bowl:
                return "All-Rounder"
            elif is_bat:
                return "Batsman"
            elif is_bowl:
                return "Bowler"
            return "Batsman"

        df["role"] = df.apply(classify, axis=1)

        # ── ML Scoring ────────────────────────────────────────────────────────
        features = ["runs", "wickets", "balls_faced", "strike_rate", "economy",
                    "fours", "sixes", "dot_balls"]
        X = df[features]
        y = df["fantasy_points"]
        model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X, y)
        df["predicted_points"] = model.predict(X).round(1)

        # ── Player pool filtering ─────────────────────────────────────────────
        pool_matches = matches[
            ((matches["team1"] == team1) & (matches["team2"] == team2)) |
            ((matches["team1"] == team2) & (matches["team2"] == team1))
        ]
        pool = set()
        for col in ["team1players", "team2players"]:
            if col in pool_matches.columns:
                for val in pool_matches[col].dropna():
                    try:
                        pool.update(eval(val))
                    except Exception:
                        pass

        filtered = df[df["player"].isin(pool)] if pool else df
        filtered = filtered[filtered["fantasy_points"] > 0].copy()

        # ── Role-balanced XI ──────────────────────────────────────────────────
        batsmen = filtered[filtered["role"] == "Batsman"].sort_values("predicted_points", ascending=False).head(4)
        bowlers = filtered[filtered["role"] == "Bowler"].sort_values("predicted_points", ascending=False).head(4)
        allr = filtered[filtered["role"] == "All-Rounder"].sort_values("predicted_points", ascending=False).head(3)
        xi = pd.concat([batsmen, bowlers, allr]).sort_values("predicted_points", ascending=False).head(11).reset_index(drop=True)

        if len(xi) < 11:
            extras = filtered[~filtered["player"].isin(xi["player"])].sort_values("predicted_points", ascending=False).head(11 - len(xi))
            xi = pd.concat([xi, extras]).reset_index(drop=True)

        captain = xi.iloc[0]["player"]
        vc = xi.iloc[1]["player"]

    # ── Display ───────────────────────────────────────────────────────────────
    st.success(f"✅ Fantasy XI generated for **{team1}** vs **{team2}**")

    # Captain / VC highlight
    col_c, col_vc = st.columns(2)
    with col_c:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#e94560,#c0392b);padding:16px;border-radius:12px;text-align:center">
            <div style="font-size:1.8rem">⭐</div>
            <div style="font-size:1.1rem;font-weight:700;color:white">CAPTAIN</div>
            <div style="font-size:1.4rem;font-weight:800;color:white">{captain}</div>
            <div style="color:#ffcccc">{xi.iloc[0]['role']} · {xi.iloc[0]['predicted_points']:.0f} pts</div>
        </div>""", unsafe_allow_html=True)
    with col_vc:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f3460,#16213e);padding:16px;border-radius:12px;text-align:center;border:2px solid #e94560">
            <div style="font-size:1.8rem">🌟</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e94560">VICE CAPTAIN</div>
            <div style="font-size:1.4rem;font-weight:800;color:white">{vc}</div>
            <div style="color:#a0aec0">{xi.iloc[1]['role']} · {xi.iloc[1]['predicted_points']:.0f} pts</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🏏 Your Fantasy XI")
    # Player cards grid
    for i in range(0, min(len(xi), 11), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(xi):
                row = xi.iloc[idx]
                badge = "⭐ C" if row["player"] == captain else ("🌟 VC" if row["player"] == vc else "")
                role_colors = {"Batsman": "#e94560", "Bowler": "#0f3460", "All-Rounder": "#533483"}
                color = role_colors.get(row["role"], "#333")
                with cols[j]:
                    st.markdown(f"""
                    <div style="border:1px solid #333;border-radius:10px;padding:14px;margin:4px;background:#1a1a2e">
                        <div style="font-size:1rem;font-weight:700;color:white">{row['player']} {badge}</div>
                        <div style="color:{color};font-size:0.8rem;margin:4px 0">{row['role']}</div>
                        <div style="color:#a0aec0;font-size:0.85rem">🏏 {int(row['runs'])} runs · 🎯 {int(row['wickets'])} wkts</div>
                        <div style="color:#e94560;font-weight:700;margin-top:6px">{row['predicted_points']:.0f} pts predicted</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Predicted Points Breakdown")
    fig = px.bar(xi, x="player", y="predicted_points", color="role",
                 color_discrete_map={"Batsman": "#e94560", "Bowler": "#0f3460", "All-Rounder": "#533483"},
                 labels={"player": "Player", "predicted_points": "Predicted Fantasy Points"},
                 template="plotly_dark")
    fig.update_layout(xaxis_tickangle=-35, height=350, legend_title="Role")
    st.plotly_chart(fig, use_container_width=True)

    # Role distribution
    role_dist = xi["role"].value_counts().reset_index()
    role_dist.columns = ["Role", "Count"]
    fig2 = px.pie(role_dist, names="Role", values="Count", hole=0.4,
                  color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
                  template="plotly_dark", title="Role Balance")
    st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.markdown("### 📋 Full Fantasy XI Data")
    st.dataframe(
        xi[["player", "role", "runs", "wickets", "strike_rate", "economy", "predicted_points"]]
        .rename(columns={"player": "Player", "role": "Role", "runs": "Runs",
                         "wickets": "Wkts", "strike_rate": "SR", "economy": "Eco",
                         "predicted_points": "Pred. Points"}),
        use_container_width=True, hide_index=True
    )

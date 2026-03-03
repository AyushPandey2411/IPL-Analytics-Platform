"""
streamlit_app/pages/5_🔮_Win_Predictor.py
------------------------------------------
XGBoost-powered match outcome predictor with win probability gauge.
"""

import os, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import joblib

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Win Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Match Win Probability Predictor")
st.caption("Powered by XGBoost · Trained on IPL 2022 match data")


@st.cache_data
def load_matches():
    df = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


matches = load_matches()
teams = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
venues = sorted(matches["venue"].dropna().unique().tolist()) if "venue" in matches.columns else []
toss_decisions = ["bat", "field"]


@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, "win_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


model_artifact = load_model()

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("🏏 Team 1", teams, index=0)
        venue = st.selectbox("📍 Venue", venues) if venues else st.text_input("Venue")
    with c2:
        team2 = st.selectbox("🆚 Team 2", [t for t in teams if t != team1], index=0)
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
        toss_decision = st.selectbox("📋 Toss Decision", toss_decisions)

    submitted = st.form_submit_button("⚡ Predict Match Outcome", use_container_width=True, type="primary")

if submitted:
    if model_artifact is None:
        st.warning("⚠️ Win model not trained yet. Run `python ml/pipeline.py` first to train the model.")
        st.info("Showing a demo prediction based on historical win rates instead...")

        # Fallback: use historical win rate
        t1_wins = (matches["winningteam"] == team1).sum()
        t2_wins = (matches["winningteam"] == team2).sum()
        total = t1_wins + t2_wins
        t1_prob = round(t1_wins / total * 100, 1) if total else 50.0
        t2_prob = round(100 - t1_prob, 1)
        predicted_winner = team1 if t1_prob >= 50 else team2
        confidence = "Medium (based on historical win rate)"
    else:
        model = model_artifact["model"]
        le = model_artifact["encoders"]

        def safe_encode(encoder, value):
            classes = list(encoder.classes_)
            return int(encoder.transform([value])[0]) if value in classes else 0

        row = np.array([[
            safe_encode(le["team1"], team1),
            safe_encode(le["team2"], team2),
            safe_encode(le["venue"], venue),
            safe_encode(le["tosswinner"], toss_winner),
            safe_encode(le["tossdecision"], toss_decision),
        ]])
        proba = model.predict_proba(row)[0]
        t1_prob = round(float(proba[1]) * 100, 1)
        t2_prob = round(100 - t1_prob, 1)
        predicted_winner = team1 if t1_prob >= 50 else team2
        gap = abs(t1_prob - 50)
        confidence = "High" if gap > 15 else "Medium" if gap > 5 else "Low"

    # ── Results display ────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 🏆 Predicted Winner: **{predicted_winner}**")

    c_g1, c_g2, c_g3 = st.columns([2, 1, 2])
    with c_g1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number", value=t1_prob,
            title={"text": team1, "font": {"color": "white", "size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e94560"},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 40], "color": "#16213e"},
                    {"range": [40, 60], "color": "#0f3460"},
                    {"range": [60, 100], "color": "#1a1a2e"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig1.update_layout(template="plotly_dark", height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with c_g2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### VS")
        st.metric("Confidence", confidence)

    with c_g3:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=t2_prob,
            title={"text": team2, "font": {"color": "white", "size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f3460"},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 40], "color": "#16213e"},
                    {"range": [40, 60], "color": "#0f3460"},
                    {"range": [60, 100], "color": "#1a1a2e"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig2.update_layout(template="plotly_dark", height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Head-to-head historical record
    st.markdown("### 📊 Historical Head-to-Head Record")
    h2h = matches[
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    ]
    if not h2h.empty:
        t1_h2h = (h2h["winningteam"] == team1).sum()
        t2_h2h = (h2h["winningteam"] == team2).sum()
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(f"{team1} wins", int(t1_h2h))
        hc2.metric("Matches played", len(h2h))
        hc3.metric(f"{team2} wins", int(t2_h2h))
    else:
        st.info("No head-to-head matches found in 2022 data.")

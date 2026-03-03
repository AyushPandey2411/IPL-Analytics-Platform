"""
streamlit_app/pages/6_Auction_Recommender.py
---------------------------------------------
IPL auction player recommender with ML-estimated prices and value scoring.
"""

import os, sys
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Auction Recommender", page_icon="🛒", layout="wide")
st.title("🛒 IPL Auction Player Recommender")
st.caption("ML-estimated player auction values and budget-aware team building recommendations.")

SKILL_MAP = {
    "BATTER":       ["BAT", "BATSMAN"],
    "BOWLER":       ["BOWL", "BOWLER"],
    "ALL-ROUNDER":  ["ALL", "ALLROUND", "ALL-ROUNDER"],
    "WICKETKEEPER": ["WK", "KEEP", "WICKET"],
}


@st.cache_data
def load_auction_data():
    path = os.path.join(DATA_DIR, "ipl_cleaned_data.xls")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, "ipl_price_predictor.pkl")
    if not os.path.exists(path):
        return None
    try:
        artifact = joblib.load(path)
        return artifact
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


df_raw = load_auction_data()
artifact = load_model()

# ── Guard: prompt retraining if model is broken ────────────────────────────────
if artifact is None or not isinstance(artifact, dict):
    st.error("⚠️ The auction model file is incompatible with your sklearn version.")
    st.info("""
**Fix — run this once in your terminal from the project root:**
```
python retrain_auction_model.py
```
Then refresh this page.
    """)
    st.stop()

if df_raw is None:
    st.error("Auction dataset not found in data/")
    st.stop()

model    = artifact["model"]
le_skill  = artifact["le_skill"]
le_status = artifact["le_status"]

# ── Prep dataframe ─────────────────────────────────────────────────────────────
df = df_raw.copy()
df.columns = df.columns.str.strip()

df["Age"]      = pd.to_numeric(df.get("Age", 28),      errors="coerce").fillna(28)
df["IPL Caps"] = pd.to_numeric(df.get("IPL Caps", 0),  errors="coerce").fillna(0)

def safe_encode(le, val):
    val = str(val).upper().strip()
    if val in le.classes_:
        return int(le.transform([val])[0])
    return 0

df["Skill_enc"]  = df["Skill"].apply(lambda x: safe_encode(le_skill, x))
df["Status_enc"] = df["Player Status"].apply(lambda x: safe_encode(le_status, x)) \
                   if "Player Status" in df.columns else 0

features = ["Age", "Skill_enc", "IPL Caps", "Status_enc"]
df["Estimated Price (₹L)"] = model.predict(df[features]).round(2)

# Base price column (flexible name)
base_col = next((c for c in df.columns if "base" in c.lower() and "price" in c.lower()), None)
if base_col:
    df["Base Price (Lakh)"] = pd.to_numeric(df[base_col], errors="coerce").fillna(0)
    df["Value Score"] = (
        df["IPL Caps"] * 2
        + (df["Estimated Price (₹L)"] - df["Base Price (Lakh)"]).clip(lower=0)
        - df["Age"] * 0.3
    ).round(2)
else:
    df["Base Price (Lakh)"] = 0
    df["Value Score"] = (df["IPL Caps"] * 2 - df["Age"] * 0.3).round(2)

# ── UI ─────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    player_type = st.selectbox("Player Type", list(SKILL_MAP.keys()))
with c2:
    budget = st.slider("Budget (Lakh ₹)", min_value=10, max_value=2000, value=500, step=10)
with c3:
    num_recs = st.slider("Recommendations", 1, 15, 5)

if st.button("🔍 Find Players", type="primary", use_container_width=True):
    keywords = SKILL_MAP[player_type]
    filtered = df[df["Skill"].str.upper().apply(lambda x: any(k in x for k in keywords))].copy()

    if filtered.empty:
        st.warning("No players found for selected type.")
        st.stop()

    within_budget = filtered[filtered["Estimated Price (₹L)"] <= budget]
    if within_budget.empty:
        st.warning(f"No players found under ₹{budget}L. Try increasing budget.")
        st.stop()

    top = within_budget.sort_values("Value Score", ascending=False).head(num_recs)
    st.success(f"Found {len(top)} {player_type.lower()}(s) within ₹{budget}L budget")

    # Player cards
    for i in range(0, len(top), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(top):
                row = top.iloc[idx]
                name = row.get("Player Name", row.get("player_name", "N/A"))
                with cols[j]:
                    st.markdown(f"""
                    <div style="border:1px solid #333;border-radius:10px;padding:14px;background:#1a1a2e">
                        <div style="font-size:1rem;font-weight:700;color:white">{name}</div>
                        <div style="color:#a0aec0;font-size:0.8rem">{row.get('Skill','N/A')} · Age {int(row.get('Age',0))}</div>
                        <div style="margin-top:8px">
                            <span style="color:#a0aec0">IPL Caps:</span>
                            <strong style="color:white">{int(row.get('IPL Caps',0))}</strong><br>
                            <span style="color:#a0aec0">Base Price:</span>
                            <strong style="color:white">₹{row.get('Base Price (Lakh)',0):.0f}L</strong><br>
                            <span style="color:#e94560;font-weight:700">
                                Est. Price: ₹{row.get('Estimated Price (₹L)',0):.1f}L
                            </span>
                        </div>
                        <div style="margin-top:6px;color:#27ae60;font-size:0.85rem">
                            Value Score: {row.get('Value Score',0):.1f}
                        </div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Value Score vs Estimated Price")
    name_col = "Player Name" if "Player Name" in top.columns else top.columns[0]
    fig = px.scatter(top, x="Estimated Price (₹L)", y="Value Score",
                     text=name_col, color="Value Score",
                     color_continuous_scale="RdYlGn", template="plotly_dark")
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Full Table")
    display_cols = [c for c in [name_col, "Skill", "Age", "IPL Caps",
                                 "Base Price (Lakh)", "Estimated Price (₹L)", "Value Score"]
                    if c in top.columns]
    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

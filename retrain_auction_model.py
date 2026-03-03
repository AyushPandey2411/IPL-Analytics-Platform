"""
retrain_auction_model.py
Run this once from project root:
    python retrain_auction_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
MODEL_DIR = "models"

print("Loading auction data...")
path = os.path.join(DATA_DIR, "ipl_cleaned_data.xls")

try:
    df = pd.read_csv(path)
except Exception:
    df = pd.read_excel(path)

print(f"Columns found: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# ── Clean & prep ───────────────────────────────────────────────────────────────
df = df.copy()
df.columns = df.columns.str.strip()

# Find the price column (target)
price_col = None
for col in df.columns:
    if "price" in col.lower() and "base" not in col.lower():
        price_col = col
        break
if price_col is None:
    for col in df.columns:
        if "price" in col.lower():
            price_col = col
            break

print(f"Target column: {price_col}")

# Drop rows where target is missing
df = df.dropna(subset=[price_col])

# Encode categorical columns
le_skill = LabelEncoder()
le_status = LabelEncoder()

df["Skill_enc"] = le_skill.fit_transform(df["Skill"].astype(str).str.upper().str.strip())
if "Player Status" in df.columns:
    df["Status_enc"] = le_status.fit_transform(df["Player Status"].astype(str).str.strip())
else:
    df["Status_enc"] = 0

df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median() if "Age" in df.columns else 28)
df["IPL Caps"] = pd.to_numeric(df["IPL Caps"], errors="coerce").fillna(0)

features = ["Age", "Skill_enc", "IPL Caps", "Status_enc"]
X = df[features]
y = pd.to_numeric(df[price_col], errors="coerce").fillna(0)

print(f"Training on {len(X)} samples...")

# ── Train ──────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42)
model.fit(X, y)

# ── Save ───────────────────────────────────────────────────────────────────────
artifact = {
    "model": model,
    "le_skill": le_skill,
    "le_status": le_status,
    "features": features,
    "price_col": price_col,
}
out_path = os.path.join(MODEL_DIR, "ipl_price_predictor.pkl")
joblib.dump(artifact, out_path)
print(f"✅ Model saved to {out_path}")
print("Now restart the Streamlit app.")

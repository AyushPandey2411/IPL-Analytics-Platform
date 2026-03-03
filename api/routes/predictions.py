"""
api/routes/predictions.py
--------------------------
ML prediction endpoints:
    POST /api/predict/match   — Win probability
    POST /api/fantasy/xi      — Fantasy XI recommendation
    GET  /api/auction/recommend — Auction value recommendations
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import MatchPredictRequest, FantasyXIRequest, AuctionRequest

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ── Win Probability ───────────────────────────────────────────────────────────

@router.post("/predict/match", summary="Predict match win probability")
def predict_match(req: MatchPredictRequest):
    """
    Given two teams, venue, toss winner, and toss decision,
    returns win probability for each team using the trained XGBoost model.
    """
    model_path = os.path.join(MODEL_DIR, "win_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail="Win model not trained. Run ml/pipeline.py first.",
        )

    artifact = joblib.load(model_path)
    model = artifact["model"]
    le = artifact["encoders"]

    # Encode inputs — handle unseen labels gracefully
    def safe_encode(encoder, value):
        classes = list(encoder.classes_)
        if value not in classes:
            # Use closest known label or default to index 0
            return 0
        return int(encoder.transform([value])[0])

    try:
        row = np.array([[
            safe_encode(le["team1"], req.team1),
            safe_encode(le["team2"], req.team2),
            safe_encode(le["venue"], req.venue),
            safe_encode(le["tosswinner"], req.toss_winner),
            safe_encode(le["tossdecision"], req.toss_decision),
        ]])

        proba = model.predict_proba(row)[0]
        team1_prob = round(float(proba[1]) * 100, 1)
        team2_prob = round(100 - team1_prob, 1)
        winner = req.team1 if team1_prob >= 50 else req.team2
        confidence = "High" if abs(team1_prob - 50) > 15 else "Medium" if abs(team1_prob - 50) > 5 else "Low"

        return {
            "team1": req.team1,
            "team2": req.team2,
            "team1_win_probability": team1_prob,
            "team2_win_probability": team2_prob,
            "predicted_winner": winner,
            "confidence": confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── Fantasy XI ────────────────────────────────────────────────────────────────

@router.post("/fantasy/xi", summary="Generate ML-powered Fantasy XI")
def generate_fantasy_xi(req: FantasyXIRequest):
    """
    Generate an optimized Fantasy XI with Captain and Vice-Captain
    for a given team matchup, powered by the trained Random Forest model.
    """
    try:
        from ml.fantasy_recommender import recommend_fantasy_xi
        result = recommend_fantasy_xi(req.team1, req.team2)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fantasy XI error: {str(e)}")


# ── Auction Recommender ───────────────────────────────────────────────────────

SKILL_MAP = {
    "BATTER": ["BAT", "BATSMAN"],
    "BOWLER": ["BOWL", "BOWLER"],
    "ALLROUNDER": ["ALL", "ALLROUND", "ALL-ROUNDER"],
    "WICKETKEEPER": ["WK", "KEEP", "WICKET"],
}


@router.post("/auction/recommend", summary="Budget-aware auction player recommendations")
def auction_recommend(req: AuctionRequest):
    """
    Given a player type and budget, recommend value-for-money players
    using the trained price predictor model.
    """
    data_path = os.path.join(DATA_DIR, "ipl_cleaned_data.xls")
    model_path = os.path.join(MODEL_DIR, "ipl_price_predictor.pkl")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Auction data/model not available.")

    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_excel(data_path)
    model = joblib.load(model_path)

    keywords = SKILL_MAP.get(req.player_type.upper(), [])
    filtered = df[df["Skill"].str.upper().apply(lambda x: any(k in x for k in keywords))]

    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No players found for type '{req.player_type}'.")

    try:
        features = filtered[["Age", "Skill", "IPL Caps", "Player Status"]]
        filtered = filtered.copy()
        filtered["estimated_price_lakh"] = model.predict(features).round(2)
        filtered["player_score"] = (
            filtered["IPL Caps"].fillna(0) * 2
            + (filtered["estimated_price_lakh"] - filtered["Base Price (Lakh)"]).clip(lower=0)
            - filtered["Age"].fillna(30) * 0.3
        ).round(2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    within_budget = filtered[filtered["estimated_price_lakh"] <= req.budget_lakh]
    if within_budget.empty:
        raise HTTPException(status_code=404, detail="No players found within the given budget.")

    top = within_budget.sort_values("player_score", ascending=False).head(req.num_recommendations)
    return top[["Player Name", "Skill", "Age", "IPL Caps",
                "Base Price (Lakh)", "estimated_price_lakh", "player_score"]].to_dict(orient="records")

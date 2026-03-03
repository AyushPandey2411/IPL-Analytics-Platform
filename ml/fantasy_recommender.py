"""
ml/fantasy_recommender.py
--------------------------
ML-powered Fantasy XI selector.
Given two teams, loads player stats, scores them with the trained model,
applies role-balance constraints, and returns an optimal 11 with
Captain / Vice-Captain recommendations.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ROLE_SLOTS = {"Batsman": 4, "Bowler": 4, "All-Rounder": 2, "Wicketkeeper": 1}
FALLBACK_SLOTS = {"Batsman": 5, "Bowler": 4, "All-Rounder": 2}


def load_player_stats() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "player_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("player_stats.csv not found. Run ml/pipeline.py first.")
    return pd.read_csv(path)


def load_fantasy_model():
    path = os.path.join(MODEL_DIR, "fantasy_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("fantasy_model.pkl not found. Run ml/pipeline.py first.")
    return joblib.load(path)


def get_player_pool(matches_df: pd.DataFrame, team1: str, team2: str) -> set:
    """Extract all players who played in any fixture between the two teams."""
    relevant = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ]
    pool = set()
    for col in ["team1players", "team2players"]:
        if col in relevant.columns:
            for val in relevant[col].dropna():
                try:
                    pool.update(eval(val))
                except Exception:
                    pass
    return pool


def recommend_fantasy_xi(team1: str, team2: str) -> dict:
    """
    Generate an ML-optimized Fantasy XI for a given matchup.

    Returns:
        dict with keys: xi (list of player dicts), captain, vice_captain
    """
    stats = load_player_stats()
    artifact = load_fantasy_model()
    model = artifact["model"]
    features = artifact["features"]

    # Load matches for player pool filtering
    matches_path = os.path.join(DATA_DIR, "IPL_Matches_2022.csv")
    matches_df = pd.read_csv(matches_path)
    matches_df.columns = matches_df.columns.str.strip().str.lower().str.replace(" ", "_")

    pool = get_player_pool(matches_df, team1, team2)
    if pool:
        filtered = stats[stats["player"].isin(pool)].copy()
    else:
        # Fallback: use all players (no squad data in CSV)
        filtered = stats.copy()

    if filtered.empty:
        raise ValueError(f"No player data found for {team1} vs {team2}.")

    # Fill missing features with 0
    for f in features:
        if f not in filtered.columns:
            filtered[f] = 0

    filtered["predicted_points"] = model.predict(filtered[features])

    # Role-balanced selection
    xi_players = []
    roles_used = {r: 0 for r in FALLBACK_SLOTS}

    for role, slots in FALLBACK_SLOTS.items():
        candidates = filtered[filtered["role"] == role].sort_values(
            "predicted_points", ascending=False
        )
        selected = candidates.head(slots)
        xi_players.append(selected)
        roles_used[role] = len(selected)

    xi = pd.concat(xi_players).sort_values("predicted_points", ascending=False).head(11)

    # If fewer than 11, fill remainder with top-scoring remaining players
    if len(xi) < 11:
        already_selected = set(xi["player"].tolist())
        remainder = filtered[~filtered["player"].isin(already_selected)].sort_values(
            "predicted_points", ascending=False
        ).head(11 - len(xi))
        xi = pd.concat([xi, remainder]).reset_index(drop=True)

    xi = xi.reset_index(drop=True)

    captain = xi.iloc[0]["player"]
    vice_captain = xi.iloc[1]["player"]

    output = {
        "team1": team1,
        "team2": team2,
        "xi": xi[["player", "role", "runs", "wickets", "predicted_points"]].to_dict(orient="records"),
        "captain": captain,
        "vice_captain": vice_captain,
    }
    return output


if __name__ == "__main__":
    # Quick test
    result = recommend_fantasy_xi("Chennai Super Kings", "Mumbai Indians")
    print("Fantasy XI:")
    for p in result["xi"]:
        print(f"  {p['player']} ({p['role']}) — {p['predicted_points']:.1f} pts")
    print(f"Captain: {result['captain']}")
    print(f"Vice-Captain: {result['vice_captain']}")

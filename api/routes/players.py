"""
api/routes/players.py
----------------------
Player-level endpoints:
    GET /api/players/              — list all players
    GET /api/players/{name}        — full player profile
    GET /api/players/{name}/form   — rolling form (last N matches)
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_stats() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "player_stats.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail="Player stats not available. Run ml/pipeline.py first.",
        )
    return pd.read_csv(path)


@router.get("/", summary="List all players")
def list_players(
    role: str = Query(None, description="Filter by role: Batsman, Bowler, All-Rounder"),
    limit: int = Query(50, ge=1, le=500),
):
    """Return a paginated list of all players, optionally filtered by role."""
    df = _load_stats()
    if role:
        df = df[df["role"].str.lower() == role.lower()]
    df = df.sort_values("fantasy_points", ascending=False).head(limit)
    return df[["player", "role", "runs", "wickets", "fantasy_points", "pii"]].to_dict(orient="records")


@router.get("/{name}", summary="Get full player profile")
def get_player(name: str):
    """Return comprehensive stats for a single player (case-insensitive partial match)."""
    df = _load_stats()
    matches = df[df["player"].str.lower().str.contains(name.lower(), na=False)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")
    return matches.fillna(0).to_dict(orient="records")


@router.get("/{name}/phase-stats", summary="Player performance by game phase")
def player_phase_stats(name: str):
    """Return a player's batting/bowling stats split by Powerplay, Middle, and Death overs."""
    balls_path = os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv")
    if not os.path.exists(balls_path):
        raise HTTPException(status_code=503, detail="Ball-by-ball data not available.")

    balls = pd.read_csv(balls_path)
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")

    # Reconstruct phase
    balls["delivery_num"] = balls.groupby(["id", "innings"]).cumcount() + 1
    balls["over_num"] = ((balls["delivery_num"] - 1) // 6) + 1
    balls["phase"] = pd.cut(balls["over_num"], bins=[0, 6, 15, 20],
                            labels=["powerplay", "middle", "death"])

    # Batting phase breakdown
    bat_data = balls[balls["batter"].str.lower().str.contains(name.lower(), na=False)]
    if bat_data.empty:
        raise HTTPException(status_code=404, detail=f"No batting data for '{name}'.")

    phase_bat = bat_data.groupby("phase").agg(
        runs=("batsman_run", "sum"),
        balls=("batsman_run", "count"),
    ).reset_index()
    phase_bat["strike_rate"] = (phase_bat["runs"] / phase_bat["balls"] * 100).round(2)

    return {"player": name, "batting_by_phase": phase_bat.to_dict(orient="records")}

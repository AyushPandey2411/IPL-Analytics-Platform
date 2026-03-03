"""
api/routes/teams.py
--------------------
Team-level analytics endpoints.
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    return balls, matches


@router.get("/", summary="List all teams")
def list_teams():
    _, matches = _load()
    teams = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
    return {"teams": teams}


@router.get("/{team}/form", summary="Team win/loss form for the season")
def team_form(team: str):
    _, matches = _load()
    team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
    if team_matches.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found.")
    wins = (team_matches["winningteam"] == team).sum()
    played = len(team_matches)
    return {
        "team": team,
        "matches_played": int(played),
        "wins": int(wins),
        "losses": int(played - wins),
        "win_pct": round(wins / played * 100, 1) if played else 0,
    }


@router.get("/{team}/phase-analysis", summary="Runs and wickets by game phase")
def phase_analysis(team: str):
    balls, _ = _load()
    balls["delivery_num"] = balls.groupby(["id", "innings"]).cumcount() + 1
    balls["over_num"] = ((balls["delivery_num"] - 1) // 6) + 1

    def phase(o):
        if o <= 6:
            return "powerplay"
        elif o <= 15:
            return "middle"
        return "death"

    balls["phase"] = balls["over_num"].apply(phase)
    team_balls = balls[balls.get("battingteam", balls.get("batting_team", pd.Series(dtype=str))) == team]
    if team_balls.empty:
        # Try with original column name
        col = [c for c in balls.columns if "batting" in c]
        if col:
            team_balls = balls[balls[col[0]] == team]

    if team_balls.empty:
        raise HTTPException(status_code=404, detail=f"No ball-by-ball data for team '{team}'.")

    phase_stats = team_balls.groupby("phase").agg(
        runs=("batsman_run", "sum"),
        wickets=("iswicketdelivery", "sum"),
        balls=("batsman_run", "count"),
    ).reset_index()
    phase_stats["run_rate"] = (phase_stats["runs"] / (phase_stats["balls"] / 6)).round(2)
    return {"team": team, "phase_breakdown": phase_stats.to_dict(orient="records")}

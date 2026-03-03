"""
api/routes/matches.py
----------------------
Match-level endpoints.
"""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load():
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    balls.columns = balls.columns.str.strip().str.lower().str.replace(" ", "_")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    return balls, matches


@router.get("/", summary="List all matches")
def list_matches(limit: int = 20):
    _, matches = _load()
    cols = [c for c in ["id", "team1", "team2", "venue", "winningteam", "date"] if c in matches.columns]
    return matches[cols].head(limit).to_dict(orient="records")


@router.get("/{match_id}", summary="Scorecard for a specific match")
def match_detail(match_id: int):
    balls, matches = _load()
    m = matches[matches["id"] == match_id]
    if m.empty:
        raise HTTPException(status_code=404, detail=f"Match ID {match_id} not found.")

    match_balls = balls[balls["id"] == match_id]

    # Build innings summaries
    innings = []
    bat_col = [c for c in match_balls.columns if "battingteam" in c or "batting_team" in c]
    if bat_col:
        for team in match_balls[bat_col[0]].unique():
            tb = match_balls[match_balls[bat_col[0]] == team]
            innings.append({
                "team": team,
                "runs": int(tb["batsman_run"].sum()),
                "extras": int(tb["total_run"].sum() - tb["batsman_run"].sum()),
                "wickets": int(tb["iswicketdelivery"].sum()),
                "balls": len(tb),
            })

    return {
        "match_id": match_id,
        "team1": m.iloc[0].get("team1"),
        "team2": m.iloc[0].get("team2"),
        "venue": m.iloc[0].get("venue"),
        "winner": m.iloc[0].get("winningteam"),
        "innings": innings,
    }


@router.get("/{match_id}/top-performers", summary="Key players from a match")
def top_performers(match_id: int):
    balls, matches = _load()
    if not (balls["id"] == match_id).any():
        raise HTTPException(status_code=404, detail=f"Match ID {match_id} not found.")

    mb = balls[balls["id"] == match_id]

    # Top batter
    top_bat = mb.groupby("batter")["batsman_run"].sum().idxmax()
    top_bat_runs = int(mb.groupby("batter")["batsman_run"].sum().max())

    # Top bowler
    wkt = mb[mb["iswicketdelivery"] == 1]
    if not wkt.empty:
        top_bowl = wkt.groupby("bowler")["iswicketdelivery"].count().idxmax()
        top_bowl_wkts = int(wkt.groupby("bowler")["iswicketdelivery"].count().max())
    else:
        top_bowl, top_bowl_wkts = "N/A", 0

    return {
        "match_id": match_id,
        "top_batter": {"player": top_bat, "runs": top_bat_runs},
        "top_bowler": {"player": top_bowl, "wickets": top_bowl_wkts},
    }

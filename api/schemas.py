"""
api/schemas.py
--------------
Pydantic models for request validation and response serialization.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────────────────────

class MatchPredictRequest(BaseModel):
    team1: str = Field(..., example="Chennai Super Kings")
    team2: str = Field(..., example="Mumbai Indians")
    venue: str = Field(..., example="Wankhede Stadium")
    toss_winner: str = Field(..., example="Mumbai Indians")
    toss_decision: str = Field(..., example="bat", pattern="^(bat|field)$")


class FantasyXIRequest(BaseModel):
    team1: str = Field(..., example="Chennai Super Kings")
    team2: str = Field(..., example="Mumbai Indians")


class AuctionRequest(BaseModel):
    player_type: str = Field(..., example="BATTER")
    budget_lakh: float = Field(..., example=500.0, ge=10)
    num_recommendations: int = Field(default=5, ge=1, le=10)


# ── Response Models ────────────────────────────────────────────────────────────

class PlayerStatsResponse(BaseModel):
    player: str
    runs: float
    balls_faced: float
    strike_rate: float
    fours: float
    sixes: float
    wickets: float
    economy: float
    fantasy_points: float
    pii: float
    role: str


class TeamFormResponse(BaseModel):
    team: str
    matches_played: int
    wins: int
    losses: int
    win_pct: float
    avg_score: float
    top_scorer: str
    top_wicket_taker: str


class WinProbabilityResponse(BaseModel):
    team1: str
    team2: str
    team1_win_probability: float
    team2_win_probability: float
    predicted_winner: str
    confidence: str


class PlayerCardResponse(BaseModel):
    player: str
    role: str
    runs: float
    wickets: float
    predicted_points: float


class FantasyXIResponse(BaseModel):
    team1: str
    team2: str
    xi: list[PlayerCardResponse]
    captain: str
    vice_captain: str


class PhaseStatsResponse(BaseModel):
    team: str
    powerplay_runs: float
    middle_runs: float
    death_runs: float
    powerplay_wickets: float
    middle_wickets: float
    death_wickets: float

"""
ml/pipeline.py
--------------
End-to-end ETL pipeline: ingestion → cleaning → feature engineering → model training.
Run directly to train and persist all models:
    python ml/pipeline.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. INGESTION ─────────────────────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw ball-by-ball and match-level CSVs."""
    balls = pd.read_csv(os.path.join(DATA_DIR, "IPL_Ball_by_Ball_2022.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "IPL_Matches_2022.csv"))
    print(f"[Ingest] Loaded {len(balls):,} deliveries across {len(matches)} matches.")
    return balls, matches


# ─── 2. CLEANING ──────────────────────────────────────────────────────────────

def clean_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, drop nulls, cast types."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["id", "batter", "bowler", "battingteam"])
    df["batsman_run"] = pd.to_numeric(df["batsman_run"], errors="coerce").fillna(0).astype(int)
    df["total_run"] = pd.to_numeric(df["total_run"], errors="coerce").fillna(0).astype(int)
    df["iswicketdelivery"] = pd.to_numeric(df["iswicketdelivery"], errors="coerce").fillna(0).astype(int)
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize match metadata."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["id", "team1", "team2", "winningteam"])
    return df


# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

def add_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    """Label each delivery with its game phase based on over number."""
    df = df.copy()
    # Reconstruct over number from delivery index within each innings
    df["delivery_num"] = df.groupby(["id", "innings"]).cumcount() + 1
    df["over_num"] = ((df["delivery_num"] - 1) // 6) + 1

    def phase(over):
        if over <= 6:
            return "powerplay"
        elif over <= 15:
            return "middle"
        else:
            return "death"

    df["phase"] = df["over_num"].apply(phase)
    return df


def build_batting_stats(balls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate season-level batting stats per player."""
    g = balls.groupby("batter")
    stats = g.agg(
        runs=("batsman_run", "sum"),
        balls_faced=("batsman_run", "count"),
        fours=("batsman_run", lambda x: (x == 4).sum()),
        sixes=("batsman_run", lambda x: (x == 6).sum()),
        dismissals=("iswicketdelivery", "sum"),
        matches=("id", "nunique"),
    ).reset_index().rename(columns={"batter": "player"})

    stats["strike_rate"] = (stats["runs"] / stats["balls_faced"] * 100).round(2)
    stats["boundary_pct"] = ((stats["fours"] + stats["sixes"]) / stats["balls_faced"] * 100).round(2)
    stats["avg"] = (stats["runs"] / stats["dismissals"].replace(0, 1)).round(2)
    return stats


def build_bowling_stats(balls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate season-level bowling stats per player."""
    wicket_balls = balls[balls["iswicketdelivery"] == 1]
    legal = balls[balls.get("extra_type", pd.Series(dtype=str)).isna()]

    g_all = balls.groupby("bowler")
    g_legal = legal.groupby("bowler") if "extra_type" in balls.columns else g_all
    g_wkt = wicket_balls.groupby("bowler")

    runs_con = g_all["total_run"].sum().rename("runs_conceded")
    legal_balls = g_legal["total_run"].count().rename("legal_balls")
    wickets = g_wkt["iswicketdelivery"].count().rename("wickets")
    dot_balls = balls[balls["batsman_run"] == 0].groupby("bowler").size().rename("dot_balls")

    stats = pd.concat([runs_con, legal_balls, wickets, dot_balls], axis=1).fillna(0).reset_index()
    stats.rename(columns={"bowler": "player"}, inplace=True)
    stats["overs"] = (stats["legal_balls"] / 6).round(1)
    stats["economy"] = (stats["runs_conceded"] / (stats["legal_balls"] / 6).replace(0, 1)).round(2)
    stats["dot_pct"] = (stats["dot_balls"] / stats["legal_balls"].replace(0, 1) * 100).round(2)
    return stats


def build_player_impact(batting: pd.DataFrame, bowling: pd.DataFrame) -> pd.DataFrame:
    """Compute the Player Impact Index (PII) for each player."""
    merged = pd.merge(batting, bowling, on="player", how="outer", suffixes=("_bat", "_bowl")).fillna(0)

    merged["batting_impact"] = (
        merged["runs"] * 0.5
        + merged["strike_rate"] * 0.3
        + (merged["fours"] + merged["sixes"]) * 0.2
    )
    merged["bowling_impact"] = (
        merged["wickets"] * 0.4
        + (1 / merged["economy"].replace(0, 1)) * 40   # scaled
        + merged["dot_pct"] * 0.2
    )
    merged["pii"] = (merged["batting_impact"] * 0.6 + merged["bowling_impact"] * 0.4).round(2)

    # Role classification
    def classify_role(row):
        is_bat = row["runs"] > 200 or row["strike_rate"] > 130
        is_bowl = row["wickets"] > 5 or row["economy"] < 8
        if is_bat and is_bowl:
            return "All-Rounder"
        elif is_bat:
            return "Batsman"
        elif is_bowl:
            return "Bowler"
        else:
            return "Batsman"  # fallback

    merged["role"] = merged.apply(classify_role, axis=1)
    return merged


def build_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Dream11-style fantasy scoring formula."""
    df = df.copy()
    df["fantasy_points"] = (
        df["runs"]
        + df.get("wickets", 0) * 25
        + (df["runs"] // 50) * 8       # 50-run bonus
        + (df["runs"] // 100) * 16     # 100-run bonus
        + df.get("fours", 0) * 1
        + df.get("sixes", 0) * 2
        - df.get("dismissals", 0) * 2  # duck penalty
        + df.get("dot_balls", 0) * 0.5
    ).round(2)
    return df


# ─── 4. MODEL TRAINING ────────────────────────────────────────────────────────

def train_win_predictor(matches: pd.DataFrame) -> XGBClassifier:
    """
    Train XGBoost Win Probability classifier.
    Features: team1, team2, venue, toss_winner, toss_decision
    """
    df = matches.dropna(subset=["team1", "team2", "venue", "tosswinner", "tossdecision", "winningteam"])

    le = {}
    for col in ["team1", "team2", "venue", "tosswinner", "tossdecision"]:
        le[col] = LabelEncoder().fit(df[col])
        df[col + "_enc"] = le[col].transform(df[col])

    df["target"] = (df["winningteam"] == df["team1"]).astype(int)

    feature_cols = ["team1_enc", "team2_enc", "venue_enc", "tosswinner_enc", "tossdecision_enc"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[WinPredictor] Test accuracy: {acc:.2%}")

    # Save model + encoders
    joblib.dump({"model": model, "encoders": le}, os.path.join(MODEL_DIR, "win_model.pkl"))
    print("[WinPredictor] Model saved → models/win_model.pkl")
    return model


def train_fantasy_scorer(player_df: pd.DataFrame) -> RandomForestRegressor:
    """
    Train Random Forest regressor to predict fantasy points.
    """
    feature_cols = ["runs", "wickets", "balls_faced", "strike_rate",
                    "economy", "fours", "sixes", "dot_balls"]
    # Only use columns that exist
    available = [c for c in feature_cols if c in player_df.columns]
    target = "fantasy_points"

    df = player_df.dropna(subset=available + [target])
    X = df[available]
    y = df[target]

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump({"model": model, "features": available},
                os.path.join(MODEL_DIR, "fantasy_model.pkl"))
    print("[FantasyScorer] Model saved → models/fantasy_model.pkl")
    return model


# ─── 5. MAIN ──────────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("IPL Analytics Platform — ML Pipeline")
    print("=" * 60)

    # Ingest
    balls_raw, matches_raw = load_raw_data()

    # Clean
    balls = clean_balls(balls_raw)
    matches = clean_matches(matches_raw)

    # Feature engineering
    balls = add_phase_column(balls)
    batting = build_batting_stats(balls)
    bowling = build_bowling_stats(balls)
    players = build_player_impact(batting, bowling)
    players = build_fantasy_points(players)

    # Save processed data for API/app use
    players.to_csv(os.path.join(DATA_DIR, "player_stats.csv"), index=False)
    print(f"[Pipeline] Player stats saved: {len(players)} players.")

    # Train models
    train_win_predictor(matches)
    train_fantasy_scorer(players)

    print("\n✅ Pipeline complete. All models trained and saved.")
    return players, matches


if __name__ == "__main__":
    run_pipeline()

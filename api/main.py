"""
api/main.py
-----------
FastAPI application entry point.
Start with: uvicorn api.main:app --reload --port 8000
Docs at:    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import players, teams, matches, predictions

app = FastAPI(
    title="IPL Analytics Platform API",
    description=(
        "Production-grade REST API for IPL 2022 cricket analytics. "
        "Exposes player stats, team insights, win probability predictions, "
        "and ML-powered Fantasy XI recommendations."
    ),
    version="1.0.0",
    contact={"name": "Ayush Pandey"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(players.router, prefix="/api/players", tags=["Players"])
app.include_router(teams.router, prefix="/api/teams", tags=["Teams"])
app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
app.include_router(predictions.router, prefix="/api", tags=["Predictions & Fantasy"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "project": "IPL Analytics Platform",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

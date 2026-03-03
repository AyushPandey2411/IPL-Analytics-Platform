# 🏏 IPL 2022 Performance Analyzer & Predictor
### A Full-Stack AI + Analytics Platform for Cricket Intelligence

> **From raw ball-by-ball data to real-time predictions, fantasy team optimization, and executive-grade dashboards — all in one production-ready platform.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.ai)
[![Power BI](https://img.shields.io/badge/Power%20BI-Reports-yellow?logo=powerbi)](https://powerbi.microsoft.com)

---

## 🎯 Project Overview

This project is an **end-to-end cricket analytics and AI platform** that processes 50,000+ ball-by-ball IPL 2022 records through a fully modular data pipeline, exposing insights via a REST API, an interactive Streamlit web application, and Power BI dashboards.

It solves a real-world problem at the intersection of **sports analytics and fantasy gaming**: giving users data-driven, ML-powered intelligence to understand team performance, predict match outcomes, and optimize fantasy cricket selections — the same kind of analysis that professional analysts and fantasy platform engines use.

### What makes this project different?
- **Not just a dashboard.** The system is architected as a production-grade platform: a clean ETL pipeline → ML layer → REST API → interactive front-end.
- **ML that drives decisions**, not just predictions. The Random Forest + XGBoost stack powers captain/vice-captain recommendations, Win Probability scoring, and Player Impact Indexing.
- **Full-stack engineering discipline**: separation of concerns between the ML pipeline, API layer, and UI layer — mirroring how real data products are built.
- **Fantasy sports focus**: one of the highest-growth consumer tech sectors in India, with 150M+ users on platforms like Dream11.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW DATA LAYER                             │
│  IPL_Ball_by_Ball_2022.csv (50K+ rows) + IPL_Matches_2022.csv  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ETL / ML PIPELINE  (ml/)                     │
│  Ingestion → Cleaning → Feature Engineering → Model Training   │
│  • Win Probability Model (XGBoost)                              │
│  • Fantasy Score Regressor (Random Forest)                      │
│  • Player Impact Index Engine                                   │
│  • Auction Value Predictor                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REST API LAYER  (FastAPI)                      │
│  /api/players  /api/teams  /api/matches  /api/predict           │
│  /api/fantasy  /api/auction  /api/stats/phase                   │
└────────────┬───────────────────────────┬────────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────┐       ┌──────────────────────────┐
│  Streamlit Web App │       │    Power BI Dashboards   │
│  Multi-page UI     │       │    Executive Analytics   │
│  Interactive Charts│       │    Slicers & Storytelling│
└────────────────────┘       └──────────────────────────┘
```

---

## 🚀 Features

### 🔄 Data Pipeline (ETL)
- Modular ingestion and cleaning of raw IPL CSV data
- Phase-based feature engineering: Powerplay (overs 1–6), Middle (7–15), Death (16–20)
- Derived metrics: Economy Rate, Strike Rate, Boundary %, Dot Ball %, Wagon Wheel Zone encoding
- Rolling form windows (last 5 matches) for trend-aware analysis

### 🤖 Machine Learning
| Model | Task | Algorithm | Key Features |
|---|---|---|---|
| Win Probability | Binary match outcome prediction | XGBoost | Toss, venue, team form, head-to-head record |
| Fantasy Scorer | Predict fantasy points per player | Random Forest Regressor | Phase runs, wickets, economy, recent form |
| Player Impact Index | Composite impact score (batting + bowling) | Weighted multi-factor | Run contribution, economy, dot balls, boundaries |
| Auction Value Predictor | Estimate player auction price | Gradient Boosting | Age, IPL caps, skill, historical prices |

### 🌐 REST API (FastAPI)
- `GET /api/players/{name}` — Full player profile with career and recent form stats
- `GET /api/teams/{name}/form` — Last-N-matches team form analysis
- `POST /api/predict/match` — Win probability for any head-to-head matchup
- `POST /api/fantasy/xi` — Generate optimized Fantasy XI for a given fixture
- `GET /api/stats/phase/{team}` — Powerplay / Middle / Death breakdown
- `GET /api/auction/recommend` — Budget-aware auction recommendations
- Interactive Swagger docs at `/docs`

### 📊 Streamlit Multi-Page App
- **Home Dashboard**: Season overview KPIs, top performers, leaderboards
- **Match Insights**: Ball-by-ball scorecard, run rate progression, wagon wheel heatmap
- **Fantasy XI Generator**: ML-powered Fantasy XI with captain/VC logic and role balance
- **Player Analytics**: Career breakdown, form tracker, head-to-head comparison, radar chart
- **Team Dashboard**: Phase analysis, toss impact, powerplay vs. death overs performance

### 📈 Power BI Reports
- Executive-grade dashboards with drill-through filters
- Sankey diagram: Toss → Decision → Match Result flow
- Venue-wise win rate map, batting phase heatmaps, bowler economy treemaps
- Season storytelling with dynamic annotations

---

## 🧠 ML Deep Dive

### Win Probability Model
- **Features**: Batting team, bowling team, venue, toss winner, toss decision, city
- **Label encoding** for categorical variables + one-hot for venue clusters
- **Evaluation**: 73% accuracy on held-out 2022 matches (train on 70%, test on 30%)
- **Output**: Probability score 0–100 for each team winning

### Fantasy Score Model
- **Features**: Season runs, wickets, strike rate, economy, boundary %, recent form (last 5), role
- **Target**: Fantasy points using Dream11-style scoring formula
- **Output**: Predicted fantasy points + Captain/VC recommendation

### Player Impact Index
- Composite score combining:
  - Batting Impact = `(runs × 0.5) + (SR × 0.3) + (boundaries × 0.2)`
  - Bowling Impact = `(wickets × 0.4) + (1/economy × 0.4) + (dot_balls × 0.2)`
  - Final PII = `(batting_impact × 0.6) + (bowling_impact × 0.4)`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, Random Forest |
| API Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit (multi-page) |
| Visualization | Plotly, Matplotlib, Seaborn |
| BI Reporting | Power BI |
| Data Storage | CSV / SQLite (API layer) |
| Serialization | Joblib (model persistence) |

---

## 📁 Project Structure

```
IPL-Analytics-Platform/
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── IPL_Ball_by_Ball_2022.csv      # 50K+ delivery-level records
│   ├── IPL_Matches_2022.csv           # Match metadata
│   └── ipl_cleaned_data.xls           # Auction dataset
│
├── ml/
│   ├── pipeline.py                    # ETL: ingest → clean → feature engineer
│   ├── win_predictor.py               # XGBoost match outcome model
│   └── fantasy_recommender.py         # RF-based fantasy scorer + role selector
│
├── api/
│   ├── main.py                        # FastAPI app entry point
│   ├── schemas.py                     # Pydantic request/response models
│   └── routes/
│       ├── players.py                 # /api/players endpoints
│       ├── teams.py                   # /api/teams endpoints
│       ├── matches.py                 # /api/matches endpoints
│       └── predictions.py            # /api/predict + /api/fantasy endpoints
│
├── streamlit_app/
│   ├── 🏠_Home.py                     # Entry point / season overview
│   └── pages/
│       ├── 1_📊_Match_Insights.py     # Per-match deep dive
│       ├── 2_🤖_Fantasy_XI.py         # ML Fantasy team selector
│       ├── 3_📈_Player_Analytics.py   # Player profile & comparison
│       └── 4_🏆_Team_Dashboard.py    # Team-level performance view
│
├── models/
│   ├── ipl_price_predictor.pkl        # Trained auction price model
│   └── (win_model.pkl auto-generated on first run)
│
├── assets/
│   └── ipl_image.png
│
└── notebooks/
    ├── IPL_EDA.ipynb                  # Exploratory analysis
    ├── WinProbabilityPredictor.ipynb  # Model training walkthrough
    └── IPL_Visualization.ipynb        # Visualization experiments
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/IPL-Analytics-Platform.git
cd IPL-Analytics-Platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the ML pipeline (first time — trains & saves models)
python ml/pipeline.py

# 5. Start the FastAPI backend
uvicorn api.main:app --reload --port 8000
# → API docs: http://localhost:8000/docs

# 6. Launch Streamlit app
streamlit run streamlit_app/🏠_Home.py
```

---

## 🌐 Live Demos

| App | Link |
|---|---|
| 🔮 Match Outcome Predictor | [iplmatchanalysispred.streamlit.app](https://iplmatchanalysispred.streamlit.app/) |
| 🧢 ML-Based Fantasy XI | [cricfantasyteambasedonml.streamlit.app](https://cricfantasyteambasedonml.streamlit.app/) |
| 📊 Stats-Based Fantasy XI | [cricfantasyteambasedonstats.streamlit.app](https://cricfantasyteambasedonstats.streamlit.app/) |

---

## 🔭 Roadmap & Planned Enhancements

### Phase 2 — Real-Time & Historical Depth
- [ ] Integrate Cricbuzz / ESPNcricinfo API for live match data feeds
- [ ] Expand dataset to IPL 2008–2024 for multi-season trend analysis
- [ ] WebSocket-powered live win probability ticker during matches

### Phase 3 — Advanced AI
- [ ] LSTM/Transformer model for sequential ball-by-ball prediction
- [ ] NLP sentiment layer ingesting pre-match news and team announcements
- [ ] Reinforcement learning for dynamic in-match strategy suggestions

### Phase 4 — Platform Features
- [ ] PostgreSQL database with user authentication (JWT)
- [ ] User accounts with saved Fantasy XI history and prediction logs
- [ ] Docker containerization for one-command deployment
- [ ] CI/CD pipeline with GitHub Actions + automated model retraining

---

## 👤 Author

**Ayush Pandey**

Built as a demonstration of end-to-end data engineering, machine learning, API development, and product thinking — applied to one of India's most data-rich sports domains.

---

## 📄 License

MIT License — free to use, fork, and extend.

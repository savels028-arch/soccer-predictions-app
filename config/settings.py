"""
Soccer Predictions App - Global Settings & Configuration
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, DB_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────
DATABASE_PATH = DB_DIR / "soccer_predictions.db"

# ──────────────────────────────────────────────
# API KEYS (Set via environment variables)
# ──────────────────────────────────────────────
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")

# ──────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# ──────────────────────────────────────────────
# SUPPORTED LEAGUES
# ──────────────────────────────────────────────
LEAGUES = {
    "PL":  {"name": "Premier League",   "country": "England",  "fd_code": "PL",  "api_id": 39,  "emoji": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "PD":  {"name": "La Liga",          "country": "Spain",    "fd_code": "PD",  "api_id": 140, "emoji": "🇪🇸"},
    "BL1": {"name": "Bundesliga",       "country": "Germany",  "fd_code": "BL1", "api_id": 78,  "emoji": "🇩🇪"},
    "SA":  {"name": "Serie A",          "country": "Italy",    "fd_code": "SA",  "api_id": 135, "emoji": "🇮🇹"},
    "FL1": {"name": "Ligue 1",          "country": "France",   "fd_code": "FL1", "api_id": 61,  "emoji": "🇫🇷"},
    "CL":  {"name": "Champions League", "country": "Europe",   "fd_code": "CL",  "api_id": 2,   "emoji": "🏆"},
    "EL":  {"name": "Europa League",    "country": "Europe",   "fd_code": "EL",  "api_id": 3,   "emoji": "🏆"},
    "DED": {"name": "Eredivisie",       "country": "Netherlands","fd_code": "DED","api_id": 88,  "emoji": "🇳🇱"},
    "PPL": {"name": "Primeira Liga",    "country": "Portugal", "fd_code": "PPL", "api_id": 94,  "emoji": "🇵🇹"},
    "BSA": {"name": "Série A",          "country": "Brazil",   "fd_code": "BSA", "api_id": 71,  "emoji": "🇧🇷"},
}

# ──────────────────────────────────────────────
# ML MODEL SETTINGS  (v1 — baseline)
# ──────────────────────────────────────────────
ML_SETTINGS = {
    "test_size": 0.2,
    "random_state": 42,
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "ensemble": {
        "weights": {"xgboost": 0.4, "neural_network": 0.35, "random_forest": 0.25},
    },
    "coupon": {
        "min_edge_pct": 5.0,          # require ≥5% edge (backtest-validated)
        "min_confidence_pct": 40.0,    # require ≥40% confidence
        "min_picks": 2,
        "max_picks": 6,
        "max_per_league": 2,
        "skip_high_disagreement": False,
    },
}

# ──────────────────────────────────────────────
# PAPER TRADING — live tracking before real money
# Based on full 10-league holdout backtest (9613 matches):
#   v1 overall: +3.7% ROI (flat), +2.7% ROI (Kelly)
# ──────────────────────────────────────────────
PAPER_TRADING = {
    # Leagues with positive ROI in backtest (exclude ELC -4.3%)
    "profitable_leagues": ["DED", "BL1", "SA", "BSA", "PL", "FL1", "BL2", "PD", "PPL"],
    "excluded_leagues": ["ELC"],  # Championship: -4.3% ROI in backtest
    # Filtering
    "min_edge_pct": 5.0,
    "min_confidence_pct": 40.0,
    # Paper bankroll (DKK) — for tracking, not real money
    "starting_bankroll": 10000,
    "stake_per_bet": 100,       # flat 100 DKK per bet
    "use_kelly": False,         # start with flat staking
}

# ──────────────────────────────────────────────
# ML MODEL SETTINGS v2 — optimised challenger
# ──────────────────────────────────────────────
ML_SETTINGS_V2 = {
    "test_size": 0.2,
    "random_state": 42,
    "model_suffix": "_v2",           # separate model files: xgboost_v2_model.pkl
    "use_v2_features": True,         # enable ELO, weighted form, CSV extra_data
    "num_training_seasons": 6,       # 6 seasons instead of 3
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_weight": 5,
        "gamma": 0.3,
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dropout_rates": [0.4, 0.3, 0.2],
        "use_lr_scheduler": True,
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
    },
    "lightgbm": {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 30,
        "num_leaves": 20,
    },
    "ensemble": {
        "weights": {
            "xgboost": 0.30,
            "neural_network": 0.25,
            "random_forest": 0.20,
            "lightgbm": 0.25,
        },
        "use_stacking": True,
    },
    "coupon": {
        "min_edge_pct": 5.0,          # require ≥5% edge
        "min_confidence_pct": 55.0,    # require ≥55% confidence
        "min_picks": 2,
        "max_picks": 6,
        "max_per_league": 2,
        "skip_high_disagreement": True,
        "sort_by": "edge_x_confidence",  # edge*confidence composite
    },
}

# ──────────────────────────────────────────────
# A/B TEST CONFIGURATION
# ──────────────────────────────────────────────
AB_TEST = {
    "enabled": False,
    "v1_label": "ML Ensemble v1",
    "v2_label": "ML Ensemble v2",
    "v1_config": "ML_SETTINGS",
    "v2_config": "ML_SETTINGS_V2",
    "auto_promote": True,           # coupon uses whichever version leads
    "min_samples_to_compare": 20,    # need ≥20 evaluated matches to switch
}

# ──────────────────────────────────────────────
# GUI SETTINGS
# ──────────────────────────────────────────────
GUI_SETTINGS = {
    "title": "⚽ Soccer Predictions Pro",
    "width": 1400,
    "height": 900,
    "min_width": 1100,
    "min_height": 700,
    "theme": "dark",
    "refresh_interval_seconds": 60,
    "colors": {
        "bg_dark": "#0f0f1a",
        "bg_medium": "#1a1a2e",
        "bg_light": "#16213e",
        "bg_card": "#1e2a4a",
        "accent": "#00d4ff",
        "accent_green": "#00e676",
        "accent_red": "#ff5252",
        "accent_yellow": "#ffd740",
        "accent_orange": "#ff9100",
        "text_primary": "#ffffff",
        "text_secondary": "#a0b4d0",
        "text_muted": "#5a6a8a",
        "border": "#2a3a5a",
        "win_color": "#00e676",
        "draw_color": "#ffd740",
        "lose_color": "#ff5252",
        "high_value": "#00e676",
        "medium_value": "#ffd740",
        "low_value": "#ff5252",
    },
}

# ──────────────────────────────────────────────
# DATA REFRESH
# ──────────────────────────────────────────────
DATA_SETTINGS = {
    "cache_ttl_minutes": 15,
    "max_historical_seasons": 5,
    "live_refresh_seconds": 30,
}

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
LOG_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": LOGS_DIR / "app.log",
}

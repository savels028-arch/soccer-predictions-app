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
# ML MODEL SETTINGS
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

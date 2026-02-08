"""
Database Manager - SQLite backend for Soccer Predictions App
Handles all data storage: matches, teams, predictions, history.
"""
import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DATABASE_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for the soccer predictions app."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"Database connected: {self.db_path}")

    def _create_tables(self):
        """Create all required tables."""
        cursor = self.conn.cursor()

        # ── Teams ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                short_name TEXT,
                country TEXT,
                league_code TEXT,
                logo_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Matches ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_id INTEGER UNIQUE,
                league_code TEXT NOT NULL,
                season INTEGER,
                matchday INTEGER,
                match_date TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'SCHEDULED',
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team_name TEXT NOT NULL,
                away_team_name TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                home_ht_score INTEGER,
                away_ht_score INTEGER,
                venue TEXT,
                referee TEXT,
                home_odds REAL,
                draw_odds REAL,
                away_odds REAL,
                extra_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams(id),
                FOREIGN KEY (away_team_id) REFERENCES teams(id)
            )
        """)

        # ── Team Statistics ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                team_name TEXT NOT NULL,
                league_code TEXT NOT NULL,
                season INTEGER NOT NULL,
                matches_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                goals_scored INTEGER DEFAULT 0,
                goals_conceded INTEGER DEFAULT 0,
                clean_sheets INTEGER DEFAULT 0,
                home_wins INTEGER DEFAULT 0,
                home_draws INTEGER DEFAULT 0,
                home_losses INTEGER DEFAULT 0,
                away_wins INTEGER DEFAULT 0,
                away_draws INTEGER DEFAULT 0,
                away_losses INTEGER DEFAULT 0,
                home_goals_scored INTEGER DEFAULT 0,
                home_goals_conceded INTEGER DEFAULT 0,
                away_goals_scored INTEGER DEFAULT 0,
                away_goals_conceded INTEGER DEFAULT 0,
                form TEXT,
                avg_goals_scored REAL DEFAULT 0.0,
                avg_goals_conceded REAL DEFAULT 0.0,
                xg REAL DEFAULT 0.0,
                xga REAL DEFAULT 0.0,
                possession_avg REAL DEFAULT 0.0,
                shots_avg REAL DEFAULT 0.0,
                shots_on_target_avg REAL DEFAULT 0.0,
                corners_avg REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_name, league_code, season)
            )
        """)

        # ── Head to Head ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS head_to_head (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league_code TEXT,
                match_date TIMESTAMP,
                home_score INTEGER,
                away_score INTEGER,
                season INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Predictions ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                match_date TIMESTAMP,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league_code TEXT,
                model_name TEXT NOT NULL,
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                predicted_home_goals REAL,
                predicted_away_goals REAL,
                predicted_outcome TEXT,
                confidence REAL,
                value_rating REAL,
                suggestion TEXT,
                actual_outcome TEXT,
                is_correct INTEGER,
                extra_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(id)
            )
        """)

        # ── Model Performance ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                roi REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name)
            )
        """)

        # ── Cached API Data ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Indexes ──
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_stats(team_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_h2h_teams ON head_to_head(home_team, away_team)")

        self.conn.commit()
        logger.info("Database tables created/verified")

    # ──────────────────────────────────────────────
    # TEAM OPERATIONS
    # ──────────────────────────────────────────────
    def upsert_team(self, api_id: int, name: str, short_name: str = "",
                    country: str = "", league_code: str = "", logo_url: str = "") -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO teams (api_id, name, short_name, country, league_code, logo_url)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(api_id) DO UPDATE SET
                name=excluded.name, short_name=excluded.short_name,
                country=excluded.country, league_code=excluded.league_code,
                logo_url=excluded.logo_url, updated_at=CURRENT_TIMESTAMP
        """, (api_id, name, short_name, country, league_code, logo_url))
        self.conn.commit()
        return cursor.lastrowid

    def get_team(self, name: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM teams WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ──────────────────────────────────────────────
    # MATCH OPERATIONS
    # ──────────────────────────────────────────────
    def upsert_match(self, match_data: Dict) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO matches (
                api_id, league_code, season, matchday, match_date, status,
                home_team_name, away_team_name, home_score, away_score,
                home_ht_score, away_ht_score, venue, referee,
                home_odds, draw_odds, away_odds, extra_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(api_id) DO UPDATE SET
                status=excluded.status, home_score=excluded.home_score,
                away_score=excluded.away_score, home_ht_score=excluded.home_ht_score,
                away_ht_score=excluded.away_ht_score, home_odds=excluded.home_odds,
                draw_odds=excluded.draw_odds, away_odds=excluded.away_odds,
                extra_data=excluded.extra_data, updated_at=CURRENT_TIMESTAMP
        """, (
            match_data.get("api_id"), match_data.get("league_code"),
            match_data.get("season"), match_data.get("matchday"),
            match_data.get("match_date"), match_data.get("status", "SCHEDULED"),
            match_data.get("home_team_name"), match_data.get("away_team_name"),
            match_data.get("home_score"), match_data.get("away_score"),
            match_data.get("home_ht_score"), match_data.get("away_ht_score"),
            match_data.get("venue"), match_data.get("referee"),
            match_data.get("home_odds"), match_data.get("draw_odds"),
            match_data.get("away_odds"),
            json.dumps(match_data.get("extra_data", {})),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_matches_by_date(self, match_date: str, league_code: Optional[str] = None) -> List[Dict]:
        cursor = self.conn.cursor()
        if league_code:
            cursor.execute("""
                SELECT * FROM matches
                WHERE DATE(match_date) = ? AND league_code = ?
                ORDER BY match_date
            """, (match_date, league_code))
        else:
            cursor.execute("""
                SELECT * FROM matches
                WHERE DATE(match_date) = ?
                ORDER BY league_code, match_date
            """, (match_date,))
        return [dict(row) for row in cursor.fetchall()]

    def get_todays_matches(self, league_code: Optional[str] = None) -> List[Dict]:
        today = date.today().isoformat()
        return self.get_matches_by_date(today, league_code)

    def get_upcoming_matches(self, days: int = 7, league_code: Optional[str] = None) -> List[Dict]:
        cursor = self.conn.cursor()
        if league_code:
            cursor.execute("""
                SELECT * FROM matches
                WHERE DATE(match_date) BETWEEN DATE('now') AND DATE('now', ? || ' days')
                AND league_code = ?
                ORDER BY match_date
            """, (str(days), league_code))
        else:
            cursor.execute("""
                SELECT * FROM matches
                WHERE DATE(match_date) BETWEEN DATE('now') AND DATE('now', ? || ' days')
                ORDER BY match_date
            """, (str(days),))
        return [dict(row) for row in cursor.fetchall()]

    def get_team_matches(self, team_name: str, limit: int = 20) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM matches
            WHERE (home_team_name = ? OR away_team_name = ?)
            AND status = 'FINISHED'
            ORDER BY match_date DESC LIMIT ?
        """, (team_name, team_name, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_live_matches(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM matches
            WHERE status IN ('IN_PLAY', 'PAUSED', 'HALFTIME', 'LIVE', '1H', '2H', 'HT')
            ORDER BY match_date
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_finished_matches(self, league_code: Optional[str] = None,
                              season: Optional[int] = None, limit: int = 500) -> List[Dict]:
        cursor = self.conn.cursor()
        conditions = ["status = 'FINISHED'"]
        params = []
        if league_code:
            conditions.append("league_code = ?")
            params.append(league_code)
        if season:
            conditions.append("season = ?")
            params.append(season)
        where_clause = " AND ".join(conditions)
        params.append(limit)
        cursor.execute(f"""
            SELECT * FROM matches WHERE {where_clause}
            ORDER BY match_date DESC LIMIT ?
        """, params)
        return [dict(row) for row in cursor.fetchall()]

    # ──────────────────────────────────────────────
    # TEAM STATS OPERATIONS
    # ──────────────────────────────────────────────
    def upsert_team_stats(self, stats: Dict):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO team_stats (
                team_name, league_code, season, matches_played,
                wins, draws, losses, goals_scored, goals_conceded,
                clean_sheets, home_wins, home_draws, home_losses,
                away_wins, away_draws, away_losses,
                home_goals_scored, home_goals_conceded,
                away_goals_scored, away_goals_conceded,
                form, avg_goals_scored, avg_goals_conceded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_name, league_code, season) DO UPDATE SET
                matches_played=excluded.matches_played, wins=excluded.wins,
                draws=excluded.draws, losses=excluded.losses,
                goals_scored=excluded.goals_scored, goals_conceded=excluded.goals_conceded,
                clean_sheets=excluded.clean_sheets,
                home_wins=excluded.home_wins, home_draws=excluded.home_draws,
                home_losses=excluded.home_losses,
                away_wins=excluded.away_wins, away_draws=excluded.away_draws,
                away_losses=excluded.away_losses,
                home_goals_scored=excluded.home_goals_scored,
                home_goals_conceded=excluded.home_goals_conceded,
                away_goals_scored=excluded.away_goals_scored,
                away_goals_conceded=excluded.away_goals_conceded,
                form=excluded.form,
                avg_goals_scored=excluded.avg_goals_scored,
                avg_goals_conceded=excluded.avg_goals_conceded,
                updated_at=CURRENT_TIMESTAMP
        """, (
            stats["team_name"], stats["league_code"], stats["season"],
            stats.get("matches_played", 0), stats.get("wins", 0),
            stats.get("draws", 0), stats.get("losses", 0),
            stats.get("goals_scored", 0), stats.get("goals_conceded", 0),
            stats.get("clean_sheets", 0),
            stats.get("home_wins", 0), stats.get("home_draws", 0), stats.get("home_losses", 0),
            stats.get("away_wins", 0), stats.get("away_draws", 0), stats.get("away_losses", 0),
            stats.get("home_goals_scored", 0), stats.get("home_goals_conceded", 0),
            stats.get("away_goals_scored", 0), stats.get("away_goals_conceded", 0),
            stats.get("form", ""), stats.get("avg_goals_scored", 0.0),
            stats.get("avg_goals_conceded", 0.0),
        ))
        self.conn.commit()

    def get_team_stats(self, team_name: str, league_code: str = "",
                       season: Optional[int] = None) -> Optional[Dict]:
        cursor = self.conn.cursor()
        if season and league_code:
            cursor.execute("""
                SELECT * FROM team_stats
                WHERE team_name = ? AND league_code = ? AND season = ?
            """, (team_name, league_code, season))
        elif league_code:
            cursor.execute("""
                SELECT * FROM team_stats
                WHERE team_name = ? AND league_code = ?
                ORDER BY season DESC LIMIT 1
            """, (team_name, league_code))
        else:
            cursor.execute("""
                SELECT * FROM team_stats
                WHERE team_name = ?
                ORDER BY season DESC LIMIT 1
            """, (team_name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ──────────────────────────────────────────────
    # HEAD TO HEAD
    # ──────────────────────────────────────────────
    def add_h2h(self, home: str, away: str, home_score: int, away_score: int,
                league_code: str = "", match_date: str = "", season: int = 0):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO head_to_head (home_team, away_team, league_code, match_date, home_score, away_score, season)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (home, away, league_code, match_date, home_score, away_score, season))
        self.conn.commit()

    def get_h2h(self, team1: str, team2: str, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM head_to_head
            WHERE (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)
            ORDER BY match_date DESC LIMIT ?
        """, (team1, team2, team2, team1, limit))
        return [dict(row) for row in cursor.fetchall()]

    # ──────────────────────────────────────────────
    # PREDICTION OPERATIONS
    # ──────────────────────────────────────────────
    def save_prediction(self, prediction: Dict) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                match_id, match_date, home_team, away_team, league_code,
                model_name, home_win_prob, draw_prob, away_win_prob,
                predicted_home_goals, predicted_away_goals,
                predicted_outcome, confidence, value_rating,
                suggestion, extra_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.get("match_id"), prediction.get("match_date"),
            prediction["home_team"], prediction["away_team"],
            prediction.get("league_code"), prediction["model_name"],
            prediction.get("home_win_prob", 0), prediction.get("draw_prob", 0),
            prediction.get("away_win_prob", 0),
            prediction.get("predicted_home_goals"), prediction.get("predicted_away_goals"),
            prediction.get("predicted_outcome"), prediction.get("confidence", 0),
            prediction.get("value_rating", 0), prediction.get("suggestion", ""),
            json.dumps(prediction.get("extra_data", {})),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_predictions_for_match(self, match_id: int) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions WHERE match_id = ?
            ORDER BY model_name
        """, (match_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_todays_predictions(self) -> List[Dict]:
        cursor = self.conn.cursor()
        today = date.today().isoformat()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE DATE(match_date) = ?
            ORDER BY confidence DESC
        """, (today,))
        return [dict(row) for row in cursor.fetchall()]

    # ──────────────────────────────────────────────
    # MODEL PERFORMANCE
    # ──────────────────────────────────────────────
    def update_model_performance(self, model_name: str, total: int, correct: int,
                                  accuracy: float, roi: float = 0.0, avg_conf: float = 0.0):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_performance (model_name, total_predictions, correct_predictions, accuracy, roi, avg_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                total_predictions=excluded.total_predictions,
                correct_predictions=excluded.correct_predictions,
                accuracy=excluded.accuracy, roi=excluded.roi,
                avg_confidence=excluded.avg_confidence,
                last_updated=CURRENT_TIMESTAMP
        """, (model_name, total, correct, accuracy, roi, avg_conf))
        self.conn.commit()

    def get_all_model_performance(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM model_performance ORDER BY accuracy DESC")
        return [dict(row) for row in cursor.fetchall()]

    # ──────────────────────────────────────────────
    # CACHE OPERATIONS
    # ──────────────────────────────────────────────
    def set_cache(self, key: str, data: Any, ttl_minutes: int = 15):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO api_cache (cache_key, data, expires_at)
            VALUES (?, ?, datetime('now', ? || ' minutes'))
            ON CONFLICT(cache_key) DO UPDATE SET
                data=excluded.data, expires_at=excluded.expires_at
        """, (key, json.dumps(data), str(ttl_minutes)))
        self.conn.commit()

    def get_cache(self, key: str) -> Optional[Any]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT data FROM api_cache
            WHERE cache_key = ? AND expires_at > datetime('now')
        """, (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def clear_expired_cache(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM api_cache WHERE expires_at <= datetime('now')")
        self.conn.commit()

    # ──────────────────────────────────────────────
    # STATISTICS COMPUTATION
    # ──────────────────────────────────────────────
    def compute_team_stats_from_matches(self, team_name: str, league_code: str, season: int) -> Dict:
        """Compute stats from stored match data."""
        matches = self.get_team_matches(team_name, limit=100)
        matches = [m for m in matches if m.get("league_code") == league_code and m.get("season") == season]

        stats = {
            "team_name": team_name, "league_code": league_code, "season": season,
            "matches_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_scored": 0, "goals_conceded": 0, "clean_sheets": 0,
            "home_wins": 0, "home_draws": 0, "home_losses": 0,
            "away_wins": 0, "away_draws": 0, "away_losses": 0,
            "home_goals_scored": 0, "home_goals_conceded": 0,
            "away_goals_scored": 0, "away_goals_conceded": 0,
            "form": "",
        }

        form_list = []
        for m in matches:
            if m["home_score"] is None or m["away_score"] is None:
                continue
            stats["matches_played"] += 1
            is_home = m["home_team_name"] == team_name
            gs = m["home_score"] if is_home else m["away_score"]
            gc = m["away_score"] if is_home else m["home_score"]
            stats["goals_scored"] += gs
            stats["goals_conceded"] += gc

            if gc == 0:
                stats["clean_sheets"] += 1

            if gs > gc:
                result = "W"
                stats["wins"] += 1
                if is_home:
                    stats["home_wins"] += 1
                else:
                    stats["away_wins"] += 1
            elif gs == gc:
                result = "D"
                stats["draws"] += 1
                if is_home:
                    stats["home_draws"] += 1
                else:
                    stats["away_draws"] += 1
            else:
                result = "L"
                stats["losses"] += 1
                if is_home:
                    stats["home_losses"] += 1
                else:
                    stats["away_losses"] += 1

            if is_home:
                stats["home_goals_scored"] += gs
                stats["home_goals_conceded"] += gc
            else:
                stats["away_goals_scored"] += gs
                stats["away_goals_conceded"] += gc

            form_list.append(result)

        stats["form"] = "".join(form_list[:5])
        if stats["matches_played"] > 0:
            stats["avg_goals_scored"] = round(stats["goals_scored"] / stats["matches_played"], 2)
            stats["avg_goals_conceded"] = round(stats["goals_conceded"] / stats["matches_played"], 2)

        return stats

    # ──────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def get_match_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM matches")
        return cursor.fetchone()["cnt"]

    def get_prediction_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
        return cursor.fetchone()["cnt"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

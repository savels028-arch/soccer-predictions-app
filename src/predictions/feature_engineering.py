"""
Feature Engineering for Soccer Predictions
Builds ML-ready features from match & team data.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for ML models from raw match data."""

    # Feature names for documentation/reference
    FEATURE_NAMES = [
        "home_win_pct", "home_draw_pct", "home_loss_pct",
        "away_win_pct", "away_draw_pct", "away_loss_pct",
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        "home_home_win_pct", "home_home_draw_pct",
        "away_away_win_pct", "away_away_draw_pct",
        "home_home_goals_avg", "home_home_conceded_avg",
        "away_away_goals_avg", "away_away_conceded_avg",
        "home_form_score", "away_form_score",
        "home_clean_sheet_pct", "away_clean_sheet_pct",
        "goal_diff_home", "goal_diff_away",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        "h2h_home_goals_avg", "h2h_away_goals_avg",
        "home_points_per_game", "away_points_per_game",
        "odds_home", "odds_draw", "odds_away",
        "implied_prob_home", "implied_prob_draw", "implied_prob_away",
        "has_real_odds",
        "ai_consensus_home", "ai_consensus_draw", "ai_consensus_away",
        "ai_num_sources", "ai_agreement",
    ]

    @staticmethod
    def form_to_score(form: str) -> float:
        """Convert form string (e.g. 'WWDLW') to numeric score (0-1)."""
        if not form or form == "-----":
            return 0.5
        score_map = {"W": 3, "D": 1, "L": 0}
        total = sum(score_map.get(c, 0) for c in form)
        max_score = len(form) * 3
        return total / max_score if max_score > 0 else 0.5

    @staticmethod
    def safe_pct(part: int, total: int) -> float:
        """Safe percentage calculation."""
        return round(part / total, 4) if total > 0 else 0.0

    @staticmethod
    def implied_probability(odds: Optional[float]) -> float:
        """Convert decimal odds to implied probability."""
        if odds and odds > 0:
            return round(1 / odds, 4)
        return 0.33

    @classmethod
    def build_match_features(cls, home_stats: Dict, away_stats: Dict,
                              h2h: List[Dict] = None,
                              home_odds: float = None,
                              draw_odds: float = None,
                              away_odds: float = None,
                              ai_predictions: List[Dict] = None) -> np.ndarray:
        """
        Build a feature vector for a single match prediction.

        Args:
            home_stats: Team statistics dict for home team
            away_stats: Team statistics dict for away team
            h2h: Head-to-head match history
            home_odds: Decimal odds for home win
            draw_odds: Decimal odds for draw
            away_odds: Decimal odds for away win
            ai_predictions: List of AI-site prediction dicts with home/draw/away probs

        Returns:
            1D numpy array of features
        """
        h_mp = home_stats.get("matches_played", 1) or 1
        a_mp = away_stats.get("matches_played", 1) or 1

        features = []

        # ── Overall percentages ──
        features.append(cls.safe_pct(home_stats.get("wins", 0), h_mp))
        features.append(cls.safe_pct(home_stats.get("draws", 0), h_mp))
        features.append(cls.safe_pct(home_stats.get("losses", 0), h_mp))
        features.append(cls.safe_pct(away_stats.get("wins", 0), a_mp))
        features.append(cls.safe_pct(away_stats.get("draws", 0), a_mp))
        features.append(cls.safe_pct(away_stats.get("losses", 0), a_mp))

        # ── Goals averages ──
        features.append(home_stats.get("avg_goals_scored", 1.3))
        features.append(home_stats.get("avg_goals_conceded", 1.1))
        features.append(away_stats.get("avg_goals_scored", 1.2))
        features.append(away_stats.get("avg_goals_conceded", 1.2))

        # ── Home/Away specific ──
        h_home_mp = (home_stats.get("home_wins", 0) + home_stats.get("home_draws", 0) +
                     home_stats.get("home_losses", 0)) or 1
        a_away_mp = (away_stats.get("away_wins", 0) + away_stats.get("away_draws", 0) +
                     away_stats.get("away_losses", 0)) or 1

        features.append(cls.safe_pct(home_stats.get("home_wins", 0), h_home_mp))
        features.append(cls.safe_pct(home_stats.get("home_draws", 0), h_home_mp))
        features.append(cls.safe_pct(away_stats.get("away_wins", 0), a_away_mp))
        features.append(cls.safe_pct(away_stats.get("away_draws", 0), a_away_mp))

        h_home_goals = home_stats.get("home_goals_scored", 0) / h_home_mp
        h_home_conceded = home_stats.get("home_goals_conceded", 0) / h_home_mp
        a_away_goals = away_stats.get("away_goals_scored", 0) / a_away_mp
        a_away_conceded = away_stats.get("away_goals_conceded", 0) / a_away_mp

        features.extend([h_home_goals, h_home_conceded, a_away_goals, a_away_conceded])

        # ── Form ──
        features.append(cls.form_to_score(home_stats.get("form", "")))
        features.append(cls.form_to_score(away_stats.get("form", "")))

        # ── Clean sheets ──
        features.append(cls.safe_pct(home_stats.get("clean_sheets", 0), h_mp))
        features.append(cls.safe_pct(away_stats.get("clean_sheets", 0), a_mp))

        # ── Goal difference ──
        features.append((home_stats.get("goals_scored", 0) - home_stats.get("goals_conceded", 0)) / h_mp)
        features.append((away_stats.get("goals_scored", 0) - away_stats.get("goals_conceded", 0)) / a_mp)

        # ── Head to Head ──
        h2h = h2h or []
        h2h_total = len(h2h) or 1
        h2h_home_wins = 0
        h2h_draws = 0
        h2h_away_wins = 0
        h2h_home_goals = 0
        h2h_away_goals = 0

        home_name = home_stats.get("team_name", "")
        for match in h2h:
            hs = match.get("home_score", 0) or 0
            aws = match.get("away_score", 0) or 0
            if match.get("home_team") == home_name:
                h2h_home_goals += hs
                h2h_away_goals += aws
                if hs > aws:
                    h2h_home_wins += 1
                elif hs == aws:
                    h2h_draws += 1
                else:
                    h2h_away_wins += 1
            else:
                h2h_home_goals += aws
                h2h_away_goals += hs
                if aws > hs:
                    h2h_home_wins += 1
                elif aws == hs:
                    h2h_draws += 1
                else:
                    h2h_away_wins += 1

        features.append(h2h_home_wins / h2h_total)
        features.append(h2h_draws / h2h_total)
        features.append(h2h_away_wins / h2h_total)
        features.append(h2h_home_goals / h2h_total)
        features.append(h2h_away_goals / h2h_total)

        # ── Points per game ──
        h_ppg = (home_stats.get("wins", 0) * 3 + home_stats.get("draws", 0)) / h_mp
        a_ppg = (away_stats.get("wins", 0) * 3 + away_stats.get("draws", 0)) / a_mp
        features.extend([h_ppg, a_ppg])

        # ── Odds features (Q1 fix: use 0 when missing, add has_real_odds flag) ──
        has_real_odds = 1.0 if (home_odds and home_odds > 1.0 and
                                draw_odds and draw_odds > 1.0 and
                                away_odds and away_odds > 1.0) else 0.0
        features.append(home_odds if has_real_odds else 0.0)
        features.append(draw_odds if has_real_odds else 0.0)
        features.append(away_odds if has_real_odds else 0.0)
        features.append(cls.implied_probability(home_odds) if has_real_odds else 0.33)
        features.append(cls.implied_probability(draw_odds) if has_real_odds else 0.33)
        features.append(cls.implied_probability(away_odds) if has_real_odds else 0.33)
        features.append(has_real_odds)

        # ── AI consensus features (Q3: use AI-site predictions as ML input) ──
        ai_predictions = ai_predictions or []
        if ai_predictions:
            ai_home = np.mean([p.get("home", p.get("home_win_pct", 0.33)) for p in ai_predictions])
            ai_draw = np.mean([p.get("draw", p.get("draw_pct", 0.33)) for p in ai_predictions])
            ai_away = np.mean([p.get("away", p.get("away_win_pct", 0.33)) for p in ai_predictions])
            # Normalize to sum to 1
            ai_total = ai_home + ai_draw + ai_away
            if ai_total > 0:
                ai_home /= ai_total
                ai_draw /= ai_total
                ai_away /= ai_total
            # Agreement: how many sources agree on the winner
            winners = []
            for p in ai_predictions:
                h = p.get("home", p.get("home_win_pct", 0))
                d = p.get("draw", p.get("draw_pct", 0))
                a = p.get("away", p.get("away_win_pct", 0))
                best = max(h, d, a)
                if best == h:
                    winners.append("home")
                elif best == a:
                    winners.append("away")
                else:
                    winners.append("draw")
            from collections import Counter
            most_common_count = Counter(winners).most_common(1)[0][1]
            agreement = most_common_count / len(ai_predictions)
            features.extend([ai_home, ai_draw, ai_away, float(len(ai_predictions)), agreement])
        else:
            features.extend([0.33, 0.33, 0.33, 0.0, 0.0])

        return np.array(features, dtype=np.float64)

    @classmethod
    def build_training_data(cls, matches: List[Dict], db_manager) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build training dataset from historical matches.

        Returns:
            X: feature matrix (n_samples, n_features)
            y: labels (0=home_win, 1=draw, 2=away_win)
            dates: list of match_date strings for temporal ordering (Q5)
        """
        X_list = []
        y_list = []
        date_list = []

        # Q5: Sort matches by date for temporal split later
        sorted_matches = sorted(matches, key=lambda m: m.get("match_date", ""))

        for match in sorted_matches:
            if match.get("home_score") is None or match.get("away_score") is None:
                continue
            if match.get("status") != "FINISHED":
                continue

            home_name = match["home_team_name"]
            away_name = match["away_team_name"]
            league_code = match.get("league_code", "")
            season = match.get("season", 2025)

            # Get stats
            home_stats = db_manager.get_team_stats(home_name, league_code, season)
            away_stats = db_manager.get_team_stats(away_name, league_code, season)

            if not home_stats:
                home_stats = db_manager.compute_team_stats_from_matches(home_name, league_code, season)
                if home_stats.get("matches_played", 0) < 3:
                    continue
                db_manager.upsert_team_stats(home_stats)

            if not away_stats:
                away_stats = db_manager.compute_team_stats_from_matches(away_name, league_code, season)
                if away_stats.get("matches_played", 0) < 3:
                    continue
                db_manager.upsert_team_stats(away_stats)

            h2h = db_manager.get_h2h(home_name, away_name)

            try:
                features = cls.build_match_features(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds")
                )

                # Label
                hs = match["home_score"]
                aws = match["away_score"]
                if hs > aws:
                    label = 0  # Home win
                elif hs == aws:
                    label = 1  # Draw
                else:
                    label = 2  # Away win

                X_list.append(features)
                y_list.append(label)
                date_list.append(match.get("match_date", ""))

            except Exception as e:
                logger.error(f"Feature engineering error: {e}")
                continue

        if not X_list:
            logger.warning("No training data could be built")
            return np.empty((0, len(cls.FEATURE_NAMES))), np.empty(0), []

        return np.array(X_list), np.array(y_list), date_list

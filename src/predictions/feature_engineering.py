"""
Feature Engineering for Soccer Predictions
Builds ML-ready features from match & team data.
v1: 42 features (baseline)
v2: 42 + 18 = 60 features (ELO, weighted form, CSV extra, days rest, goal trend)
"""
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# ELO TRACKER  — running ELO ratings computed from historical results
# ──────────────────────────────────────────────────────────────
class EloTracker:
    """Compute running ELO ratings from historical results."""

    K = 20               # rating sensitivity
    HOME_ADV = 100       # home-field advantage in ELO points
    DEFAULT_ELO = 1500

    def __init__(self):
        self.ratings: Dict[str, float] = defaultdict(lambda: self.DEFAULT_ELO)

    def expected(self, elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home: str, away: str, home_score: int, away_score: int):
        """Update ELO ratings after a match (call in chronological order)."""
        r_h = self.ratings[home] + self.HOME_ADV
        r_a = self.ratings[away]
        e_h = self.expected(r_h, r_a)
        e_a = 1.0 - e_h

        if home_score > away_score:
            s_h, s_a = 1.0, 0.0
        elif home_score == away_score:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        self.ratings[home] += self.K * (s_h - e_h)
        self.ratings[away] += self.K * (s_a - e_a)

    def process_matches(self, matches: List[Dict]):
        """Process a sorted-by-date list of finished matches to build ELO ratings."""
        for m in matches:
            hs = m.get("home_score")
            aws = m.get("away_score")
            if hs is None or aws is None:
                continue
            home = m.get("home_team_name", "")
            away = m.get("away_team_name", "")
            if home and away:
                self.update(home, away, int(hs), int(aws))

    def get(self, team: str) -> float:
        return self.ratings[team]


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


# ──────────────────────────────────────────────────────────────
# V2 FEATURE ENGINEER — all v1 features + 18 new features
# ──────────────────────────────────────────────────────────────
class FeatureEngineerV2(FeatureEngineer):
    """Enhanced feature engineer with ELO, weighted form, CSV stats, etc."""

    EXTRA_FEATURE_NAMES = [
        # ELO (3)
        "home_elo", "away_elo", "elo_diff",
        # Weighted form (6) — exponentially-weighted last 3/5/10
        "home_form_w3", "home_form_w5", "home_form_w10",
        "away_form_w3", "away_form_w5", "away_form_w10",
        # CSV extra_data averages (10)
        "home_shots_on_target_avg", "away_shots_on_target_avg",
        "home_corners_avg", "away_corners_avg",
        "home_cards_avg", "away_cards_avg",
        "home_total_shots_avg", "away_total_shots_avg",
        "home_fouls_avg", "away_fouls_avg",
        # Days since last match (2)
        "home_days_rest", "away_days_rest",
        # Goal trend: last-5 avg minus season avg (1)
        "goal_trend_diff",
        # League encoding (10 — one-hot for top leagues)
        "league_PL", "league_PD", "league_BL1", "league_SA", "league_FL1",
        "league_CL", "league_EL", "league_DED", "league_PPL", "league_BSA",
        # Season progress / matchday (2)
        "season_progress", "matchday_norm",
        # Kickoff features (2)
        "kickoff_hour_sin", "kickoff_hour_cos",
        # Bookmaker overround (1)
        "overround",
        # H2H recency-weighted (2)
        "h2h_home_win_recency", "h2h_away_win_recency",
        # Strength of schedule — opponent ELO weighted form (2)
        "home_sos", "away_sos",
    ]

    FEATURE_NAMES = FeatureEngineer.FEATURE_NAMES + EXTRA_FEATURE_NAMES

    @staticmethod
    def weighted_form(results: List[str], n: int) -> float:
        """Exponentially-weighted form: recent results count more.
        results: list of 'W'/'D'/'L' from most-recent to oldest.
        """
        if not results:
            return 0.5
        vals = {"W": 3.0, "D": 1.0, "L": 0.0}
        subset = results[:n]
        if not subset:
            return 0.5
        decay = 0.9
        w_sum = 0.0
        weight_sum = 0.0
        for i, r in enumerate(subset):
            w = decay ** i
            w_sum += vals.get(r, 1.0) * w
            weight_sum += 3.0 * w
        return w_sum / weight_sum if weight_sum > 0 else 0.5

    @classmethod
    def build_match_features_v2(cls, home_stats: Dict, away_stats: Dict,
                                 h2h: List[Dict] = None,
                                 home_odds: float = None,
                                 draw_odds: float = None,
                                 away_odds: float = None,
                                 ai_predictions: List[Dict] = None,
                                 elo_tracker: 'EloTracker' = None,
                                 home_form_list: List[str] = None,
                                 away_form_list: List[str] = None,
                                 home_extra: Dict = None,
                                 away_extra: Dict = None,
                                 home_days_rest: float = 7.0,
                                 away_days_rest: float = 7.0,
                                 home_recent_goals_avg: float = None,
                                 away_recent_goals_avg: float = None,
                                 is_training: bool = False,
                                 league_code: str = "",
                                 matchday: int = 0,
                                 total_matchdays: int = 38,
                                 match_datetime: str = "",
                                 home_sos: float = 0.0,
                                 away_sos: float = 0.0) -> np.ndarray:
        """Build feature vector: 42 v1 features + v2 extra features."""

        # Build base v1 features — mask AI features during training
        if is_training:
            base = cls.build_match_features(
                home_stats, away_stats, h2h,
                home_odds, draw_odds, away_odds,
                ai_predictions=None,
            )
        else:
            base = cls.build_match_features(
                home_stats, away_stats, h2h,
                home_odds, draw_odds, away_odds,
                ai_predictions=ai_predictions,
            )

        extra = []

        # ── ELO ratings (3) ──
        home_name = home_stats.get("team_name", "")
        away_name = away_stats.get("team_name", "")
        if elo_tracker:
            h_elo = elo_tracker.get(home_name)
            a_elo = elo_tracker.get(away_name)
        else:
            h_elo = 1500.0
            a_elo = 1500.0
        extra.extend([h_elo / 1000.0, a_elo / 1000.0, (h_elo - a_elo) / 400.0])

        # ── Weighted form (6) ──
        hf = home_form_list or []
        af = away_form_list or []
        extra.append(cls.weighted_form(hf, 3))
        extra.append(cls.weighted_form(hf, 5))
        extra.append(cls.weighted_form(hf, 10))
        extra.append(cls.weighted_form(af, 3))
        extra.append(cls.weighted_form(af, 5))
        extra.append(cls.weighted_form(af, 10))

        # ── CSV extra_data averages (10) — expanded with total shots + fouls ──
        he = home_extra or {}
        ae = away_extra or {}
        extra.append(he.get("avg_shots_on_target", 0.0))
        extra.append(ae.get("avg_shots_on_target", 0.0))
        extra.append(he.get("avg_corners", 0.0))
        extra.append(ae.get("avg_corners", 0.0))
        extra.append(he.get("avg_cards", 0.0))
        extra.append(ae.get("avg_cards", 0.0))
        extra.append(he.get("avg_total_shots", 0.0))
        extra.append(ae.get("avg_total_shots", 0.0))
        extra.append(he.get("avg_fouls", 0.0))
        extra.append(ae.get("avg_fouls", 0.0))

        # ── Days rest (2) ──
        extra.append(min(home_days_rest, 30.0) / 7.0)
        extra.append(min(away_days_rest, 30.0) / 7.0)

        # ── Goal trend diff (1) ──
        h_season_avg = home_stats.get("avg_goals_scored", 1.3)
        a_season_avg = away_stats.get("avg_goals_scored", 1.2)
        h_recent = home_recent_goals_avg if home_recent_goals_avg is not None else h_season_avg
        a_recent = away_recent_goals_avg if away_recent_goals_avg is not None else a_season_avg
        goal_trend = (h_recent - h_season_avg) - (a_recent - a_season_avg)
        extra.append(round(goal_trend, 4))

        # ── League one-hot encoding (10) ──
        league_codes = ["PL", "PD", "BL1", "SA", "FL1", "CL", "EL", "DED", "PPL", "BSA"]
        for lc in league_codes:
            extra.append(1.0 if league_code == lc else 0.0)

        # ── Season progress / matchday (2) ──
        matchday = matchday or 0
        total_matchdays = total_matchdays or 38
        if matchday > 0 and total_matchdays > 0:
            extra.append(matchday / total_matchdays)  # season_progress
            extra.append(matchday / 38.0)             # matchday_norm (38 = standard league)
        else:
            extra.append(0.5)
            extra.append(0.5)

        # ── Kickoff hour as cyclical sin/cos (2) ──
        kickoff_hour = 15.0  # default
        if match_datetime:
            try:
                dt = datetime.fromisoformat(match_datetime[:19].replace("Z", ""))
                kickoff_hour = dt.hour + dt.minute / 60.0
            except Exception:
                pass
        extra.append(math.sin(2 * math.pi * kickoff_hour / 24.0))
        extra.append(math.cos(2 * math.pi * kickoff_hour / 24.0))

        # ── Bookmaker overround (1) ──
        if (home_odds and home_odds > 1.0 and
                draw_odds and draw_odds > 1.0 and
                away_odds and away_odds > 1.0):
            overround = (1/home_odds + 1/draw_odds + 1/away_odds) - 1.0
        else:
            overround = 0.05  # typical ~5%
        extra.append(round(overround, 4))

        # ── H2H recency-weighted wins (2) ──
        h2h = h2h or []
        h2h_home_recency = 0.0
        h2h_away_recency = 0.0
        h2h_weight_sum = 0.0
        decay = 0.85
        for i, match in enumerate(h2h):
            w = decay ** i
            hs = match.get("home_score", 0) or 0
            aws = match.get("away_score", 0) or 0
            mh = match.get("home_team", "")
            if mh == home_name:
                if hs > aws:
                    h2h_home_recency += w
                elif hs < aws:
                    h2h_away_recency += w
            else:
                if aws > hs:
                    h2h_home_recency += w
                elif aws < hs:
                    h2h_away_recency += w
            h2h_weight_sum += w
        if h2h_weight_sum > 0:
            extra.append(h2h_home_recency / h2h_weight_sum)
            extra.append(h2h_away_recency / h2h_weight_sum)
        else:
            extra.append(0.5)
            extra.append(0.5)

        # ── Strength of Schedule (2) ──
        extra.append(home_sos)
        extra.append(away_sos)

        return np.concatenate([base, np.array(extra, dtype=np.float64)])

    @classmethod
    def compute_csv_extra_averages(cls, matches: List[Dict], team_name: str) -> Dict:
        """Compute average shots-on-target, corners, cards, total shots, fouls from CSV extra_data."""
        sot_sum = 0.0
        corner_sum = 0.0
        card_sum = 0.0
        total_shots_sum = 0.0
        fouls_sum = 0.0
        count = 0

        for m in matches:
            ed = m.get("extra_data", {})
            if not ed:
                continue
            if isinstance(ed, str):
                try:
                    import json
                    ed = json.loads(ed)
                except Exception:
                    continue
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if h == team_name:
                sot = ed.get("home_shots_target")
                cor = ed.get("home_corners")
                yel = ed.get("home_yellow", 0) or 0
                red = ed.get("home_red", 0) or 0
                ts = ed.get("home_shots", ed.get("home_total_shots"))
                fl = ed.get("home_fouls")
            elif a == team_name:
                sot = ed.get("away_shots_target")
                cor = ed.get("away_corners")
                yel = ed.get("away_yellow", 0) or 0
                red = ed.get("away_red", 0) or 0
                ts = ed.get("away_shots", ed.get("away_total_shots"))
                fl = ed.get("away_fouls")
            else:
                continue
            if sot is not None:
                sot_sum += sot
                corner_sum += (cor or 0)
                card_sum += yel + red
                total_shots_sum += (ts or 0)
                fouls_sum += (fl or 0)
                count += 1

        if count == 0:
            return {}
        return {
            "avg_shots_on_target": round(sot_sum / count, 2),
            "avg_corners": round(corner_sum / count, 2),
            "avg_cards": round(card_sum / count, 2),
            "avg_total_shots": round(total_shots_sum / count, 2),
            "avg_fouls": round(fouls_sum / count, 2),
        }

    @classmethod
    def compute_form_list(cls, matches: List[Dict], team_name: str) -> List[str]:
        """Build ordered form list (most recent first) from sorted matches."""
        form = []
        for m in reversed(matches):
            hs = m.get("home_score")
            aws = m.get("away_score")
            if hs is None or aws is None:
                continue
            if m.get("status") != "FINISHED":
                continue
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if h == team_name:
                gs, gc = int(hs), int(aws)
            elif a == team_name:
                gs, gc = int(aws), int(hs)
            else:
                continue
            form.append("W" if gs > gc else "D" if gs == gc else "L")
        return form

    @classmethod
    def compute_days_since_last(cls, matches: List[Dict], team_name: str,
                                 reference_date: str) -> float:
        """Compute days since team's last match before reference_date."""
        try:
            ref = datetime.fromisoformat(reference_date[:10])
        except Exception:
            return 7.0
        latest = None
        for m in matches:
            md = m.get("match_date", "")[:10]
            if not md:
                continue
            try:
                dt = datetime.fromisoformat(md)
            except Exception:
                continue
            if dt >= ref:
                continue
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if h == team_name or a == team_name:
                if latest is None or dt > latest:
                    latest = dt
        if latest:
            return max((ref - latest).days, 1)
        return 7.0

    @classmethod
    def compute_recent_goals_avg(cls, matches: List[Dict], team_name: str, n: int = 5) -> Optional[float]:
        """Average goals scored in last n matches."""
        goals = []
        for m in reversed(matches):
            hs = m.get("home_score")
            aws = m.get("away_score")
            if hs is None or aws is None or m.get("status") != "FINISHED":
                continue
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if h == team_name:
                goals.append(int(hs))
            elif a == team_name:
                goals.append(int(aws))
            if len(goals) >= n:
                break
        return sum(goals) / len(goals) if goals else None

    @classmethod
    def compute_sos(cls, matches: List[Dict], team_name: str,
                    elo_tracker: 'EloTracker' = None, n: int = 10) -> float:
        """Strength of Schedule: avg opponent ELO-weighted result over last n matches.
        High value = good results against strong opponents.
        """
        if not elo_tracker:
            return 0.0
        scores = []
        for m in reversed(matches):
            hs = m.get("home_score")
            aws = m.get("away_score")
            if hs is None or aws is None or m.get("status") != "FINISHED":
                continue
            h = m.get("home_team_name", "")
            a = m.get("away_team_name", "")
            if h == team_name:
                opp_elo = elo_tracker.get(a) / 1500.0  # normalise
                gs, gc = int(hs), int(aws)
            elif a == team_name:
                opp_elo = elo_tracker.get(h) / 1500.0
                gs, gc = int(aws), int(hs)
            else:
                continue
            result = 1.0 if gs > gc else (0.5 if gs == gc else 0.0)
            scores.append(result * opp_elo)
            if len(scores) >= n:
                break
        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def build_training_data_v2(cls, matches: List[Dict], db_manager,
                                elo_tracker: 'EloTracker' = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build v2 training dataset with advanced features.
        Same interface as v1 build_training_data but returns 60-feature vectors.
        """
        X_list = []
        y_list = []
        date_list = []

        sorted_matches = sorted(matches, key=lambda m: m.get("match_date", ""))

        # Build ELO from training matches if not provided
        if elo_tracker is None:
            elo_tracker = EloTracker()
        # We'll update ELO as we go (in temporal order) to avoid leakage
        elo_copy = EloTracker()
        elo_copy.ratings = defaultdict(lambda: EloTracker.DEFAULT_ELO, elo_tracker.ratings)

        # Pre-index matches by team for efficient lookup
        team_matches: Dict[str, List[Dict]] = defaultdict(list)
        for m in sorted_matches:
            if m.get("status") == "FINISHED" and m.get("home_score") is not None:
                team_matches[m.get("home_team_name", "")].append(m)
                team_matches[m.get("away_team_name", "")].append(m)

        for idx, match in enumerate(sorted_matches):
            if match.get("home_score") is None or match.get("away_score") is None:
                continue
            if match.get("status") != "FINISHED":
                continue

            home_name = match["home_team_name"]
            away_name = match["away_team_name"]
            league_code = match.get("league_code", "")
            season = match.get("season", 2025)
            match_date = match.get("match_date", "")

            home_stats = db_manager.get_team_stats(home_name, league_code, season)
            away_stats = db_manager.get_team_stats(away_name, league_code, season)

            if not home_stats:
                home_stats = db_manager.compute_team_stats_from_matches(home_name, league_code, season)
                if home_stats.get("matches_played", 0) < 3:
                    # Update ELO even if we skip training
                    elo_copy.update(home_name, away_name, int(match["home_score"]), int(match["away_score"]))
                    continue
                db_manager.upsert_team_stats(home_stats)

            if not away_stats:
                away_stats = db_manager.compute_team_stats_from_matches(away_name, league_code, season)
                if away_stats.get("matches_played", 0) < 3:
                    elo_copy.update(home_name, away_name, int(match["home_score"]), int(match["away_score"]))
                    continue
                db_manager.upsert_team_stats(away_stats)

            home_stats["team_name"] = home_name
            away_stats["team_name"] = away_name
            h2h = db_manager.get_h2h(home_name, away_name)

            try:
                # Matches up to this point for form/rest/trend computation
                past = sorted_matches[:idx]

                features = cls.build_match_features_v2(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                    ai_predictions=None,
                    elo_tracker=elo_copy,
                    home_form_list=cls.compute_form_list(past, home_name),
                    away_form_list=cls.compute_form_list(past, away_name),
                    home_extra=cls.compute_csv_extra_averages(past, home_name),
                    away_extra=cls.compute_csv_extra_averages(past, away_name),
                    home_days_rest=cls.compute_days_since_last(past, home_name, match_date),
                    away_days_rest=cls.compute_days_since_last(past, away_name, match_date),
                    home_recent_goals_avg=cls.compute_recent_goals_avg(past, home_name),
                    away_recent_goals_avg=cls.compute_recent_goals_avg(past, away_name),
                    is_training=True,
                    league_code=league_code,
                    matchday=match.get("matchday") or 0,
                    total_matchdays=38,
                    match_datetime=match_date,
                    home_sos=cls.compute_sos(past, home_name, elo_copy),
                    away_sos=cls.compute_sos(past, away_name, elo_copy),
                )

                hs = match["home_score"]
                aws = match["away_score"]
                label = 0 if hs > aws else (1 if hs == aws else 2)

                X_list.append(features)
                y_list.append(label)
                date_list.append(match_date)

            except Exception as e:
                logger.error(f"V2 feature engineering error for {home_name} vs {away_name}: {e}")

            # Update ELO AFTER feature extraction (no leakage)
            elo_copy.update(home_name, away_name, int(match["home_score"]), int(match["away_score"]))

        if not X_list:
            logger.warning("No v2 training data could be built")
            return np.empty((0, len(cls.FEATURE_NAMES))), np.empty(0), []

        return np.array(X_list), np.array(y_list), date_list

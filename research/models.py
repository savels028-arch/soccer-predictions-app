"""Reproducible probability model families for the AIBets research lab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import skellam
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from research.calibration import normalize_probabilities


# Teams moving between the two covered English or German divisions retain the
# same model identity.  Every other competition is deliberately isolated so a
# same-named club in another country can never share parameters accidentally.
_DIXON_COLES_COMPETITION_GROUPS = {
    "PL": "ENG",
    "ELC": "ENG",
    "BL1": "GER",
    "BL2": "GER",
    "PD": "ESP",
    "SA": "ITA",
    "FL1": "FRA",
    "DED": "NED",
    "PPL": "POR",
    "BEL1": "BEL",
}

_MIN_GOAL_RATE = 0.05
_MAX_GOAL_RATE = 5.0
_MIN_RHO = -0.19
_MAX_RHO = 0.03


IDENTITY_COLUMNS = {
    "match_id",
    "api_id",
    "natural_key",
    "match_date",
    "league_code",
    "league_name",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "target_1x2",
    "target_1x2_index",
    "target_total_goals",
    "target_over25",
    "target_btts",
}


@dataclass(frozen=True)
class ModelMatrices:
    football_numeric: tuple[str, ...]
    market_numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ("league_code",)


@dataclass(frozen=True)
class _DixonColesGroup:
    """Fitted parameters for one country/competition group."""

    attack: Mapping[str, float]
    defence: Mapping[str, float]
    intercept: float
    home_advantage: float
    rho: float
    fitted_matches: int


def _normalized_team(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.casefold().split())


def _competition_group(league_code: object) -> str:
    code = str(league_code or "").strip().upper()
    return _DIXON_COLES_COMPETITION_GROUPS.get(code, code or "UNKNOWN")


def _weighted_goal_rates(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    *,
    prior_home: float | None = None,
    prior_away: float | None = None,
    prior_strength: float = 0.0,
) -> tuple[float, float]:
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return prior_home or 1.35, prior_away or 1.10
    home_total = float(np.dot(weights, home_goals))
    away_total = float(np.dot(weights, away_goals))
    if prior_home is not None and prior_away is not None and prior_strength > 0.0:
        home_total += prior_strength * prior_home
        away_total += prior_strength * prior_away
        weight_sum += prior_strength
    return (
        float(np.clip(home_total / weight_sum, _MIN_GOAL_RATE, _MAX_GOAL_RATE)),
        float(np.clip(away_total / weight_sum, _MIN_GOAL_RATE, _MAX_GOAL_RATE)),
    )


def dixon_coles_probabilities(
    home_lambda: np.ndarray | Sequence[float] | float,
    away_lambda: np.ndarray | Sequence[float] | float,
    rho: np.ndarray | Sequence[float] | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Dixon--Coles probabilities in research class order.

    The first array is ``[home, draw, away]`` and the second is
    ``[under 2.5, over 2.5]``.  The four-cell low-score correction preserves
    total probability.  Consequently it changes the 1X2 split but, by
    construction, does not change the under/over 2.5 split because all four
    corrected cells are under 2.5 goals.
    """

    home_rate, away_rate, dependence = np.broadcast_arrays(
        np.atleast_1d(np.asarray(home_lambda, dtype=float)),
        np.atleast_1d(np.asarray(away_lambda, dtype=float)),
        np.atleast_1d(np.asarray(rho, dtype=float)),
    )
    if not (
        np.isfinite(home_rate).all()
        and np.isfinite(away_rate).all()
        and np.isfinite(dependence).all()
    ):
        raise ValueError("Dixon-Coles inputs must be finite")
    if np.any(home_rate <= 0.0) or np.any(away_rate <= 0.0):
        raise ValueError("Dixon-Coles goal rates must be positive")
    if np.any((dependence < _MIN_RHO) | (dependence > _MAX_RHO)):
        raise ValueError(f"rho must be between {_MIN_RHO} and {_MAX_RHO}")

    # The Skellam distribution gives the exact independent-Poisson goal
    # difference probabilities, avoiding a truncated score grid.
    away = skellam.cdf(-1, home_rate, away_rate)
    home = 1.0 - skellam.cdf(0, home_rate, away_rate)
    draw = 1.0 - home - away

    # Dixon-Coles' tau adjustment moves the same mass into/out of the two
    # one-goal results and twice that mass out of/into the two low draws.
    delta = home_rate * away_rate * dependence * np.exp(-(home_rate + away_rate))
    one_x_two = np.stack([home + delta, draw - 2.0 * delta, away + delta], axis=-1)
    one_x_two = normalize_probabilities(np.clip(one_x_two, 0.0, None))

    total_rate = home_rate + away_rate
    under = np.exp(-total_rate) * (1.0 + total_rate + 0.5 * total_rate**2)
    over_under = normalize_probabilities(
        np.stack([under, np.clip(1.0 - under, 0.0, 1.0)], axis=-1)
    )
    return one_x_two, over_under


class DixonColesModel:
    """Time-decayed, regularized Dixon--Coles score model.

    This model is intentionally stateless at prediction time: it is fitted on
    the fold's explicit training rows and never learns from calibration,
    selection, test, or prediction rows.  Attack and defence parameters are
    mean-centred for identification and ridge-shrunk so sparse teams regress
    toward their competition average.  An unseen team therefore has neutral
    attack/defence rather than inheriting information from the future.
    """

    def __init__(
        self,
        *,
        half_life_days: float = 365.0,
        ridge: float = 0.10,
        min_group_matches: int = 8,
        max_iter: int = 180,
    ) -> None:
        if half_life_days <= 0.0:
            raise ValueError("half_life_days must be positive")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if min_group_matches < 1:
            raise ValueError("min_group_matches must be positive")
        self.half_life_days = float(half_life_days)
        self.ridge = float(ridge)
        self.min_group_matches = int(min_group_matches)
        self.max_iter = int(max_iter)
        self.groups_: Dict[str, _DixonColesGroup] = {}
        self.global_intercept_: float | None = None
        self.global_home_advantage_: float | None = None
        self.training_cutoff_: pd.Timestamp | None = None

    def _clean_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "match_date",
            "league_code",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing Dixon-Coles columns: {', '.join(missing)}")
        clean = frame.loc[:, sorted(required)].copy()
        clean["match_date"] = pd.to_datetime(clean["match_date"], errors="coerce", utc=True)
        clean["home_score"] = pd.to_numeric(clean["home_score"], errors="coerce")
        clean["away_score"] = pd.to_numeric(clean["away_score"], errors="coerce")
        clean["home_team"] = clean["home_team"].map(_normalized_team)
        clean["away_team"] = clean["away_team"].map(_normalized_team)
        clean["competition_group"] = clean["league_code"].map(_competition_group)
        valid = (
            clean["match_date"].notna()
            & clean["home_score"].notna()
            & clean["away_score"].notna()
            & (clean["home_score"] >= 0)
            & (clean["away_score"] >= 0)
            & clean["home_team"].ne("")
            & clean["away_team"].ne("")
            & clean["home_team"].ne(clean["away_team"])
        )
        clean = clean.loc[valid].copy()
        if clean.empty:
            raise ValueError("no valid scored training rows for Dixon-Coles model")
        clean["home_score"] = clean["home_score"].astype(int)
        clean["away_score"] = clean["away_score"].astype(int)
        return clean.sort_values(
            ["match_date", "competition_group", "home_team", "away_team"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        assert self.training_cutoff_ is not None
        age_days = (
            self.training_cutoff_ - pd.to_datetime(dates, utc=True)
        ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
        # Negative ages would mean an inconsistent cutoff; clipping keeps the
        # latest training row at weight one without ever upweighting a row.
        return np.exp(-math.log(2.0) * np.maximum(age_days, 0.0) / self.half_life_days)

    def _fallback_group(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> _DixonColesGroup:
        assert self.global_intercept_ is not None
        assert self.global_home_advantage_ is not None
        global_away = math.exp(self.global_intercept_)
        global_home = math.exp(self.global_intercept_ + self.global_home_advantage_)
        home_rate, away_rate = _weighted_goal_rates(
            home_goals,
            away_goals,
            weights,
            prior_home=global_home,
            prior_away=global_away,
            prior_strength=20.0,
        )
        return _DixonColesGroup(
            attack={},
            defence={},
            intercept=math.log(away_rate),
            home_advantage=math.log(home_rate / away_rate),
            rho=0.0,
            fitted_matches=len(home_goals),
        )

    def _fit_group(self, group: pd.DataFrame) -> _DixonColesGroup:
        home_goals = group["home_score"].to_numpy(dtype=float)
        away_goals = group["away_score"].to_numpy(dtype=float)
        weights = self._time_weights(group["match_date"])
        fallback = self._fallback_group(home_goals, away_goals, weights)
        teams = sorted(set(group["home_team"]).union(group["away_team"]))
        if len(group) < self.min_group_matches or len(teams) < 2 or weights.sum() <= 2.0:
            return fallback

        team_index = {team: index for index, team in enumerate(teams)}
        home_index = group["home_team"].map(team_index).to_numpy(dtype=int)
        away_index = group["away_team"].map(team_index).to_numpy(dtype=int)
        n_teams = len(teams)
        normalizer = float(weights.sum())
        lower_eta = math.log(_MIN_GOAL_RATE)
        upper_eta = math.log(_MAX_GOAL_RATE)
        log_factorials = gammaln(home_goals + 1.0) + gammaln(away_goals + 1.0)

        initial = np.zeros(2 * n_teams + 3, dtype=float)
        initial[2 * n_teams] = fallback.intercept
        initial[2 * n_teams + 1] = fallback.home_advantage
        initial[2 * n_teams + 2] = -0.05

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            attack_raw = parameters[:n_teams]
            defence_raw = parameters[n_teams : 2 * n_teams]
            attack = attack_raw - attack_raw.mean()
            defence = defence_raw - defence_raw.mean()
            intercept = parameters[2 * n_teams]
            home_advantage = parameters[2 * n_teams + 1]
            rho = parameters[2 * n_teams + 2]

            eta_home_raw = (
                intercept + home_advantage + attack[home_index] + defence[away_index]
            )
            eta_away_raw = intercept + attack[away_index] + defence[home_index]
            eta_home = np.clip(eta_home_raw, lower_eta, upper_eta)
            eta_away = np.clip(eta_away_raw, lower_eta, upper_eta)
            home_rate = np.exp(eta_home)
            away_rate = np.exp(eta_away)

            log_tau = np.zeros(len(group), dtype=float)
            tau_home = np.zeros(len(group), dtype=float)
            tau_away = np.zeros(len(group), dtype=float)
            tau_rho = np.zeros(len(group), dtype=float)

            score_00 = (home_goals == 0.0) & (away_goals == 0.0)
            score_01 = (home_goals == 0.0) & (away_goals == 1.0)
            score_10 = (home_goals == 1.0) & (away_goals == 0.0)
            score_11 = (home_goals == 1.0) & (away_goals == 1.0)

            if score_00.any():
                product = home_rate[score_00] * away_rate[score_00]
                tau = 1.0 - product * rho
                log_tau[score_00] = np.log(tau)
                adjustment = -product * rho / tau
                tau_home[score_00] = adjustment
                tau_away[score_00] = adjustment
                tau_rho[score_00] = -product / tau
            if score_01.any():
                rate = home_rate[score_01]
                tau = 1.0 + rate * rho
                log_tau[score_01] = np.log(tau)
                tau_home[score_01] = rate * rho / tau
                tau_rho[score_01] = rate / tau
            if score_10.any():
                rate = away_rate[score_10]
                tau = 1.0 + rate * rho
                log_tau[score_10] = np.log(tau)
                tau_away[score_10] = rate * rho / tau
                tau_rho[score_10] = rate / tau
            if score_11.any():
                tau = 1.0 - rho
                log_tau[score_11] = math.log(tau)
                tau_rho[score_11] = -1.0 / tau

            log_likelihood = (
                home_goals * eta_home
                - home_rate
                + away_goals * eta_away
                - away_rate
                - log_factorials
                + log_tau
            )
            loss = -float(np.dot(weights, log_likelihood)) / normalizer
            loss += 0.5 * self.ridge * (
                float(np.mean(attack**2)) + float(np.mean(defence**2))
            )

            scale = -weights / normalizer
            grad_eta_home = scale * (home_goals - home_rate + tau_home)
            grad_eta_away = scale * (away_goals - away_rate + tau_away)
            grad_eta_home *= (eta_home_raw > lower_eta) & (eta_home_raw < upper_eta)
            grad_eta_away *= (eta_away_raw > lower_eta) & (eta_away_raw < upper_eta)

            grad_attack_centered = np.zeros(n_teams, dtype=float)
            grad_defence_centered = np.zeros(n_teams, dtype=float)
            np.add.at(grad_attack_centered, home_index, grad_eta_home)
            np.add.at(grad_attack_centered, away_index, grad_eta_away)
            np.add.at(grad_defence_centered, away_index, grad_eta_home)
            np.add.at(grad_defence_centered, home_index, grad_eta_away)
            grad_attack_centered += self.ridge * attack / n_teams
            grad_defence_centered += self.ridge * defence / n_teams

            gradient = np.empty_like(parameters)
            gradient[:n_teams] = grad_attack_centered - grad_attack_centered.mean()
            gradient[n_teams : 2 * n_teams] = (
                grad_defence_centered - grad_defence_centered.mean()
            )
            gradient[2 * n_teams] = float(grad_eta_home.sum() + grad_eta_away.sum())
            gradient[2 * n_teams + 1] = float(grad_eta_home.sum())
            gradient[2 * n_teams + 2] = float(np.dot(scale, tau_rho))
            return loss, gradient

        bounds = (
            [(-1.5, 1.5)] * (2 * n_teams)
            + [(-1.5, 1.0), (-0.5, 1.0), (_MIN_RHO, _MAX_RHO)]
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            return fallback

        fitted = result.x
        attack = fitted[:n_teams] - fitted[:n_teams].mean()
        defence = fitted[n_teams : 2 * n_teams]
        defence = defence - defence.mean()
        return _DixonColesGroup(
            attack={team: float(attack[index]) for team, index in team_index.items()},
            defence={team: float(defence[index]) for team, index in team_index.items()},
            intercept=float(fitted[2 * n_teams]),
            home_advantage=float(fitted[2 * n_teams + 1]),
            rho=float(fitted[2 * n_teams + 2]),
            fitted_matches=len(group),
        )

    def fit(self, frame: pd.DataFrame) -> "DixonColesModel":
        clean = self._clean_training_frame(frame)
        self.training_cutoff_ = pd.Timestamp(clean["match_date"].max())
        weights = self._time_weights(clean["match_date"])
        home_rate, away_rate = _weighted_goal_rates(
            clean["home_score"].to_numpy(dtype=float),
            clean["away_score"].to_numpy(dtype=float),
            weights,
        )
        self.global_intercept_ = math.log(away_rate)
        self.global_home_advantage_ = math.log(home_rate / away_rate)
        self.groups_ = {
            str(group_name): self._fit_group(group)
            for group_name, group in clean.groupby("competition_group", sort=True)
        }
        return self

    def _require_fitted(self) -> None:
        if self.global_intercept_ is None or self.global_home_advantage_ is None:
            raise RuntimeError("DixonColesModel must be fitted before prediction")

    def predict_score_rates(self, frame: pd.DataFrame) -> np.ndarray:
        """Return pre-match ``[home_lambda, away_lambda]`` score rates."""

        self._require_fitted()
        required = {"match_date", "league_code", "home_team", "away_team"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing Dixon-Coles columns: {', '.join(missing)}")
        prediction_dates = pd.to_datetime(frame["match_date"], errors="coerce", utc=True)
        if prediction_dates.isna().any():
            raise ValueError("Dixon-Coles prediction dates must be parseable")
        assert self.training_cutoff_ is not None
        if (prediction_dates <= self.training_cutoff_).any():
            raise ValueError(
                "Dixon-Coles prediction rows must be strictly after the training cutoff"
            )
        assert self.global_intercept_ is not None
        assert self.global_home_advantage_ is not None
        rates = np.empty((len(frame), 2), dtype=float)
        for output_index, (_, row) in enumerate(frame.iterrows()):
            group_name = _competition_group(row["league_code"])
            fitted = self.groups_.get(group_name)
            if fitted is None:
                intercept = self.global_intercept_
                home_advantage = self.global_home_advantage_
                home_attack = away_attack = home_defence = away_defence = 0.0
            else:
                intercept = fitted.intercept
                home_advantage = fitted.home_advantage
                home_team = _normalized_team(row["home_team"])
                away_team = _normalized_team(row["away_team"])
                home_attack = fitted.attack.get(home_team, 0.0)
                away_attack = fitted.attack.get(away_team, 0.0)
                home_defence = fitted.defence.get(home_team, 0.0)
                away_defence = fitted.defence.get(away_team, 0.0)
            rates[output_index, 0] = np.clip(
                math.exp(intercept + home_advantage + home_attack + away_defence),
                _MIN_GOAL_RATE,
                _MAX_GOAL_RATE,
            )
            rates[output_index, 1] = np.clip(
                math.exp(intercept + away_attack + home_defence),
                _MIN_GOAL_RATE,
                _MAX_GOAL_RATE,
            )
        return rates

    def predict_proba(self, frame: pd.DataFrame, market: str) -> np.ndarray:
        self._require_fitted()
        rates = self.predict_score_rates(frame)
        rho = np.array(
            [
                fitted.rho if (fitted := self.groups_.get(_competition_group(league))) else 0.0
                for league in frame["league_code"]
            ],
            dtype=float,
        )
        one_x_two, over_under = dixon_coles_probabilities(rates[:, 0], rates[:, 1], rho)
        if market == "1x2":
            return one_x_two
        if market == "ou25":
            return over_under
        raise ValueError(f"unsupported market {market!r}")


def infer_feature_columns(frame: pd.DataFrame, market: str) -> ModelMatrices:
    """Infer pre-match numeric features while excluding outcomes and prices."""

    numeric = []
    for column in frame.columns:
        if column in IDENTITY_COLUMNS or column.startswith("odds_") or column.startswith("market_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)

    if market == "1x2":
        market_columns = (
            "market_1x2_avg_home_prob",
            "market_1x2_avg_draw_prob",
            "market_1x2_avg_away_prob",
            "market_1x2_primary_home_prob",
            "market_1x2_primary_draw_prob",
            "market_1x2_primary_away_prob",
        )
    elif market == "ou25":
        market_columns = (
            "market_ou25_avg_under25_prob",
            "market_ou25_avg_over25_prob",
            "market_ou25_primary_under25_prob",
            "market_ou25_primary_over25_prob",
        )
    else:
        raise ValueError(f"unsupported market {market!r}")
    available_market = tuple(column for column in market_columns if column in frame)
    return ModelMatrices(
        football_numeric=tuple(numeric),
        market_numeric=tuple(numeric) + available_market,
    )


def _logistic_pipeline(columns: ModelMatrices, use_market: bool, random_state: int) -> Pipeline:
    numeric = list(columns.market_numeric if use_market else columns.football_numeric)
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "league",
                OneHotEncoder(handle_unknown="ignore"),
                list(columns.categorical),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    max_iter=400,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _boosting_pipeline(columns: ModelMatrices, use_market: bool, random_state: int) -> Pipeline:
    numeric = list(columns.market_numeric if use_market else columns.football_numeric)
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "league",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                list(columns.categorical),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=140,
                    max_leaf_nodes=15,
                    min_samples_leaf=80,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _columns(frame: pd.DataFrame, names: Sequence[str], default: float) -> np.ndarray:
    arrays = []
    for name in names:
        if name in frame:
            arrays.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            arrays.append(np.full(len(frame), default, dtype=float))
    return np.column_stack(arrays)


def baseline_probabilities(frame: pd.DataFrame, market: str) -> Dict[str, np.ndarray]:
    """Return market, Poisson, league-prior and Elo probability baselines."""

    if market == "1x2":
        market_avg = _columns(
            frame,
            [
                "market_1x2_avg_home_prob",
                "market_1x2_avg_draw_prob",
                "market_1x2_avg_away_prob",
            ],
            np.nan,
        )
        market_primary = _columns(
            frame,
            [
                "market_1x2_primary_home_prob",
                "market_1x2_primary_draw_prob",
                "market_1x2_primary_away_prob",
            ],
            np.nan,
        )
        market = np.where(np.isfinite(market_avg).all(axis=1, keepdims=True), market_avg, market_primary)
        market = np.where(np.isfinite(market), market, 1.0 / 3.0)
        poisson = _columns(
            frame,
            ["poisson_home_prob", "poisson_draw_prob", "poisson_away_prob"],
            1.0 / 3.0,
        )
        league = _columns(
            frame,
            ["league_home_win_rate", "league_draw_rate", "league_away_win_rate"],
            1.0 / 3.0,
        )
        expected_home = _columns(frame, ["elo_expected_home_score"], 0.5)[:, 0]
        draw = np.clip(_columns(frame, ["league_draw_rate"], 0.27)[:, 0], 0.14, 0.34)
        elo = np.column_stack(
            [expected_home * (1.0 - draw), draw, (1.0 - expected_home) * (1.0 - draw)]
        )
        return {
            "market": normalize_probabilities(market),
            "poisson": normalize_probabilities(poisson),
            "league_prior": normalize_probabilities(league),
            "elo": normalize_probabilities(elo),
        }

    if market == "ou25":
        # Feature quotes store over then under; research labels use
        # class 0=under and class 1=over.
        market_avg = _columns(
            frame,
            ["market_ou25_avg_under25_prob", "market_ou25_avg_over25_prob"],
            np.nan,
        )
        market_primary = _columns(
            frame,
            ["market_ou25_primary_under25_prob", "market_ou25_primary_over25_prob"],
            np.nan,
        )
        market = np.where(np.isfinite(market_avg).all(axis=1, keepdims=True), market_avg, market_primary)
        market = np.where(np.isfinite(market), market, 0.5)
        poisson = _columns(frame, ["poisson_under25_prob", "poisson_over25_prob"], 0.5)
        over = _columns(frame, ["league_over25_rate"], 0.5)[:, 0]
        league = np.column_stack([1.0 - over, over])
        return {
            "market": normalize_probabilities(market),
            "poisson": normalize_probabilities(poisson),
            "league_prior": normalize_probabilities(league),
        }
    raise ValueError(f"unsupported market {market!r}")


def fit_probability_families(
    frame: pd.DataFrame,
    market: str,
    train_positions: np.ndarray,
    predict_positions: Mapping[str, np.ndarray],
    *,
    random_state: int = 20260714,
    include_boosting: bool = True,
    dixon_coles_model: DixonColesModel | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit model families once and predict each chronological target slice."""

    target_column = "target_1x2_index" if market == "1x2" else "target_over25"
    y_train = frame.iloc[train_positions][target_column].to_numpy(dtype=int)
    if len(np.unique(y_train)) < (3 if market == "1x2" else 2):
        raise ValueError("training slice does not contain every market class")
    feature_columns = infer_feature_columns(frame, market)
    pipelines = {
        "logistic_football": _logistic_pipeline(feature_columns, False, random_state),
        "logistic_market": _logistic_pipeline(feature_columns, True, random_state),
    }
    if include_boosting:
        pipelines.update(
            {
                "boosting_football": _boosting_pipeline(feature_columns, False, random_state),
                "boosting_market": _boosting_pipeline(feature_columns, True, random_state),
            }
        )

    trained = {}
    train_frame = frame.iloc[train_positions]
    for name, pipeline in pipelines.items():
        pipeline.fit(train_frame, y_train)
        trained[name] = pipeline
    if dixon_coles_model is None:
        dixon_coles = DixonColesModel().fit(train_frame)
    else:
        dixon_coles_model._require_fitted()
        expected_cutoff = pd.to_datetime(
            train_frame["match_date"], errors="coerce", utc=True
        ).max()
        if pd.isna(expected_cutoff) or dixon_coles_model.training_cutoff_ != expected_cutoff:
            raise ValueError("shared Dixon-Coles model does not match the training cutoff")
        dixon_coles = dixon_coles_model

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for slice_name, row_positions in predict_positions.items():
        target_frame = frame.iloc[row_positions]
        families = baseline_probabilities(target_frame, market)
        for name, pipeline in trained.items():
            families[name] = normalize_probabilities(pipeline.predict_proba(target_frame))
        families["dixon_coles"] = dixon_coles.predict_proba(target_frame, market)

        market_probs = families["market"]
        blend_sources = {
            "poisson",
            "dixon_coles",
            "elo",
            "logistic_market",
            "boosting_market",
        }
        for name in [key for key in families if key in blend_sources]:
            model_probs = families[name]
            for market_weight in (0.25, 0.50, 0.75):
                label = int(market_weight * 100)
                families[f"blend_{name}_market{label}"] = normalize_probabilities(
                    (1.0 - market_weight) * model_probs + market_weight * market_probs
                )
        predictions[slice_name] = families
    return predictions


def odds_matrices(frame: pd.DataFrame, market: str) -> Dict[str, np.ndarray]:
    """Return complete quote matrices; missing quotes remain NaN and unbettable."""

    if market == "1x2":
        outcomes = ("home", "draw", "away")
        bases = ("primary", "b365", "avg", "max", "close")
    elif market == "ou25":
        outcomes = ("under25", "over25")
        bases = ("primary", "b365", "pinnacle", "avg", "max", "close")
    else:
        raise ValueError(f"unsupported market {market!r}")
    result = {}
    for basis in bases:
        names = [f"odds_{market}_{basis}_{outcome}" for outcome in outcomes]
        result[basis] = _columns(frame, names, np.nan)
    return result


__all__ = [
    "DixonColesModel",
    "ModelMatrices",
    "baseline_probabilities",
    "dixon_coles_probabilities",
    "fit_probability_families",
    "infer_feature_columns",
    "odds_matrices",
]

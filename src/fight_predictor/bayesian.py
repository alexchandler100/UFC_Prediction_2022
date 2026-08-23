"""Paper-only Bayesian logistic challenger for winner probabilities.

The production winner model is an L2-regularized, zero-intercept logistic
regression.  Its fitted coefficients are the MAP estimate under independent
zero-mean Gaussian priors.  This module keeps that exact point estimate and
uses the inverse penalized-likelihood Hessian as a Laplace approximation to
the coefficient posterior.  A matchup therefore has a normal posterior on
the calibrated logit and a logit-normal posterior on win probability.

This is deliberately a challenger, not an execution policy.  The artifact
records the approximation, fixed calibration treatment, chronological
comparison, and a disabled evidence gate so downstream consumers cannot
mistake posterior uncertainty for validated betting profitability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from .point_in_time import PIT_SORT_COLUMNS, TemporalFightPredictor, _metrics


BAYESIAN_SCHEMA_VERSION = 1
BAYESIAN_MODEL_VERSION = "point-in-time-bayesian-logistic-laplace-v1"
BAYESIAN_CREDIBLE_LEVEL = 0.90
BAYESIAN_MINIMUM_MEAN_EV = 0.05
BAYESIAN_MINIMUM_PROBABILITY_POSITIVE_EV = 0.80
BAYESIAN_MAX_LOG_LOSS_INCREASE = 0.002
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.hermite.hermgauss(24)
_STANDARD_NORMAL = NormalDist()


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -709.0, 709.0)))


def _logit(probability: float) -> float:
    bounded = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return math.log(bounded / (1.0 - bounded))


def _logit_normal_mean(location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Deterministic Gauss-Hermite expectation of sigmoid(N(location, scale))."""

    locations = np.asarray(location, dtype=float).reshape(-1, 1)
    scales = np.asarray(scale, dtype=float).reshape(-1, 1)
    logits = locations + math.sqrt(2.0) * scales * _QUADRATURE_NODES.reshape(1, -1)
    return (
        _sigmoid(logits) @ _QUADRATURE_WEIGHTS.reshape(-1, 1)
    ).reshape(-1) / math.sqrt(math.pi)


def american_to_decimal(moneyline: object) -> float:
    try:
        value = float(moneyline)
    except (TypeError, ValueError) as error:
        raise ValueError("moneyline must be numeric") from error
    if not math.isfinite(value) or value == 0 or abs(value) < 100:
        raise ValueError("moneyline must be a finite American price with magnitude >= 100")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


@dataclass(frozen=True)
class LogitNormalPrediction:
    """Bounded summary of one posterior probability distribution."""

    posterior_mean_probability: float
    posterior_median_probability: float
    lower_probability: float
    upper_probability: float
    calibrated_logit_location: float
    calibrated_logit_scale: float
    credible_level: float = BAYESIAN_CREDIBLE_LEVEL

    def __post_init__(self) -> None:
        numeric = np.asarray(
            [
                self.posterior_mean_probability,
                self.posterior_median_probability,
                self.lower_probability,
                self.upper_probability,
                self.calibrated_logit_location,
                self.calibrated_logit_scale,
                self.credible_level,
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("posterior prediction contains non-finite values")
        if not (
            0.0 < self.lower_probability
            <= self.posterior_median_probability
            <= self.upper_probability
            < 1.0
        ):
            raise ValueError("posterior probability interval is invalid")
        if not 0.0 < self.posterior_mean_probability < 1.0:
            raise ValueError("posterior mean probability must be strictly bounded")
        if self.calibrated_logit_scale < 0.0:
            raise ValueError("posterior logit scale cannot be negative")
        if not 0.0 < self.credible_level < 1.0:
            raise ValueError("credible_level must be strictly between zero and one")

    def complement(self) -> "LogitNormalPrediction":
        return LogitNormalPrediction(
            posterior_mean_probability=1.0 - self.posterior_mean_probability,
            posterior_median_probability=1.0 - self.posterior_median_probability,
            lower_probability=1.0 - self.upper_probability,
            upper_probability=1.0 - self.lower_probability,
            calibrated_logit_location=-self.calibrated_logit_location,
            calibrated_logit_scale=self.calibrated_logit_scale,
            credible_level=self.credible_level,
        )

    def probability_above(self, threshold: float) -> float:
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("probability threshold must be strictly between zero and one")
        if self.calibrated_logit_scale == 0.0:
            return float(self.posterior_median_probability > float(threshold))
        z_score = (
            _logit(float(threshold)) - self.calibrated_logit_location
        ) / self.calibrated_logit_scale
        return float(1.0 - _STANDARD_NORMAL.cdf(z_score))

    def expected_return(self, moneyline: object) -> dict[str, float]:
        decimal = american_to_decimal(moneyline)
        break_even = 1.0 / decimal
        return {
            "offered_decimal_odds": decimal,
            "break_even_probability": break_even,
            "posterior_mean_expected_return": (
                decimal * self.posterior_mean_probability - 1.0
            ),
            "lower_expected_return": decimal * self.lower_probability - 1.0,
            "upper_expected_return": decimal * self.upper_probability - 1.0,
            "probability_positive_expected_return": self.probability_above(
                break_even
            ),
        }

    def to_mapping(self) -> dict[str, float]:
        return asdict(self)


def _scaled_matrix(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    scale: np.ndarray,
) -> np.ndarray:
    columns = list(feature_columns)
    values = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if values.shape[1] != len(scale):
        raise ValueError("feature matrix width differs from scaler")
    return values / scale


def laplace_covariance(
    scaled_features: np.ndarray,
    coefficients: np.ndarray,
    c_value: float,
) -> np.ndarray:
    """Return inverse Hessian for likelihood plus N(0, C I) prior."""

    design = np.asarray(scaled_features, dtype=float)
    location = np.asarray(coefficients, dtype=float)
    if design.ndim != 2 or location.shape != (design.shape[1],):
        raise ValueError("Laplace design and coefficient dimensions disagree")
    if not np.isfinite(design).all() or not np.isfinite(location).all():
        raise ValueError("Laplace inputs must be finite")
    if not math.isfinite(float(c_value)) or float(c_value) <= 0.0:
        raise ValueError("c_value must be finite and positive")
    fitted = _sigmoid(design @ location)
    weights = fitted * (1.0 - fitted)
    precision = design.T @ (design * weights.reshape(-1, 1))
    precision.flat[:: precision.shape[0] + 1] += 1.0 / float(c_value)
    try:
        factor = np.linalg.cholesky(precision)
        inverse_factor = np.linalg.solve(factor, np.eye(factor.shape[0]))
    except np.linalg.LinAlgError as error:
        raise ValueError("Laplace posterior precision is not positive definite") from error
    covariance = inverse_factor.T @ inverse_factor
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(covariance).all():
        raise ValueError("Laplace posterior covariance is non-finite")
    return covariance


def _posterior_predictions(
    scaled_features: np.ndarray,
    coefficients: np.ndarray,
    covariance: np.ndarray,
    calibration_slope: float,
    *,
    credible_level: float = BAYESIAN_CREDIBLE_LEVEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design = np.asarray(scaled_features, dtype=float)
    raw_location = design @ np.asarray(coefficients, dtype=float)
    raw_variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    raw_scale = np.sqrt(np.maximum(raw_variance, 0.0))
    location = float(calibration_slope) * raw_location
    scale = abs(float(calibration_slope)) * raw_scale
    tail = (1.0 - float(credible_level)) / 2.0
    z_value = _STANDARD_NORMAL.inv_cdf(1.0 - tail)
    mean = _logit_normal_mean(location, scale)
    median = _sigmoid(location)
    lower = _sigmoid(location - z_value * scale)
    upper = _sigmoid(location + z_value * scale)
    return mean, median, lower, upper, scale


class BayesianLogisticChallenger:
    """Portable Laplace posterior tied to one production winner model."""

    def __init__(
        self,
        *,
        builder,
        base_artifact: dict[str, object],
        covariance: np.ndarray,
        temporal_evaluation: dict[str, object],
        loaded_artifact: dict[str, object] | None = None,
    ):
        self.builder = builder
        self.base_artifact = json.loads(json.dumps(base_artifact))
        self.feature_columns = list(base_artifact["feature_columns"])
        self.scaler_scale = np.asarray(base_artifact["scaler_scale"], dtype=float)
        self.coefficients = np.asarray(base_artifact["coefficients"], dtype=float)
        self.calibration_slope = float(base_artifact["calibration_slope"])
        self.selected_c = float(base_artifact["selected_c"])
        self.covariance = np.asarray(covariance, dtype=float)
        self.temporal_evaluation = json.loads(json.dumps(temporal_evaluation))
        self._loaded_artifact = loaded_artifact

    @staticmethod
    def _fit_fold(
        train: pd.DataFrame,
        test: pd.DataFrame,
        feature_columns: list[str],
        c_value: float,
        calibration_slope: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pipeline = TemporalFightPredictor._fit_pipeline(
            train[feature_columns], train["target"], c_value
        )
        imputer, scaler, model = pipeline
        train_scaled = scaler.transform(imputer.transform(train[feature_columns]))
        covariance = laplace_covariance(
            train_scaled, np.asarray(model.coef_[0], dtype=float), c_value
        )
        test_scaled = scaler.transform(imputer.transform(test[feature_columns]))
        mean, _, lower, upper, _ = _posterior_predictions(
            test_scaled,
            np.asarray(model.coef_[0], dtype=float),
            covariance,
            calibration_slope,
        )
        return mean, lower, upper

    @classmethod
    def _chronological_evaluation(
        cls,
        predictor: TemporalFightPredictor,
    ) -> dict[str, object]:
        point_evaluation = predictor.evaluation
        required = {"calibrated_model", "selected_c", "evaluation_calibration_slope", "walk_forward"}
        if not required.issubset(point_evaluation):
            return {"status": "unavailable_for_fixture"}
        feature_columns = list(predictor.feature_columns)
        frame = predictor.training_data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        row_cutoff = max(1, int(len(frame) * 0.8))
        holdout_date = frame.iloc[row_cutoff]["date"]
        development = frame[frame["date"] < holdout_date].reset_index(drop=True)
        holdout = frame[frame["date"] >= holdout_date].reset_index(drop=True)
        holdout_mean, holdout_lower, holdout_upper = cls._fit_fold(
            development,
            holdout,
            feature_columns,
            float(point_evaluation["selected_c"]),
            float(point_evaluation["evaluation_calibration_slope"]),
        )
        holdout_metrics = {
            **_metrics(holdout["target"], holdout_mean),
            "mean_90_probability_interval_width": float(
                np.mean(holdout_upper - holdout_lower)
            ),
        }

        full = predictor.point_in_time_data.copy()
        full["date"] = pd.to_datetime(full["date"], errors="raise")
        fold_contracts = point_evaluation["walk_forward"].get("folds", {})
        fold_metrics: dict[str, object] = {}
        all_truth: list[np.ndarray] = []
        all_probability: list[np.ndarray] = []
        all_widths: list[np.ndarray] = []
        for year_text in sorted(fold_contracts, key=int):
            contract = fold_contracts[year_text]
            year = int(year_text)
            test_start = pd.Timestamp(year=year, month=1, day=1)
            train_start = test_start - pd.DateOffset(years=10)
            train = full[
                (full["date"] >= train_start) & (full["date"] < test_start)
            ].reset_index(drop=True)
            test = full[full["date"].dt.year == year].reset_index(drop=True)
            if train.empty or test.empty:
                continue
            mean, lower, upper = cls._fit_fold(
                train,
                test,
                feature_columns,
                float(contract["selected_c"]),
                float(contract["calibration_slope"]),
            )
            widths = upper - lower
            all_truth.append(test["target"].to_numpy(dtype=int))
            all_probability.append(mean)
            all_widths.append(widths)
            fold_metrics[year_text] = {
                **_metrics(test["target"], mean),
                "mean_90_probability_interval_width": float(np.mean(widths)),
                "selected_c": float(contract["selected_c"]),
                "calibration_slope": float(contract["calibration_slope"]),
            }
        if not all_truth:
            raise ValueError("Bayesian challenger has no eligible walk-forward folds")
        aggregate = {
            **_metrics(np.concatenate(all_truth), np.concatenate(all_probability)),
            "mean_90_probability_interval_width": float(
                np.mean(np.concatenate(all_widths))
            ),
        }
        point_holdout = point_evaluation["calibrated_model"]
        point_walk = point_evaluation["walk_forward"]["aggregate"]
        comparison = {
            "holdout_log_loss_delta_vs_point": float(
                holdout_metrics["log_loss"] - point_holdout["log_loss"]
            ),
            "holdout_brier_delta_vs_point": float(
                holdout_metrics["brier"] - point_holdout["brier"]
            ),
            "walk_forward_log_loss_delta_vs_point": float(
                aggregate["log_loss"] - point_walk["log_loss"]
            ),
            "walk_forward_brier_delta_vs_point": float(
                aggregate["brier"] - point_walk["brier"]
            ),
        }
        noninferior = (
            comparison["walk_forward_log_loss_delta_vs_point"]
            <= BAYESIAN_MAX_LOG_LOSS_INCREASE
        )
        return {
            "status": "evaluated_chronologically",
            "credible_level": BAYESIAN_CREDIBLE_LEVEL,
            "holdout": holdout_metrics,
            "walk_forward": {"folds": fold_metrics, "aggregate": aggregate},
            "comparison_to_point_model": comparison,
            "evidence_gate": {
                "status": (
                    "eligible_for_prospective_paper_tracking"
                    if noninferior
                    else "failed_predictive_noninferiority"
                ),
                "maximum_allowed_walk_forward_log_loss_increase": (
                    BAYESIAN_MAX_LOG_LOSS_INCREASE
                ),
                "predictive_noninferiority_met": noninferior,
                "minimum_mean_expected_return": BAYESIAN_MINIMUM_MEAN_EV,
                "minimum_probability_positive_expected_return": (
                    BAYESIAN_MINIMUM_PROBABILITY_POSITIVE_EV
                ),
                "prospective_clv_requirement_met": False,
                "prospective_return_requirement_met": False,
                "execution_enabled": False,
            },
        }

    @classmethod
    def fit(
        cls,
        predictor: TemporalFightPredictor,
        *,
        evaluate: bool = True,
    ) -> "BayesianLogisticChallenger":
        base = predictor.artifact()
        design = _scaled_matrix(
            predictor.training_data,
            base["feature_columns"],
            np.asarray(base["scaler_scale"], dtype=float),
        )
        covariance = laplace_covariance(
            design,
            np.asarray(base["coefficients"], dtype=float),
            float(base["selected_c"]),
        )
        evaluation = (
            cls._chronological_evaluation(predictor)
            if evaluate
            else {"status": "not_requested"}
        )
        return cls(
            builder=predictor.builder,
            base_artifact=base,
            covariance=covariance,
            temporal_evaluation=evaluation,
        )

    def prediction(self, diff_row: pd.DataFrame) -> LogitNormalPrediction:
        design = _scaled_matrix(diff_row, self.feature_columns, self.scaler_scale)
        if len(design) != 1:
            raise ValueError("one matchup row is required for posterior prediction")
        mean, median, lower, upper, scale = _posterior_predictions(
            design,
            self.coefficients,
            self.covariance,
            self.calibration_slope,
        )
        location = float(self.calibration_slope * (design @ self.coefficients)[0])
        return LogitNormalPrediction(
            posterior_mean_probability=float(mean[0]),
            posterior_median_probability=float(median[0]),
            lower_probability=float(lower[0]),
            upper_probability=float(upper[0]),
            calibrated_logit_location=location,
            calibrated_logit_scale=float(scale[0]),
        )

    def annotate_upcoming_fights(
        self,
        frame: pd.DataFrame,
        card_date: str,
    ) -> pd.DataFrame:
        output = frame.copy(deep=True)
        columns = {
            "bayesian model id": "",
            "bayesian model version": BAYESIAN_MODEL_VERSION,
            "bayesian posterior mean": np.nan,
            "bayesian posterior median": np.nan,
            "bayesian probability lower": np.nan,
            "bayesian probability upper": np.nan,
            "bayesian credible level": BAYESIAN_CREDIBLE_LEVEL,
            "bayesian calibrated logit location": np.nan,
            "bayesian calibrated logit scale": np.nan,
            "bayesian status": "abstain_unresolved_identity",
        }
        for column, default in columns.items():
            if column not in output:
                output[column] = default
        model_id = self.artifact()["model_id"]
        for index, row in output.iterrows():
            fighter_id = str(row.get("fighter id") or "").strip()
            opponent_id = str(row.get("opponent id") or "").strip()
            if not fighter_id or not opponent_id or fighter_id == opponent_id:
                continue
            division = str(row.get("division") or "Unknown")
            features = self.builder.matchup_features(
                fighter_id, opponent_id, card_date, division
            )
            prediction = self.prediction(features)
            output.at[index, "bayesian model id"] = model_id
            output.at[index, "bayesian model version"] = BAYESIAN_MODEL_VERSION
            output.at[index, "bayesian posterior mean"] = (
                prediction.posterior_mean_probability
            )
            output.at[index, "bayesian posterior median"] = (
                prediction.posterior_median_probability
            )
            output.at[index, "bayesian probability lower"] = (
                prediction.lower_probability
            )
            output.at[index, "bayesian probability upper"] = (
                prediction.upper_probability
            )
            output.at[index, "bayesian credible level"] = prediction.credible_level
            output.at[index, "bayesian calibrated logit location"] = (
                prediction.calibrated_logit_location
            )
            output.at[index, "bayesian calibrated logit scale"] = (
                prediction.calibrated_logit_scale
            )
            try:
                minimum_history = min(
                    int(float(row.get("fighter prior fights"))),
                    int(float(row.get("opponent prior fights"))),
                )
            except (TypeError, ValueError):
                minimum_history = 0
            output.at[index, "bayesian status"] = (
                "paper_only_challenger"
                if minimum_history >= 2
                and prediction.calibrated_logit_scale > 0.0
                else "abstain_low_history_uncertainty"
            )
        return output

    def annotate_best_price_expected_returns(
        self,
        frame: pd.DataFrame,
        bookies: Iterable[str],
    ) -> pd.DataFrame:
        """Freeze the best displayed price and Bayesian shadow-policy result."""

        output = frame.copy(deep=True)
        defaults = {
            "bayesian decision policy": "bayesian-moneyline-shadow-v1",
            "bayesian candidate selection": "",
            "bayesian candidate book": "",
            "bayesian candidate odds": np.nan,
            "bayesian posterior mean ev": np.nan,
            "bayesian ev lower": np.nan,
            "bayesian ev upper": np.nan,
            "bayesian probability positive ev": np.nan,
            "bayesian paper action": "pass",
            "bayesian paper threshold met": False,
            "bayesian decision status": "pass_no_eligible_price",
        }
        for column, default in defaults.items():
            if column not in output:
                output[column] = default
        ordered_bookies = tuple(str(book).strip() for book in bookies if str(book).strip())
        for index, row in output.iterrows():
            if row.get("bayesian status") != "paper_only_challenger":
                output.at[index, "bayesian decision status"] = (
                    "pass_low_history_uncertainty"
                )
                continue
            try:
                fighter_prediction = LogitNormalPrediction(
                    posterior_mean_probability=float(
                        row["bayesian posterior mean"]
                    ),
                    posterior_median_probability=float(
                        row["bayesian posterior median"]
                    ),
                    lower_probability=float(row["bayesian probability lower"]),
                    upper_probability=float(row["bayesian probability upper"]),
                    calibrated_logit_location=float(
                        row["bayesian calibrated logit location"]
                    ),
                    calibrated_logit_scale=float(
                        row["bayesian calibrated logit scale"]
                    ),
                    credible_level=float(row["bayesian credible level"]),
                )
            except (KeyError, TypeError, ValueError):
                output.at[index, "bayesian decision status"] = (
                    "pass_invalid_posterior"
                )
                continue
            candidates: list[tuple[float, float, str, str, int, dict[str, float]]] = []
            for book in ordered_bookies:
                for side, prediction, name_column in (
                    ("fighter", fighter_prediction, "fighter name"),
                    ("opponent", fighter_prediction.complement(), "opponent name"),
                ):
                    value = row.get(f"{side} {book}")
                    try:
                        numeric_odds = int(float(value))
                        result = prediction.expected_return(numeric_odds)
                    except (TypeError, ValueError):
                        continue
                    candidates.append(
                        (
                            result["posterior_mean_expected_return"],
                            result["probability_positive_expected_return"],
                            book.casefold(),
                            side,
                            numeric_odds,
                            result,
                        )
                    )
            if not candidates:
                continue
            # Highest posterior-mean EV wins; confidence and stable book/side
            # ordering make ties deterministic without consulting outcomes.
            selected = min(
                candidates,
                key=lambda item: (-item[0], -item[1], item[2], item[3]),
            )
            mean_ev, probability_positive, book_key, side, odds, result = selected
            display_book = next(
                book for book in ordered_bookies if book.casefold() == book_key
            )
            selection = str(row.get(f"{side} name") or "").strip()
            if not selection:
                selection = str(row.get(
                    "fighter name" if side == "fighter" else "opponent name"
                ) or "").strip()
            threshold_met = (
                mean_ev >= BAYESIAN_MINIMUM_MEAN_EV
                and probability_positive
                >= BAYESIAN_MINIMUM_PROBABILITY_POSITIVE_EV
            )
            output.at[index, "bayesian candidate selection"] = selection
            output.at[index, "bayesian candidate book"] = display_book
            output.at[index, "bayesian candidate odds"] = odds
            output.at[index, "bayesian posterior mean ev"] = mean_ev
            output.at[index, "bayesian ev lower"] = result[
                "lower_expected_return"
            ]
            output.at[index, "bayesian ev upper"] = result[
                "upper_expected_return"
            ]
            output.at[index, "bayesian probability positive ev"] = (
                probability_positive
            )
            output.at[index, "bayesian paper threshold met"] = threshold_met
            output.at[index, "bayesian paper action"] = (
                side if threshold_met else "pass"
            )
            output.at[index, "bayesian decision status"] = (
                "shadow_selection" if threshold_met else "pass_below_threshold"
            )
        return output

    @staticmethod
    def _flatten_lower(matrix: np.ndarray) -> list[float]:
        return [
            float(matrix[row, column])
            for row in range(matrix.shape[0])
            for column in range(row + 1)
        ]

    @staticmethod
    def _expand_lower(values: object, size: int) -> np.ndarray:
        expected = size * (size + 1) // 2
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError("Bayesian posterior Cholesky length is invalid")
        matrix = np.zeros((size, size), dtype=float)
        offset = 0
        for row in range(size):
            width = row + 1
            matrix[row, :width] = np.asarray(values[offset : offset + width], dtype=float)
            offset += width
        if not np.isfinite(matrix).all() or (np.diag(matrix) <= 0.0).any():
            raise ValueError("Bayesian posterior Cholesky is invalid")
        return matrix

    def artifact(self) -> dict[str, object]:
        if self._loaded_artifact is not None:
            return json.loads(json.dumps(self._loaded_artifact))
        factor = np.linalg.cholesky(self.covariance)
        base = self.base_artifact
        body: dict[str, object] = {
            "schema_version": BAYESIAN_SCHEMA_VERSION,
            "model_version": BAYESIAN_MODEL_VERSION,
            "model_type": "Gaussian-prior logistic regression with Laplace posterior",
            "paper_only": True,
            "execution_enabled": False,
            "base_model_id": base["model_id"],
            "base_model_version": base["model_version"],
            "data_through": base["data_through"],
            "training_labels_through": base["training_labels_through"],
            "training_fights": base["training_fights"],
            "training_fingerprint_sha256": base["training_fingerprint_sha256"],
            "state_fingerprint_sha256": base["state_fingerprint_sha256"],
            "feature_columns": self.feature_columns,
            "scaler_scale": [float(value) for value in self.scaler_scale],
            "coefficient_location": [float(value) for value in self.coefficients],
            "posterior_cholesky_lower": self._flatten_lower(factor),
            "selected_c": self.selected_c,
            "coefficient_prior": {
                "distribution": "independent_normal",
                "mean": 0.0,
                "standard_deviation": math.sqrt(self.selected_c),
                "precision": 1.0 / self.selected_c,
            },
            "posterior_approximation": "inverse penalized-likelihood Hessian at MAP",
            "calibration_slope": self.calibration_slope,
            "calibration_uncertainty": "fixed_from_causal_out_of-fold_calibration",
            "credible_level": BAYESIAN_CREDIBLE_LEVEL,
            "temporal_evaluation": self.temporal_evaluation,
            "decision_policy": {
                "version": "bayesian-moneyline-shadow-v1",
                "minimum_mean_expected_return": BAYESIAN_MINIMUM_MEAN_EV,
                "minimum_probability_positive_expected_return": (
                    BAYESIAN_MINIMUM_PROBABILITY_POSITIVE_EV
                ),
                "execution_enabled": False,
            },
        }
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        body["model_id"] = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return body

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        builder,
        base_artifact: dict[str, object],
    ) -> "BayesianLogisticChallenger":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema_version", "model_version", "model_id", "base_model_id",
            "base_model_version", "data_through", "training_labels_through",
            "training_fights", "training_fingerprint_sha256",
            "state_fingerprint_sha256", "feature_columns", "scaler_scale",
            "coefficient_location", "posterior_cholesky_lower", "selected_c",
            "calibration_slope", "credible_level", "temporal_evaluation",
            "decision_policy", "paper_only", "execution_enabled",
        }
        missing = sorted(required - set(artifact))
        if missing:
            raise ValueError(f"Bayesian artifact is missing fields: {missing}")
        if artifact["schema_version"] != BAYESIAN_SCHEMA_VERSION:
            raise ValueError("unsupported Bayesian artifact schema version")
        if artifact["model_version"] != BAYESIAN_MODEL_VERSION:
            raise ValueError("unsupported Bayesian model version")
        unhashed = dict(artifact)
        supplied_id = unhashed.pop("model_id")
        canonical = json.dumps(
            unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected_id = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        if supplied_id != expected_id:
            raise ValueError("Bayesian model_id does not match artifact contents")
        base_contract = {
            "base_model_id": base_artifact["model_id"],
            "base_model_version": base_artifact["model_version"],
            "data_through": base_artifact["data_through"],
            "training_labels_through": base_artifact["training_labels_through"],
            "training_fights": base_artifact["training_fights"],
            "training_fingerprint_sha256": base_artifact[
                "training_fingerprint_sha256"
            ],
            "state_fingerprint_sha256": base_artifact["state_fingerprint_sha256"],
            "feature_columns": base_artifact["feature_columns"],
            "scaler_scale": base_artifact["scaler_scale"],
            "coefficient_location": base_artifact["coefficients"],
            "selected_c": base_artifact["selected_c"],
            "calibration_slope": base_artifact["calibration_slope"],
        }
        for key, expected in base_contract.items():
            if artifact.get(key) != expected:
                raise ValueError(f"Bayesian artifact {key} differs from winner model")
        size = len(base_artifact["feature_columns"])
        factor = cls._expand_lower(artifact["posterior_cholesky_lower"], size)
        covariance = factor @ factor.T
        return cls(
            builder=builder,
            base_artifact=base_artifact,
            covariance=covariance,
            temporal_evaluation=artifact["temporal_evaluation"],
            loaded_artifact=artifact,
        )

    def save_artifact(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            self.artifact(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
                target.write(text)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

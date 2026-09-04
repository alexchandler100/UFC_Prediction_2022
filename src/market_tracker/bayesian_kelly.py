"""Uncertainty-aware, paper-only Kelly sizing for moneyline probabilities.

Plain Bayesian expected-log Kelly collapses a probability posterior to its
mean.  That does not make the stake smaller when the estimate is uncertain.
This module therefore implements an explicit robust rule: use the lower 10th
percentile of a calibrated probability posterior, then apply a separate
single-bet cap.  The posterior calibration is symmetric under fighter swaps.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ._common import canonical_hash


SCHEMA_VERSION = 1
POLICY_VERSION = "robust-bayesian-kelly-moneyline-v1"
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "content/data/model_research/bayesian_kelly_market_calibration.json"
)
DEFAULT_REPLAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "content/data/market_history_backfill/current_model_market_replay.csv"
)
PRIOR_LOG_SLOPE_LOCATION = 0.0
PRIOR_LOG_SLOPE_SCALE = 0.5
POSTERIOR_DRAW_COUNT = 257
GRID_SIZE = 30_001
LOWER_TAIL_PROBABILITY = 0.10
MAX_SINGLE_BET_FRACTION = 0.05


def _bounded_probability(value: object, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"{field} must be strictly between zero and one")
    return numeric


def _profit_multiple(moneyline: object) -> float:
    try:
        line = int(moneyline)
    except (TypeError, ValueError) as error:
        raise ValueError("moneyline must be an American price") from error
    if line == 0 or abs(line) < 100:
        raise ValueError("moneyline must have magnitude of at least 100")
    return line / 100.0 if line > 0 else 100.0 / abs(line)


def full_kelly_fraction(probability: object, moneyline: object) -> float:
    chance = _bounded_probability(probability, "Kelly probability")
    profit = _profit_multiple(moneyline)
    return min(1.0, max(0.0, (profit * chance - (1.0 - chance)) / profit))


def expected_log_growth(
    fraction: object, probability: object, moneyline: object
) -> float:
    stake = float(fraction)
    chance = _bounded_probability(probability, "growth probability")
    profit = _profit_multiple(moneyline)
    if not math.isfinite(stake) or not 0.0 <= stake < 1.0:
        raise ValueError("stake fraction must be finite and in [0, 1)")
    return chance * math.log1p(stake * profit) + (1.0 - chance) * math.log1p(-stake)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -709.0, 709.0)))


def _logit(probability: np.ndarray | float) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(bounded / (1.0 - bounded))


def _posterior_slope_draws(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    draw_count: int = POSTERIOR_DRAW_COUNT,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    if draw_count < 33 or grid_size < 1001:
        raise ValueError("Bayesian calibration resolution is too small")
    logits = _logit(probabilities)
    log_slope = np.linspace(math.log(0.1), math.log(4.0), grid_size)
    slopes = np.exp(log_slope)
    log_likelihood = np.empty(grid_size, dtype=float)
    for start in range(0, grid_size, 500):
        linear = np.outer(slopes[start : start + 500], logits)
        log_likelihood[start : start + 500] = np.sum(
            targets * -np.logaddexp(0.0, -linear)
            + (1.0 - targets) * -np.logaddexp(0.0, linear),
            axis=1,
        )
    log_prior = -0.5 * (
        (log_slope - PRIOR_LOG_SLOPE_LOCATION) / PRIOR_LOG_SLOPE_SCALE
    ) ** 2
    log_weights = log_likelihood + log_prior
    weights = np.exp(log_weights - float(log_weights.max()))
    weights /= float(weights.sum())
    cumulative = np.cumsum(weights)
    quantiles = (np.arange(draw_count, dtype=float) + 0.5) / draw_count
    return np.interp(quantiles, cumulative, slopes)


def _posterior_probabilities(
    probability: float, slope_draws: np.ndarray
) -> np.ndarray:
    return _sigmoid(float(_logit(probability)) * slope_draws)


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    bounded = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    truth = np.asarray(targets, dtype=float)
    return {
        "log_loss": float(
            np.mean(-(truth * np.log(bounded) + (1.0 - truth) * np.log1p(-bounded)))
        ),
        "brier": float(np.mean((bounded - truth) ** 2)),
        "accuracy": float(np.mean((bounded >= 0.5) == truth)),
    }


def _prepared_replay(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "event_date",
        "event_id",
        "fight_id",
        "market_probability",
        "target",
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"market calibration replay is missing columns: {missing}")
    prepared = frame[list(required)].copy()
    prepared["market_probability"] = pd.to_numeric(
        prepared["market_probability"], errors="coerce"
    )
    prepared["target"] = pd.to_numeric(prepared["target"], errors="coerce")
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="coerce")
    prepared = prepared.dropna().drop_duplicates("fight_id", keep=False)
    if prepared.empty:
        raise ValueError("market calibration replay has no unique complete fights")
    if not prepared["market_probability"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("market calibration probability is outside (0, 1)")
    if not prepared["target"].isin([0, 1]).all():
        raise ValueError("market calibration target is not binary")
    return prepared.sort_values(
        ["event_date", "event_id", "fight_id"], kind="stable"
    ).reset_index(drop=True)


def _calibrated_means(
    probabilities: np.ndarray, slope_draws: np.ndarray
) -> np.ndarray:
    logits = _logit(probabilities).reshape(-1, 1)
    return _sigmoid(logits * slope_draws.reshape(1, -1)).mean(axis=1)


def fit_market_calibration(
    frame: pd.DataFrame,
    *,
    source_sha256: str,
    created_at_utc: object | None = None,
) -> dict[str, object]:
    """Fit and chronologically check a one-parameter symmetric posterior."""

    prepared = _prepared_replay(frame)
    if len(prepared) < 100:
        raise ValueError("market calibration requires at least 100 completed fights")
    events = prepared[["event_date", "event_id"]].drop_duplicates().reset_index(drop=True)
    if len(events) < 10:
        raise ValueError("market calibration requires at least 10 events")
    split_index = max(1, int(len(events) * 0.8)) - 1
    cutoff = events.iloc[split_index]
    development_mask = (prepared["event_date"] < cutoff["event_date"]) | (
        (prepared["event_date"] == cutoff["event_date"])
        & (prepared["event_id"] <= cutoff["event_id"])
    )
    development = prepared[development_mask]
    holdout = prepared[~development_mask]
    if min(len(development), len(holdout)) < 50:
        raise ValueError("market calibration chronological check is too small")
    development_draws = _posterior_slope_draws(
        development["market_probability"].to_numpy(dtype=float),
        development["target"].to_numpy(dtype=float),
    )
    holdout_raw = holdout["market_probability"].to_numpy(dtype=float)
    holdout_truth = holdout["target"].to_numpy(dtype=float)
    holdout_calibrated = _calibrated_means(holdout_raw, development_draws)
    raw_metrics = _metrics(holdout_truth, holdout_raw)
    calibrated_metrics = _metrics(holdout_truth, holdout_calibrated)
    final_draws = _posterior_slope_draws(
        prepared["market_probability"].to_numpy(dtype=float),
        prepared["target"].to_numpy(dtype=float),
    )
    issued = pd.Timestamp(created_at_utc or datetime.now(timezone.utc))
    issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")
    source_records = [
        {
            "event_date": row.event_date.date().isoformat(),
            "event_id": str(row.event_id),
            "fight_id": str(row.fight_id),
            "market_probability": float(row.market_probability),
            "target": int(row.target),
        }
        for row in prepared.itertuples(index=False)
    ]
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "created_at_utc": issued.isoformat(),
        "source_file_sha256": str(source_sha256),
        "source_records_sha256": canonical_hash(source_records),
        "training_fights": len(prepared),
        "training_events": len(events),
        "training_first_event_date": prepared["event_date"].min().date().isoformat(),
        "training_last_event_date": prepared["event_date"].max().date().isoformat(),
        "probability_source": "timestamped_pre_event_multi_book_moneyline_consensus",
        "calibration_model": "zero_intercept_logistic_slope",
        "symmetry_contract": "calibrate(1-p) = 1-calibrate(p)",
        "prior": {
            "distribution": "normal_on_log_slope",
            "location": PRIOR_LOG_SLOPE_LOCATION,
            "scale": PRIOR_LOG_SLOPE_SCALE,
            "slope_grid_minimum": 0.1,
            "slope_grid_maximum": 4.0,
            "grid_size": GRID_SIZE,
        },
        "posterior": {
            "representation": "deterministic_equal_mass_slope_draws",
            "draw_count": len(final_draws),
            "slope_draws": [float(value) for value in final_draws],
            "slope_p10": float(np.quantile(final_draws, 0.10)),
            "slope_median": float(np.quantile(final_draws, 0.50)),
            "slope_p90": float(np.quantile(final_draws, 0.90)),
        },
        "staking_rule": {
            "lower_tail_probability": LOWER_TAIL_PROBABILITY,
            "maximum_single_bet_fraction": MAX_SINGLE_BET_FRACTION,
            "description": (
                "Kelly at the lower 10th percentile of the calibrated chance, "
                "capped at 5% of bankroll"
            ),
        },
        "chronological_check": {
            "development_fights": len(development),
            "development_events": int(
                development[["event_date", "event_id"]].drop_duplicates().shape[0]
            ),
            "development_last_event_date": development["event_date"].max().date().isoformat(),
            "holdout_fights": len(holdout),
            "holdout_events": int(
                holdout[["event_date", "event_id"]].drop_duplicates().shape[0]
            ),
            "holdout_first_event_date": holdout["event_date"].min().date().isoformat(),
            "holdout_last_event_date": holdout["event_date"].max().date().isoformat(),
            "raw_consensus": raw_metrics,
            "posterior_mean_calibrated": calibrated_metrics,
            "log_loss_change": calibrated_metrics["log_loss"] - raw_metrics["log_loss"],
            "brier_change": calibrated_metrics["brier"] - raw_metrics["brier"],
        },
        "limitations": [
            "This posterior measures uncertainty in population moneyline calibration, not every source of matchup error.",
            "The 5% cap separately limits unmodeled error and correlated same-card exposure.",
            "The calibration is not valid for total-round or method-of-victory probabilities.",
            "This policy is paper-only until prospectively evaluated.",
        ],
    }
    body["artifact_sha256"] = canonical_hash(body)
    return validate_market_calibration_artifact(body)


def validate_market_calibration_artifact(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Bayesian Kelly calibration artifact must be an object")
    artifact = dict(value)
    supplied = str(artifact.get("artifact_sha256") or "")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if supplied != canonical_hash(unhashed):
        raise ValueError("Bayesian Kelly calibration artifact hash is invalid")
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("policy_version") != POLICY_VERSION
        or artifact.get("candidate_only") is not True
        or artifact.get("paper_only") is not True
        or artifact.get("execution_enabled") is not False
    ):
        raise ValueError("Bayesian Kelly calibration policy is invalid")
    posterior = artifact.get("posterior")
    if not isinstance(posterior, dict):
        raise ValueError("Bayesian Kelly posterior is missing")
    draws = posterior.get("slope_draws")
    if (
        not isinstance(draws, list)
        or posterior.get("draw_count") != len(draws)
        or len(draws) < 33
    ):
        raise ValueError("Bayesian Kelly posterior draws are invalid")
    numeric = np.asarray(draws, dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0.0).any():
        raise ValueError("Bayesian Kelly posterior slope must be finite and positive")
    if np.any(numeric[:-1] > numeric[1:]):
        raise ValueError("Bayesian Kelly posterior slope draws must be sorted")
    if int(artifact.get("training_fights") or 0) < 100:
        raise ValueError("Bayesian Kelly calibration has too few fights")
    if int(artifact.get("training_events") or 0) < 10:
        raise ValueError("Bayesian Kelly calibration has too few events")
    rule = artifact.get("staking_rule")
    if not isinstance(rule, dict):
        raise ValueError("Bayesian Kelly staking rule is missing")
    tail = float(rule.get("lower_tail_probability"))
    cap = float(rule.get("maximum_single_bet_fraction"))
    if not 0.0 < tail < 0.5 or not 0.0 < cap < 1.0:
        raise ValueError("Bayesian Kelly staking limits are invalid")
    if len(str(artifact.get("source_file_sha256") or "")) != 64:
        raise ValueError("Bayesian Kelly source fingerprint is invalid")
    return artifact


class BayesianKellyCalibrator:
    def __init__(self, artifact: Mapping[str, object]):
        self.artifact = validate_market_calibration_artifact(dict(artifact))
        posterior = self.artifact["posterior"]
        rule = self.artifact["staking_rule"]
        self.slope_draws = np.asarray(posterior["slope_draws"], dtype=float)
        self.lower_tail_probability = float(rule["lower_tail_probability"])
        self.maximum_single_bet_fraction = float(
            rule["maximum_single_bet_fraction"]
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "BayesianKellyCalibrator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def assessment(
        self,
        probability: object,
        moneyline: object,
        *,
        assessment_timing: str = "prospective_at_publication",
    ) -> dict[str, object]:
        nominal = _bounded_probability(probability, "estimated win probability")
        posterior = np.sort(_posterior_probabilities(nominal, self.slope_draws))
        lower = float(np.quantile(posterior, self.lower_tail_probability))
        upper = float(np.quantile(posterior, 1.0 - self.lower_tail_probability))
        mean = float(posterior.mean())
        profit = _profit_multiple(moneyline)
        break_even = 1.0 / (1.0 + profit)
        robust = full_kelly_fraction(lower, moneyline)
        recommended = min(robust, self.maximum_single_bet_fraction)
        return {
            "status": "available",
            "policy_version": POLICY_VERSION,
            "calibration_artifact_sha256": self.artifact["artifact_sha256"],
            "calibration_training_fights": self.artifact["training_fights"],
            "calibration_training_events": self.artifact["training_events"],
            "calibration_trained_through": self.artifact[
                "training_last_event_date"
            ],
            "assessment_timing": assessment_timing,
            "nominal_probability": nominal,
            "posterior_mean_probability": mean,
            "posterior_lower_probability": lower,
            "posterior_upper_probability": upper,
            "posterior_interval_mass": 1.0 - 2.0 * self.lower_tail_probability,
            "probability_positive_edge": float(np.mean(posterior > break_even)),
            "break_even_probability": break_even,
            "posterior_mean_full_kelly_fraction": full_kelly_fraction(mean, moneyline),
            "robust_uncapped_kelly_fraction": robust,
            "maximum_single_bet_fraction": self.maximum_single_bet_fraction,
            "recommended_fraction": recommended,
            "cap_applied": bool(robust > self.maximum_single_bet_fraction),
            "expected_log_growth_at_probability_floor": expected_log_growth(
                recommended, lower, moneyline
            ),
        }


def unavailable_assessment(reason: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "policy_version": POLICY_VERSION,
        "reason": " ".join(str(reason).split()),
    }


def validate_bayesian_kelly_assessment(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Bayesian Kelly assessment must be an object")
    result = dict(value)
    if result.get("policy_version") != POLICY_VERSION:
        raise ValueError("Bayesian Kelly assessment policy is invalid")
    if result.get("status") == "unavailable":
        if not str(result.get("reason") or "").strip():
            raise ValueError("unavailable Bayesian Kelly assessment needs a reason")
        return result
    if result.get("status") != "available":
        raise ValueError("Bayesian Kelly assessment status is invalid")
    probabilities = (
        "nominal_probability",
        "posterior_mean_probability",
        "posterior_lower_probability",
        "posterior_upper_probability",
        "break_even_probability",
    )
    values = {key: _bounded_probability(result.get(key), key) for key in probabilities}
    if not (
        values["posterior_lower_probability"]
        <= values["posterior_mean_probability"]
        <= values["posterior_upper_probability"]
    ):
        raise ValueError("Bayesian Kelly probability interval is inconsistent")
    for key in (
        "probability_positive_edge",
        "posterior_mean_full_kelly_fraction",
        "robust_uncapped_kelly_fraction",
        "maximum_single_bet_fraction",
        "recommended_fraction",
    ):
        numeric = float(result.get(key))
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Bayesian Kelly {key} is invalid")
    interval_mass = float(result.get("posterior_interval_mass"))
    if not math.isfinite(interval_mass) or not 0.0 < interval_mass < 1.0:
        raise ValueError("Bayesian Kelly posterior interval mass is invalid")
    if abs(float(result["recommended_fraction"]) - min(
        float(result["robust_uncapped_kelly_fraction"]),
        float(result["maximum_single_bet_fraction"]),
    )) > 1e-12:
        raise ValueError("Bayesian Kelly recommended fraction is inconsistent")
    cap_applied = bool(
        float(result["robust_uncapped_kelly_fraction"])
        > float(result["maximum_single_bet_fraction"])
    )
    if result.get("cap_applied") is not cap_applied:
        raise ValueError("Bayesian Kelly cap indicator is inconsistent")
    growth = float(result.get("expected_log_growth_at_probability_floor"))
    if not math.isfinite(growth) or growth < -1e-12:
        raise ValueError("Bayesian Kelly expected log growth is invalid")
    if len(str(result.get("calibration_artifact_sha256") or "")) != 64:
        raise ValueError("Bayesian Kelly calibration identity is invalid")
    return result

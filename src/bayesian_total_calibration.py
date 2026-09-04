"""Bayesian calibration and uncertainty-aware Kelly sizing for fight totals.

The duration model remains the coherent source of each Over probability.  This
module learns, separately for each half-round line, how strongly historical
out-of-sample probabilities should be trusted.  A posterior over that one
calibration coefficient gives an auditable probability range without making
the live updater depend on a heavyweight sampling framework.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from fight_semantics import SCHEDULE_CONTRACT_VERSION


SCHEMA_VERSION = 1
CALIBRATION_POLICY_VERSION = "bayesian-logistic-total-calibration-v1"
KELLY_POLICY_VERSION = "robust-bayesian-kelly-totals-v1"
DEFAULT_EVALUATION_PATH = (
    Path(__file__).resolve().parent
    / "content/data/external/outcome_model_evaluation.json"
)
PRIOR_LOG_SLOPE_LOCATION = 0.0
PRIOR_LOG_SLOPE_SCALE = 0.5
POSTERIOR_DRAW_COUNT = 257
GRID_SIZE = 30_001
LOWER_TAIL_PROBABILITY = 0.10
MAX_SINGLE_BET_FRACTION = 0.05
MINIMUM_LINE_FIGHTS = 40
MINIMUM_LINE_EVENTS = 8
MINIMUM_CHECK_FIGHTS = 8


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _bounded_probability(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{field} must be strictly between zero and one")
    return number


def _profit_multiple(moneyline: object) -> float:
    try:
        line = int(moneyline)
    except (TypeError, ValueError) as error:
        raise ValueError("moneyline must be an American price") from error
    if line == 0 or abs(line) < 100:
        raise ValueError("moneyline must have magnitude of at least 100")
    return line / 100.0 if line > 0 else 100.0 / abs(line)


def _full_kelly_fraction(probability: float, moneyline: object) -> float:
    profit = _profit_multiple(moneyline)
    return min(
        1.0,
        max(0.0, (profit * probability - (1.0 - probability)) / profit),
    )


def _expected_log_growth(
    fraction: float, probability: float, moneyline: object
) -> float:
    profit = _profit_multiple(moneyline)
    return probability * math.log1p(fraction * profit) + (
        1.0 - probability
    ) * math.log1p(-fraction)


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
    """Return deterministic equal-mass draws from a one-coefficient posterior."""

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


def _posterior_means(
    probabilities: np.ndarray, slope_draws: np.ndarray
) -> np.ndarray:
    logits = _logit(probabilities).reshape(-1, 1)
    return _sigmoid(logits * slope_draws.reshape(1, -1)).mean(axis=1)


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    truth = np.asarray(targets, dtype=float)
    bounded = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return {
        "log_loss": float(
            np.mean(
                -(truth * np.log(bounded) + (1.0 - truth) * np.log1p(-bounded))
            )
        ),
        "brier": float(np.mean((bounded - truth) ** 2)),
        "accuracy": float(np.mean((bounded >= 0.5) == truth)),
    }


def _line_key(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("total-round line must be numeric") from error
    if not math.isfinite(number) or number <= 0.0 or number * 2 % 2 != 1:
        raise ValueError("total-round line must be a positive half round")
    return f"{number:.1f}"


def _prepared_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_date",
        "event_id",
        "fight_id",
        "line",
        "model_probability",
        "target",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"total calibration data is missing columns: {missing}")
    prepared = frame[list(sorted(required))].copy()
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="coerce")
    prepared["model_probability"] = pd.to_numeric(
        prepared["model_probability"], errors="coerce"
    )
    prepared["target"] = pd.to_numeric(prepared["target"], errors="coerce")
    prepared["line"] = prepared["line"].map(_line_key)
    prepared = prepared.dropna()
    duplicated = prepared.duplicated(["fight_id", "line"], keep=False)
    if duplicated.any():
        raise ValueError("total calibration repeats a fight and line")
    if not prepared["model_probability"].between(
        0.0, 1.0, inclusive="neither"
    ).all():
        raise ValueError("total calibration probability is outside (0, 1)")
    if not prepared["target"].isin([0, 1]).all():
        raise ValueError("total calibration target is not binary")
    return prepared.sort_values(
        ["event_date", "event_id", "fight_id", "line"], kind="stable"
    ).reset_index(drop=True)


def fit_total_calibration(frame: pd.DataFrame) -> dict[str, object]:
    """Fit each totals line and check it on the latest 20% of its events."""

    prepared = _prepared_predictions(frame)
    if prepared.empty:
        raise ValueError("total calibration has no complete predictions")
    line_artifacts: dict[str, object] = {}
    source_records: list[dict[str, object]] = []
    for row in prepared.itertuples(index=False):
        source_records.append(
            {
                "event_date": row.event_date.date().isoformat(),
                "event_id": str(row.event_id),
                "fight_id": str(row.fight_id),
                "line": str(row.line),
                "model_probability": float(row.model_probability),
                "target": int(row.target),
            }
        )
    for line, group in prepared.groupby("line", sort=True):
        group = group.reset_index(drop=True)
        events = group[["event_date", "event_id"]].drop_duplicates().reset_index(
            drop=True
        )
        support_ok = (
            len(group) >= MINIMUM_LINE_FIGHTS
            and len(events) >= MINIMUM_LINE_EVENTS
        )
        item: dict[str, object] = {
            "status": "available" if support_ok else "unavailable",
            "training_fights": int(len(group)),
            "training_events": int(len(events)),
            "training_first_event_date": group["event_date"].min().date().isoformat(),
            "training_last_event_date": group["event_date"].max().date().isoformat(),
        }
        if not support_ok:
            item["reason"] = (
                f"Needs at least {MINIMUM_LINE_FIGHTS} fights across "
                f"{MINIMUM_LINE_EVENTS} events for this line."
            )
            line_artifacts[str(line)] = item
            continue
        split_index = max(1, int(len(events) * 0.8)) - 1
        cutoff = events.iloc[split_index]
        development_mask = (group["event_date"] < cutoff["event_date"]) | (
            (group["event_date"] == cutoff["event_date"])
            & (group["event_id"] <= cutoff["event_id"])
        )
        development = group[development_mask]
        holdout = group[~development_mask]
        chronological_check: dict[str, object]
        if min(len(development), len(holdout)) < MINIMUM_CHECK_FIGHTS:
            chronological_check = {
                "status": "too_small",
                "development_fights": int(len(development)),
                "holdout_fights": int(len(holdout)),
            }
            item["status"] = "unavailable"
            item["reason"] = (
                f"Needs at least {MINIMUM_CHECK_FIGHTS} fights in both the earlier "
                "calibration sample and the later check before staking is available."
            )
            item["chronological_check"] = chronological_check
            line_artifacts[str(line)] = item
            continue
        else:
            development_draws = _posterior_slope_draws(
                development["model_probability"].to_numpy(dtype=float),
                development["target"].to_numpy(dtype=float),
            )
            holdout_probability = holdout["model_probability"].to_numpy(dtype=float)
            holdout_truth = holdout["target"].to_numpy(dtype=float)
            raw_metrics = _metrics(holdout_truth, holdout_probability)
            calibrated_metrics = _metrics(
                holdout_truth,
                _posterior_means(holdout_probability, development_draws),
            )
            chronological_check = {
                "status": "complete",
                "development_fights": int(len(development)),
                "development_events": int(
                    development[["event_date", "event_id"]].drop_duplicates().shape[0]
                ),
                "development_last_event_date": development[
                    "event_date"
                ].max().date().isoformat(),
                "holdout_fights": int(len(holdout)),
                "holdout_events": int(
                    holdout[["event_date", "event_id"]].drop_duplicates().shape[0]
                ),
                "holdout_first_event_date": holdout[
                    "event_date"
                ].min().date().isoformat(),
                "holdout_last_event_date": holdout[
                    "event_date"
                ].max().date().isoformat(),
                "raw_duration_model": raw_metrics,
                "posterior_mean_calibrated": calibrated_metrics,
                "log_loss_change": (
                    calibrated_metrics["log_loss"] - raw_metrics["log_loss"]
                ),
                "brier_change": calibrated_metrics["brier"] - raw_metrics["brier"],
            }
        final_draws = _posterior_slope_draws(
            group["model_probability"].to_numpy(dtype=float),
            group["target"].to_numpy(dtype=float),
        )
        item.update(
            {
                "posterior": {
                    "representation": "deterministic_equal_mass_slope_draws",
                    "draw_count": int(len(final_draws)),
                    "slope_draws": [float(value) for value in final_draws],
                    "slope_p10": float(np.quantile(final_draws, 0.10)),
                    "slope_median": float(np.quantile(final_draws, 0.50)),
                    "slope_p90": float(np.quantile(final_draws, 0.90)),
                },
                "chronological_check": chronological_check,
            }
        )
        if chronological_check.get("status") == "complete" and (
            float(chronological_check["log_loss_change"]) > 0.0
            or float(chronological_check["brier_change"]) > 0.0
        ):
            item["status"] = "unavailable"
            item["reason"] = (
                "The Bayesian adjustment made probability accuracy worse on "
                "the later-fight check, so it is not used for staking."
            )
        line_artifacts[str(line)] = item
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": CALIBRATION_POLICY_VERSION,
        "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
        "candidate_only": True,
        "paper_only": True,
        "execution_enabled": False,
        "probability_source": (
            "duration_model_predictions_for_fights_after_model_training_period"
        ),
        "calibration_model": "zero_intercept_bayesian_logistic_slope_by_total_line",
        "symmetry_contract": "under_draw = 1 - over_draw",
        "source_records_sha256": _canonical_hash(source_records),
        "training_prediction_count": int(len(prepared)),
        "training_fight_count": int(prepared["fight_id"].nunique()),
        "training_event_count": int(
            prepared[["event_date", "event_id"]].drop_duplicates().shape[0]
        ),
        "training_first_event_date": prepared["event_date"].min().date().isoformat(),
        "training_last_event_date": prepared["event_date"].max().date().isoformat(),
        "prior": {
            "distribution": "normal_on_log_slope",
            "location": PRIOR_LOG_SLOPE_LOCATION,
            "scale": PRIOR_LOG_SLOPE_SCALE,
            "slope_grid_minimum": 0.1,
            "slope_grid_maximum": 4.0,
            "grid_size": GRID_SIZE,
        },
        "staking_rule": {
            "lower_tail_probability": LOWER_TAIL_PROBABILITY,
            "maximum_single_bet_fraction": MAX_SINGLE_BET_FRACTION,
            "description": (
                "Kelly at the lower 10th percentile of the calibrated chance, "
                "capped at 5% of bankroll"
            ),
        },
        "lines": line_artifacts,
        "limitations": [
            "The posterior measures historical calibration uncertainty, not every source of matchup error.",
            "Long five-round totals have fewer completed fights and therefore wider uncertainty.",
            "This policy is paper-only until it has enough prospective settlements.",
        ],
    }
    body["artifact_sha256"] = _canonical_hash(body)
    return validate_total_calibration_artifact(body)


def validate_total_calibration_artifact(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Bayesian total calibration artifact must be an object")
    artifact = dict(value)
    supplied = str(artifact.get("artifact_sha256") or "")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if supplied != _canonical_hash(unhashed):
        raise ValueError("Bayesian total calibration artifact hash is invalid")
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("policy_version") != CALIBRATION_POLICY_VERSION
        or artifact.get("candidate_only") is not True
        or artifact.get("paper_only") is not True
        or artifact.get("execution_enabled") is not False
    ):
        raise ValueError("Bayesian total calibration policy is invalid")
    schedule_contract = artifact.get("schedule_contract_version")
    if schedule_contract not in {None, SCHEDULE_CONTRACT_VERSION}:
        raise ValueError("Bayesian total calibration schedule contract is invalid")
    lines = artifact.get("lines")
    if not isinstance(lines, dict) or not lines:
        raise ValueError("Bayesian total calibration lines are missing")
    for line, item in lines.items():
        if _line_key(line) != line or not isinstance(item, dict):
            raise ValueError("Bayesian total calibration line is invalid")
        if item.get("status") == "unavailable":
            if not str(item.get("reason") or "").strip():
                raise ValueError("unavailable total calibration needs a reason")
            continue
        if item.get("status") != "available":
            raise ValueError("Bayesian total calibration status is invalid")
        # Historical artifacts remain readable for audits and settled records.
        # Only newly certified artifacts may be used to generate current stakes.
        if schedule_contract == SCHEDULE_CONTRACT_VERSION:
            check = item.get("chronological_check")
            if (
                int(item.get("training_fights") or 0) < MINIMUM_LINE_FIGHTS
                or int(item.get("training_events") or 0) < MINIMUM_LINE_EVENTS
                or not isinstance(check, dict)
                or check.get("status") != "complete"
                or int(check.get("development_fights") or 0) < MINIMUM_CHECK_FIGHTS
                or int(check.get("holdout_fights") or 0) < MINIMUM_CHECK_FIGHTS
            ):
                raise ValueError("Bayesian total calibration lacks a supported later-fight check")
            for metric in ("log_loss_change", "brier_change"):
                change = float(check.get(metric, math.nan))
                if not math.isfinite(change) or change > 0.0:
                    raise ValueError("Bayesian total calibration did not pass its later-fight check")
        posterior = item.get("posterior")
        if not isinstance(posterior, dict):
            raise ValueError("Bayesian total posterior is missing")
        draws = posterior.get("slope_draws")
        if (
            not isinstance(draws, list)
            or posterior.get("draw_count") != len(draws)
            or len(draws) < 33
        ):
            raise ValueError("Bayesian total posterior draws are invalid")
        numeric = np.asarray(draws, dtype=float)
        if (
            not np.isfinite(numeric).all()
            or (numeric <= 0.0).any()
            or np.any(numeric[:-1] > numeric[1:])
        ):
            raise ValueError("Bayesian total posterior slopes are invalid")
    return artifact


class BayesianTotalCalibrator:
    def __init__(self, artifact: Mapping[str, object]):
        self.artifact = validate_total_calibration_artifact(dict(artifact))
        if self.artifact.get("schedule_contract_version") != SCHEDULE_CONTRACT_VERSION:
            raise ValueError(
                "Historical total calibration lacks independently verified schedules; "
                "new staking is unavailable until the duration model is rebuilt."
            )
        rule = self.artifact["staking_rule"]
        self.lower_tail_probability = float(rule["lower_tail_probability"])
        self.maximum_single_bet_fraction = float(
            rule["maximum_single_bet_fraction"]
        )

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_EVALUATION_PATH
    ) -> "BayesianTotalCalibrator":
        evaluation = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = evaluation.get("bayesian_total_calibration")
        if not isinstance(artifact, dict):
            raise ValueError("outcome evaluation has no Bayesian total calibration")
        return cls(artifact)

    def posterior_over_probabilities(
        self, probability: object, line: object
    ) -> np.ndarray:
        nominal = _bounded_probability(probability, "duration-model probability")
        key = _line_key(line)
        item = self.artifact["lines"].get(key)
        if not isinstance(item, dict) or item.get("status") != "available":
            reason = item.get("reason") if isinstance(item, dict) else "line is unknown"
            raise ValueError(f"Bayesian calibration unavailable for {key}: {reason}")
        draws = np.asarray(item["posterior"]["slope_draws"], dtype=float)
        return _posterior_probabilities(nominal, draws)

    def summary(self, probability: object, line: object) -> dict[str, object]:
        nominal = _bounded_probability(probability, "duration-model probability")
        key = _line_key(line)
        item = self.artifact["lines"].get(key)
        if not isinstance(item, dict) or item.get("status") != "available":
            return {
                "status": "unavailable",
                "policy_version": CALIBRATION_POLICY_VERSION,
                "reason": str(
                    item.get("reason")
                    if isinstance(item, dict)
                    else "No historical calibration exists for this line."
                ),
            }
        posterior = np.sort(self.posterior_over_probabilities(nominal, key))
        return {
            "status": "available",
            "policy_version": CALIBRATION_POLICY_VERSION,
            "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
            "calibration_artifact_sha256": self.artifact["artifact_sha256"],
            "calibration_training_fights": item["training_fights"],
            "calibration_training_events": item["training_events"],
            "calibration_trained_through": item["training_last_event_date"],
            "nominal_over_probability": nominal,
            "posterior_mean_over_probability": float(posterior.mean()),
            "posterior_lower_over_probability": float(
                np.quantile(posterior, self.lower_tail_probability)
            ),
            "posterior_upper_over_probability": float(
                np.quantile(posterior, 1.0 - self.lower_tail_probability)
            ),
            "posterior_interval_mass": 1.0 - 2.0 * self.lower_tail_probability,
        }

    def assessment(
        self,
        over_probability: object,
        side: object,
        line: object,
        moneyline: object,
        *,
        assessment_timing: str = "prospective_at_publication",
    ) -> dict[str, object]:
        side_text = str(side or "").strip().casefold()
        if side_text not in {"over", "under"}:
            raise ValueError("total assessment side must be over or under")
        key = _line_key(line)
        item = self.artifact["lines"].get(key)
        if not isinstance(item, dict) or item.get("status") != "available":
            return unavailable_total_assessment(
                str(
                    item.get("reason")
                    if isinstance(item, dict)
                    else "No historical calibration exists for this line."
                )
            )
        nominal_over = _bounded_probability(
            over_probability, "duration-model Over probability"
        )
        posterior = self.posterior_over_probabilities(nominal_over, key)
        nominal = nominal_over
        if side_text == "under":
            nominal = 1.0 - nominal_over
            posterior = 1.0 - posterior
        posterior = np.sort(posterior)
        lower = float(np.quantile(posterior, self.lower_tail_probability))
        upper = float(np.quantile(posterior, 1.0 - self.lower_tail_probability))
        mean = float(posterior.mean())
        profit = _profit_multiple(moneyline)
        break_even = 1.0 / (1.0 + profit)
        robust = _full_kelly_fraction(lower, moneyline)
        recommended = min(robust, self.maximum_single_bet_fraction)
        return {
            "status": "available",
            "policy_version": KELLY_POLICY_VERSION,
            "schedule_contract_version": SCHEDULE_CONTRACT_VERSION,
            "calibration_artifact_sha256": self.artifact["artifact_sha256"],
            "calibration_training_fights": item["training_fights"],
            "calibration_training_events": item["training_events"],
            "calibration_trained_through": item["training_last_event_date"],
            "assessment_timing": assessment_timing,
            "market": "total_rounds",
            "line": float(key),
            "side": side_text,
            "nominal_probability": nominal,
            "posterior_mean_probability": mean,
            "posterior_lower_probability": lower,
            "posterior_upper_probability": upper,
            "posterior_interval_mass": 1.0 - 2.0 * self.lower_tail_probability,
            "probability_positive_edge": float(np.mean(posterior > break_even)),
            "break_even_probability": break_even,
            "posterior_mean_full_kelly_fraction": _full_kelly_fraction(
                mean, moneyline
            ),
            "robust_uncapped_kelly_fraction": robust,
            "maximum_single_bet_fraction": self.maximum_single_bet_fraction,
            "recommended_fraction": recommended,
            "cap_applied": bool(robust > self.maximum_single_bet_fraction),
            "expected_log_growth_at_probability_floor": _expected_log_growth(
                recommended, lower, moneyline
            ),
        }


def unavailable_total_assessment(reason: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "policy_version": KELLY_POLICY_VERSION,
        "reason": " ".join(str(reason).split()),
    }


def validate_total_bayesian_kelly_assessment(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Bayesian total Kelly assessment must be an object")
    result = dict(value)
    if result.get("policy_version") != KELLY_POLICY_VERSION:
        raise ValueError("Bayesian total Kelly policy is invalid")
    if result.get("status") == "unavailable":
        if not str(result.get("reason") or "").strip():
            raise ValueError("unavailable Bayesian total Kelly needs a reason")
        return result
    if result.get("status") != "available":
        raise ValueError("Bayesian total Kelly status is invalid")
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
        raise ValueError("Bayesian total probability interval is inconsistent")
    for key in (
        "probability_positive_edge",
        "posterior_mean_full_kelly_fraction",
        "robust_uncapped_kelly_fraction",
        "maximum_single_bet_fraction",
        "recommended_fraction",
    ):
        numeric = float(result.get(key))
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Bayesian total Kelly {key} is invalid")
    if result.get("side") not in {"over", "under"}:
        raise ValueError("Bayesian total Kelly side is invalid")
    _line_key(result.get("line"))
    expected = min(
        float(result["robust_uncapped_kelly_fraction"]),
        float(result["maximum_single_bet_fraction"]),
    )
    if abs(float(result["recommended_fraction"]) - expected) > 1e-12:
        raise ValueError("Bayesian total Kelly recommendation is inconsistent")
    if result.get("cap_applied") is not (
        float(result["robust_uncapped_kelly_fraction"])
        > float(result["maximum_single_bet_fraction"])
    ):
        raise ValueError("Bayesian total Kelly cap indicator is inconsistent")
    interval_mass = float(result.get("posterior_interval_mass"))
    growth = float(result.get("expected_log_growth_at_probability_floor"))
    if not math.isfinite(interval_mass) or not 0.0 < interval_mass < 1.0:
        raise ValueError("Bayesian total probability interval mass is invalid")
    if not math.isfinite(growth) or growth < -1e-12:
        raise ValueError("Bayesian total Kelly expected growth is invalid")
    if len(str(result.get("calibration_artifact_sha256") or "")) != 64:
        raise ValueError("Bayesian total Kelly calibration identity is invalid")
    return result

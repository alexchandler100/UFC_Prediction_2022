"""Chronological research evaluation for simulation forecasts.

The module evaluates a compact ledger whose ``forecast`` column contains an
``AggregateForecast`` (or its serialized mapping).  It deliberately has no
production-selection side effects.  Fold runners receive only training rows
strictly before the test year and must return forecasts for that year's rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import cramervonmises, kstest
from sklearn.linear_model import LogisticRegression

from .parameters import canonical_json, canonical_sha256


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_VERSION = "fight-sim-chronological-evaluation-v2"
DEFAULT_OUTCOMES = (
    "red_ko_tko",
    "red_submission",
    "red_decision",
    "red_other",
    "blue_ko_tko",
    "blue_submission",
    "blue_decision",
    "blue_other",
    "draw",
    "no_contest",
)
METHODS = ("ko_tko", "submission", "decision", "other", "draw", "no_contest")


@dataclass(frozen=True)
class BacktestConfig:
    first_test_year: int | None = None
    last_test_year: int | None = None
    min_training_fights: int = 500
    card_bootstrap_replicates: int = 2000
    random_seed: int = 2903
    stack_min_training_fights: int = 100
    stack_l2_penalty: float = 0.01

    def __post_init__(self) -> None:
        if self.min_training_fights <= 0:
            raise ValueError("min_training_fights must be positive")
        if self.card_bootstrap_replicates <= 0:
            raise ValueError("card_bootstrap_replicates must be positive")
        if self.stack_min_training_fights <= 0:
            raise ValueError("stack_min_training_fights must be positive")
        if not math.isfinite(self.stack_l2_penalty) or self.stack_l2_penalty < 0:
            raise ValueError("stack_l2_penalty must be finite and nonnegative")
        if (
            self.first_test_year is not None
            and self.last_test_year is not None
            and self.first_test_year > self.last_test_year
        ):
            raise ValueError("first_test_year must not exceed last_test_year")


@dataclass(frozen=True)
class BacktestReport:
    schema_version: int
    evaluation_version: str
    candidate_only: bool
    production_enabled: bool
    execution_enabled: bool
    primary_metric: str
    config: dict[str, object]
    folds: tuple[dict[str, object], ...]
    aggregate: dict[str, object]
    slices: dict[str, object]
    comparisons: dict[str, object]
    coverage_warnings: tuple[str, ...]
    ledger_sha256: str
    report_sha256: str
    simulation_noise: dict[str, object] = field(default_factory=dict)

    def unhashed_dict(self) -> dict[str, object]:
        body = {
            "schema_version": self.schema_version,
            "evaluation_version": self.evaluation_version,
            "candidate_only": self.candidate_only,
            "production_enabled": self.production_enabled,
            "execution_enabled": self.execution_enabled,
            "primary_metric": self.primary_metric,
            "config": self.config,
            "folds": list(self.folds),
            "aggregate": self.aggregate,
            "slices": self.slices,
            "comparisons": self.comparisons,
            "coverage_warnings": list(self.coverage_warnings),
            "ledger_sha256": self.ledger_sha256,
        }
        # Optional for backward compatibility with already frozen v1 reports.
        # Once present, the precision summary is part of the report hash.
        if self.simulation_noise:
            body["simulation_noise"] = self.simulation_noise
        return body

    def to_dict(self) -> dict[str, object]:
        body = self.unhashed_dict()
        body["report_sha256"] = self.report_sha256
        return body

    def validate(self) -> "BacktestReport":
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported simulation evaluation schema")
        if not self.candidate_only or self.production_enabled or self.execution_enabled:
            raise ValueError("simulation evaluation must remain non-production research")
        if self.report_sha256 != canonical_sha256(self.unhashed_dict()):
            raise ValueError("simulation evaluation report hash is invalid")
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BacktestReport":
        report = cls(
            schema_version=int(value["schema_version"]),
            evaluation_version=str(value["evaluation_version"]),
            candidate_only=value.get("candidate_only") is True,
            production_enabled=value.get("production_enabled") is True,
            execution_enabled=value.get("execution_enabled") is True,
            primary_metric=str(value["primary_metric"]),
            config=dict(value.get("config") or {}),
            folds=tuple(dict(item) for item in list(value.get("folds") or [])),
            aggregate=dict(value.get("aggregate") or {}),
            slices=dict(value.get("slices") or {}),
            comparisons=dict(value.get("comparisons") or {}),
            coverage_warnings=tuple(
                str(item) for item in list(value.get("coverage_warnings") or [])
            ),
            ledger_sha256=str(value["ledger_sha256"]),
            report_sha256=str(value["report_sha256"]),
            simulation_noise=dict(value.get("simulation_noise") or {}),
        )
        return report.validate()


def write_backtest_report(
    path: str | Path, report: BacktestReport | Mapping[str, object]
) -> None:
    """Atomically persist a content-addressed candidate evaluation report."""

    validated = (
        report.validate()
        if isinstance(report, BacktestReport)
        else BacktestReport.from_dict(report)
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            validated.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_backtest_report(path: str | Path) -> BacktestReport:
    return BacktestReport.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _mapping(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("forecast must be AggregateForecast or a mapping")
    return dict(value)


def _outcome_probabilities(forecast: object) -> dict[str, float]:
    value = _mapping(forecast)
    probabilities = value.get("outcome_probabilities")
    if probabilities is None:
        counts = dict(value.get("outcome_counts") or {})
        total = int(value.get("total_paths") or sum(int(v) for v in counts.values()))
        if total <= 0:
            raise ValueError("forecast has no simulation paths")
        probabilities = {key: int(count) / total for key, count in counts.items()}
    result = {str(key): float(item) for key, item in dict(probabilities).items()}
    if not result or any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("forecast contains invalid outcome probabilities")
    if abs(sum(result.values()) - 1.0) > 1e-8:
        raise ValueError("forecast outcome probabilities do not sum to one")
    return result


def _duration_distribution(forecast: object) -> tuple[np.ndarray, np.ndarray]:
    value = _mapping(forecast)
    bins = value.get("duration_bins") or []
    total = int(value.get("total_paths") or 0)
    if isinstance(bins, Mapping):
        items = [
            {"upper_seconds": float(key), "count": int(count)}
            for key, count in bins.items()
        ]
    else:
        items = [dict(item) for item in bins]
    if not items or total <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    items.sort(key=lambda item: float(item["upper_seconds"]))
    seconds = np.asarray([float(item["upper_seconds"]) for item in items])
    counts = np.asarray([int(item["count"]) for item in items], dtype=float)
    if np.any(counts < 0) or int(counts.sum()) != total:
        raise ValueError("duration-bin counts disagree with total paths")
    return seconds, counts / total


def _statistic_distributions(
    forecast: object,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read exact statistic counts from a serialized or live forecast.

    The aggregate's integer counts are authoritative.  Summaries are
    intentionally not reverse-engineered into a distribution when an older
    publication does not contain the additive ``statistic_distributions``
    field.
    """

    value = _mapping(forecast)
    raw_distributions = value.get("statistic_distributions") or []
    if isinstance(raw_distributions, Mapping):
        rows = [
            {
                "statistic": name,
                "counts": counts,
                "total_paths": value.get("total_paths"),
            }
            for name, counts in raw_distributions.items()
        ]
    else:
        rows = [dict(item) for item in raw_distributions]

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        name = str(row.get("statistic") or "").strip()
        if not name or name in result:
            raise ValueError("statistic distributions require unique names")
        raw_counts = row.get("counts") or []
        if isinstance(raw_counts, Mapping):
            count_rows = [
                {"value": support, "count": count}
                for support, count in raw_counts.items()
            ]
        else:
            count_rows = [dict(item) for item in raw_counts]
        if not count_rows:
            raise ValueError(f"statistic distribution is empty: {name}")

        supports: list[float] = []
        counts: list[int] = []
        for item in count_rows:
            support = float(item["value"])
            raw_count = float(item["count"])
            count = int(raw_count)
            if (
                not math.isfinite(support)
                or not math.isfinite(raw_count)
                or raw_count != count
                or count < 0
            ):
                raise ValueError(f"statistic distribution contains invalid counts: {name}")
            supports.append(support)
            counts.append(count)
        if len(set(supports)) != len(supports):
            raise ValueError(f"statistic distribution contains duplicate values: {name}")

        total = int(row.get("total_paths") or value.get("total_paths") or sum(counts))
        if total <= 0 or sum(counts) != total:
            raise ValueError(
                f"statistic distribution counts disagree with total paths: {name}"
            )
        order = np.argsort(np.asarray(supports, dtype=float), kind="stable")
        ordered_support = np.asarray(supports, dtype=float)[order]
        ordered_counts = np.asarray(counts, dtype=float)[order]
        result[name] = ordered_support, ordered_counts / total
    return result


def _discrete_crps(
    actual: float,
    support: np.ndarray,
    mass: np.ndarray,
) -> float | None:
    if (
        len(support) == 0
        or len(support) != len(mass)
        or not math.isfinite(float(actual))
    ):
        return None
    # Linear-time form of E|X-y| - 1/2 E|X-X'| for sorted support.
    first = float(np.sum(mass * np.abs(support - float(actual))))
    cumulative_probability = 0.0
    cumulative_value = 0.0
    half_pair_distance = 0.0
    for value, probability in zip(support, mass, strict=True):
        half_pair_distance += float(probability) * (
            float(value) * cumulative_probability - cumulative_value
        )
        cumulative_probability += float(probability)
        cumulative_value += float(probability) * float(value)
    return first - half_pair_distance


def _weighted_quantile(
    support: np.ndarray,
    mass: np.ndarray,
    probability: float,
) -> float:
    if not 0.0 <= probability <= 1.0 or len(support) == 0:
        raise ValueError("weighted quantile requires nonempty support and a valid probability")
    index = int(np.searchsorted(np.cumsum(mass), probability, side="left"))
    return float(support[min(index, len(support) - 1)])


def _duration_integrated_brier(actual_seconds: float, forecast: object) -> float | None:
    """Return the time-averaged Brier score over the scheduled fight horizon."""

    seconds, mass = _duration_distribution(forecast)
    if len(seconds) == 0 or not math.isfinite(float(actual_seconds)):
        return None
    value = _mapping(forecast)
    scheduled_rounds = float(value.get("scheduled_rounds") or 0)
    horizon = max(
        float(np.max(seconds)),
        scheduled_rounds * 300.0,
        float(actual_seconds),
    )
    if horizon <= 0:
        return None
    boundaries = sorted(
        {
            0.0,
            horizon,
            min(max(float(actual_seconds), 0.0), horizon),
            *(min(max(float(item), 0.0), horizon) for item in seconds),
        }
    )
    integrated = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        predicted_survival = float(np.sum(mass[seconds > midpoint]))
        observed_survival = float(float(actual_seconds) > midpoint)
        integrated += (right - left) * (
            predicted_survival - observed_survival
        ) ** 2
    return integrated / horizon


def _total_over_probability(forecast: object, line_rounds: float) -> float | None:
    value = _mapping(forecast)
    for raw in list(value.get("total_lines") or []):
        item = dict(raw)
        if not math.isclose(
            float(item.get("half_rounds", math.nan)),
            float(line_rounds),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            continue
        over = int(item.get("over") or 0)
        under = int(item.get("under") or 0)
        if over < 0 or under < 0:
            raise ValueError("total-line counts must be nonnegative")
        binary_settlements = over + under
        return over / binary_settlements if binary_settlements else None
    return None


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _deterministic_uniform(*parts: object) -> float:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / float(2**64)


def _predictive_diagnostic(
    actual: float,
    support: np.ndarray,
    mass: np.ndarray,
    *,
    event_id: object,
    fight_id: object,
    statistic: str,
) -> dict[str, object]:
    mean = float(np.sum(support * mass))
    variance = float(np.sum(np.square(support - mean) * mass))
    standard_deviation = math.sqrt(max(0.0, variance))
    less = float(mass[support < actual].sum())
    equal = float(mass[support == actual].sum())
    randomized_pit = less + _deterministic_uniform(
        event_id, fight_id, statistic, "randomized-pit-v1"
    ) * equal
    randomized_tail = min(1.0, 2.0 * min(randomized_pit, 1.0 - randomized_pit))
    result: dict[str, object] = {
        "event_id": str(event_id),
        "fight_id": str(fight_id),
        "statistic": statistic,
        "observed": float(actual),
        "predictive_mean": mean,
        "predictive_standard_deviation": standard_deviation,
        "standardized_residual": (
            (float(actual) - mean) / standard_deviation
            if standard_deviation > 0.0
            else None
        ),
        "crps": _discrete_crps(float(actual), support, mass),
        "pit": randomized_pit,
        "randomized_two_sided_tail": randomized_tail,
        "point_mass_at_observed": equal,
    }
    for coverage in (0.50, 0.80, 0.90, 0.95):
        tail = (1.0 - coverage) / 2.0
        lower = _weighted_quantile(support, mass, tail)
        upper = _weighted_quantile(support, mass, 1.0 - tail)
        label = f"{int(coverage * 100)}"
        result[f"interval_{label}_lower"] = lower
        result[f"interval_{label}_upper"] = upper
        result[f"interval_{label}_contains"] = bool(lower <= actual <= upper)
    return result


def posterior_predictive_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return auditable fight/statistic diagnostics from exact forecast counts."""

    rows: list[dict[str, object]] = []
    for source in frame.to_dict("records"):
        event_id = source.get("event_id", "")
        fight_id = source.get("fight_id", "")
        duration = _finite_number(source.get("actual_duration_seconds"))
        if duration is not None:
            support, mass = _duration_distribution(source["forecast"])
            if len(support):
                rows.append(
                    _predictive_diagnostic(
                        duration,
                        support,
                        mass,
                        event_id=event_id,
                        fight_id=fight_id,
                        statistic="duration_seconds",
                    )
                )
        for statistic, (support, mass) in _statistic_distributions(
            source["forecast"]
        ).items():
            actual = _finite_number(source.get(f"actual_{statistic}"))
            if actual is None:
                continue
            rows.append(
                _predictive_diagnostic(
                    actual,
                    support,
                    mass,
                    event_id=event_id,
                    fight_id=fight_id,
                    statistic=statistic,
                )
            )
    return rows


def _diagnostic_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_statistic: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_statistic.setdefault(str(row["statistic"]), []).append(row)
    result: dict[str, object] = {}
    for statistic, items in sorted(by_statistic.items()):
        pits = np.asarray([float(item["pit"]) for item in items], dtype=float)
        residuals = np.asarray(
            [
                float(item["predictive_mean"]) - float(item["observed"])
                for item in items
            ],
            dtype=float,
        )
        standardized = np.asarray(
            [
                float(item["standardized_residual"])
                for item in items
                if item.get("standardized_residual") is not None
            ],
            dtype=float,
        )
        ks = kstest(pits, "uniform") if len(pits) >= 2 else None
        cvm = cramervonmises(pits, "uniform") if len(pits) >= 2 else None
        result[statistic] = {
            "n": len(items),
            "mean_crps": float(np.mean([float(item["crps"]) for item in items])),
            "predictive_minus_observed_mean": float(np.mean(residuals)),
            "mean_absolute_standardized_residual": (
                float(np.mean(np.abs(standardized))) if len(standardized) else None
            ),
            "pit_mean": float(np.mean(pits)),
            "pit_histogram_10": np.histogram(pits, bins=np.linspace(0.0, 1.0, 11))[0].astype(int).tolist(),
            "pit_ks_statistic": None if ks is None else float(ks.statistic),
            "pit_ks_nominal_iid_pvalue": None if ks is None else float(ks.pvalue),
            "pit_cvm_statistic": None if cvm is None else float(cvm.statistic),
            "pit_cvm_nominal_iid_pvalue": (
                None if cvm is None else float(cvm.pvalue)
            ),
            "randomized_tail_below_0_01_rate": float(np.mean([float(item["randomized_two_sided_tail"]) < 0.01 for item in items])),
            "randomized_tail_below_0_05_rate": float(np.mean([float(item["randomized_two_sided_tail"]) < 0.05 for item in items])),
            **{
                f"interval_{coverage}_coverage": float(
                    np.mean([bool(item[f"interval_{coverage}_contains"]) for item in items])
                )
                for coverage in (50, 80, 90, 95)
            },
        }
    return result


def _actual_method(outcome: str) -> str:
    for method in METHODS:
        if outcome == method or outcome.endswith(f"_{method}"):
            return method
    return "other"


def _red_win_probability(probabilities: Mapping[str, float]) -> float:
    red = float(sum(value for key, value in probabilities.items() if key.startswith("red_")))
    blue = float(sum(value for key, value in probabilities.items() if key.startswith("blue_")))
    return red / (red + blue) if red + blue > 0 else 0.5


def _binary_metrics(truth: Sequence[int], probability: Sequence[float]) -> dict[str, float | int | None]:
    y = np.asarray(truth, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    if len(y) == 0:
        return {"n": 0, "log_loss": None, "brier": None, "calibration_intercept": None, "calibration_slope": None}
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    brier = float(np.mean((p - y) ** 2))
    intercept: float | None = None
    slope: float | None = None
    if len(np.unique(y)) == 2 and np.ptp(p) > 1e-8:
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000).fit(logits, y)
        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0, 0])
    return {
        "n": int(len(y)),
        "log_loss": loss,
        "brier": brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _multiclass_log_loss(truth: Sequence[str], matrices: Sequence[Mapping[str, float]]) -> float | None:
    if not truth:
        return None
    losses = []
    for actual, probabilities in zip(truth, matrices):
        losses.append(-math.log(max(float(probabilities.get(actual, 0.0)), 1e-12)))
    return float(np.mean(losses))


def _duration_crps(actual_seconds: float, forecast: object) -> float | None:
    seconds, mass = _duration_distribution(forecast)
    return _discrete_crps(actual_seconds, seconds, mass)


def _method_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    return {
        method: float(
            sum(value for key, value in probabilities.items() if _actual_method(key) == method)
        )
        for method in METHODS
    }


def _ledger_fingerprint(frame: pd.DataFrame) -> str:
    def json_value(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            value = value.item()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        record = {
            key: (
                json_value(_mapping(value))
                if key == "forecast"
                else json_value(value)
            )
            for key, value in row.items()
        }
        records.append(record)
    return canonical_sha256(records)


def evaluate_simulation_ledger(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate coherent outcomes, duration, totals and statistic distributions."""

    required = {"event_id", "fight_id", "date", "actual_outcome", "forecast"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"simulation ledger is missing columns: {missing}")
    if frame["fight_id"].duplicated().any():
        raise ValueError("simulation ledger must contain one row per physical fight")
    predictions: list[dict[str, float]] = []
    winner_truth: list[int] = []
    winner_probability: list[float] = []
    joint_truth: list[str] = []
    method_truth: list[str] = []
    method_predictions: list[dict[str, float]] = []
    duration_scores: list[float] = []
    duration_integrated_brier_scores: list[float] = []
    total_log_losses: list[float] = []
    total_omitted_push_or_no_contest = 0
    total_missing_forecast_line = 0
    total_missing_actual_duration = 0
    statistic_scores: dict[str, dict[str, list[float]]] = {}
    process_mcses: list[float] = []
    parameter_widths: list[float] = []
    omitted_winner = 0
    for row in frame.to_dict("records"):
        probabilities = _outcome_probabilities(row["forecast"])
        actual = str(row["actual_outcome"])
        if actual not in probabilities:
            raise ValueError(f"actual outcome is absent from forecast support: {actual}")
        predictions.append(probabilities)
        joint_truth.append(actual)
        method_truth.append(_actual_method(actual))
        method_predictions.append(_method_probabilities(probabilities))
        if actual.startswith("red_") or actual.startswith("blue_"):
            winner_truth.append(int(actual.startswith("red_")))
            winner_probability.append(_red_win_probability(probabilities))
        else:
            omitted_winner += 1
        duration = _finite_number(row.get("actual_duration_seconds"))
        if duration is not None:
            score = _duration_crps(duration, row["forecast"])
            if score is not None:
                duration_scores.append(score)
            integrated_brier = _duration_integrated_brier(
                duration, row["forecast"]
            )
            if integrated_brier is not None:
                duration_integrated_brier_scores.append(integrated_brier)

        if "market_total_line_rounds" in frame.columns:
            line_rounds = _finite_number(row.get("market_total_line_rounds"))
            if line_rounds is not None:
                if duration is None:
                    total_missing_actual_duration += 1
                elif actual == "no_contest" or math.isclose(
                    duration,
                    line_rounds * 300.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    total_omitted_push_or_no_contest += 1
                else:
                    over_probability = _total_over_probability(
                        row["forecast"], line_rounds
                    )
                    if over_probability is None:
                        total_missing_forecast_line += 1
                    else:
                        over_probability = min(
                            max(over_probability, 1e-12), 1.0 - 1e-12
                        )
                        observed_over = duration > line_rounds * 300.0
                        total_log_losses.append(
                            -math.log(
                                over_probability
                                if observed_over
                                else 1.0 - over_probability
                            )
                        )

        for statistic, (support, mass) in _statistic_distributions(
            row["forecast"]
        ).items():
            actual_statistic = _finite_number(row.get(f"actual_{statistic}"))
            if actual_statistic is None:
                continue
            crps = _discrete_crps(actual_statistic, support, mass)
            if crps is None:
                continue
            lower = _weighted_quantile(support, mass, 0.05)
            upper = _weighted_quantile(support, mass, 0.95)
            accumulator = statistic_scores.setdefault(
                statistic,
                {
                    "crps": [],
                    "coverage": [],
                    "interval_width": [],
                    "observed": [],
                    "predictive": [],
                },
            )
            accumulator["crps"].append(crps)
            accumulator["coverage"].append(
                float(lower <= actual_statistic <= upper)
            )
            accumulator["interval_width"].append(upper - lower)
            accumulator["observed"].append(actual_statistic)
            accumulator["predictive"].append(float(np.sum(support * mass)))

        value = _mapping(row["forecast"])
        for uncertainty in list(value.get("uncertainty") or []):
            item = dict(uncertainty)
            if item.get("metric") in {"red_win", "red_win_probability"}:
                process_mcses.append(float(item.get("process_mcse", math.nan)))
                parameter_widths.append(
                    float(item.get("parameter_p975", math.nan))
                    - float(item.get("parameter_p025", math.nan))
                )
    joint_loss = _multiclass_log_loss(joint_truth, predictions)
    method_loss = _multiclass_log_loss(method_truth, method_predictions)
    statistic_checks: dict[str, object] = {}
    for statistic in sorted(statistic_scores):
        scores = statistic_scores[statistic]
        observed = np.asarray(scores["observed"], dtype=float)
        predictive = np.asarray(scores["predictive"], dtype=float)
        statistic_checks[statistic] = {
            "n": len(scores["crps"]),
            "count_distribution_crps": float(np.mean(scores["crps"])),
            "interval_90_coverage": float(np.mean(scores["coverage"])),
            "mean_interval_90_width": float(np.mean(scores["interval_width"])),
            "observed_mean": float(np.mean(observed)),
            "predictive_mean": float(np.mean(predictive)),
            "predictive_minus_observed_mean": float(
                np.mean(predictive - observed)
            ),
        }
    predictive_rows = posterior_predictive_rows(frame)
    posterior_checks = _diagnostic_summary(predictive_rows)
    for statistic, existing in statistic_checks.items():
        enhanced = posterior_checks.get(statistic)
        if isinstance(enhanced, Mapping):
            existing.update(
                {
                    key: value
                    for key, value in enhanced.items()
                    if key not in {"n", "mean_crps", "predictive_minus_observed_mean"}
                }
            )
    return {
        "n_fights": int(len(frame)),
        "primary_joint_side_method_log_loss": joint_loss,
        "winner": _binary_metrics(winner_truth, winner_probability),
        "winner_omitted_draw_or_no_contest": omitted_winner,
        "method_log_loss": method_loss,
        "duration_crps_seconds": (
            float(np.mean(duration_scores)) if duration_scores else None
        ),
        "duration_scored_fights": len(duration_scores),
        "duration_integrated_brier": (
            float(np.mean(duration_integrated_brier_scores))
            if duration_integrated_brier_scores
            else None
        ),
        "duration_integrated_brier_scored_fights": len(
            duration_integrated_brier_scores
        ),
        "available_totals_log_loss": (
            float(np.mean(total_log_losses)) if total_log_losses else None
        ),
        "available_totals_scored_fights": len(total_log_losses),
        "available_totals_omitted_push_or_no_contest": (
            total_omitted_push_or_no_contest
        ),
        "available_totals_missing_forecast_line": total_missing_forecast_line,
        "available_totals_missing_actual_duration": total_missing_actual_duration,
        "count_distribution_predictive_checks": statistic_checks,
        "posterior_predictive_checks": posterior_checks,
        "posterior_predictive_diagnostic_rows": len(predictive_rows),
        "posterior_predictive_pvalue_warning": (
            "PIT KS/CvM p-values are nominal iid diagnostics, not probabilities "
            "that the simulator generated the observations; event-card clustering "
            "and multiple comparisons require cautious interpretation"
        ),
        "mean_reported_process_mcse": (
            float(np.nanmean(process_mcses)) if process_mcses else None
        ),
        "mean_parameter_interval_width": (
            float(np.nanmean(parameter_widths)) if parameter_widths else None
        ),
    }


def _baseline_metrics(frame: pd.DataFrame, column: str) -> dict[str, object]:
    rows = frame.loc[
        frame[column].notna()
        & frame["actual_outcome"].astype(str).str.startswith(("red_", "blue_"))
    ]
    truth = rows["actual_outcome"].astype(str).str.startswith("red_").astype(int)
    return _binary_metrics(truth.tolist(), pd.to_numeric(rows[column], errors="raise").tolist())


def _optional_outcome_probabilities(value: object) -> dict[str, float] | None:
    """Parse an optional full forecast or a bare outcome-probability mapping."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    mapped = _mapping(value)
    if "outcome_probabilities" in mapped or "outcome_counts" in mapped:
        return _outcome_probabilities(mapped)
    probabilities = {str(key): float(item) for key, item in mapped.items()}
    if (
        not probabilities
        or any(
            not math.isfinite(probability) or probability < 0.0
            for probability in probabilities.values()
        )
        or abs(sum(probabilities.values()) - 1.0) > 1e-8
    ):
        raise ValueError("joint comparator probabilities must be finite and sum to one")
    return probabilities


def _joint_forecast_metrics(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, float | int | None]:
    truth: list[str] = []
    predictions: list[dict[str, float]] = []
    for row in frame.to_dict("records"):
        probabilities = _optional_outcome_probabilities(row.get(column))
        if probabilities is None:
            continue
        truth.append(str(row["actual_outcome"]))
        predictions.append(probabilities)
    return {
        "n": len(truth),
        "joint_side_by_method_log_loss": _multiclass_log_loss(
            truth, predictions
        ),
    }


def _joint_event_card_paired_interval(
    frame: pd.DataFrame,
    challenger_column: str,
    baseline_column: str,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Paired event-card interval for the primary joint forecast metric."""

    losses: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        challenger = _optional_outcome_probabilities(row.get(challenger_column))
        baseline = _optional_outcome_probabilities(row.get(baseline_column))
        if challenger is None or baseline is None:
            continue
        actual = str(row["actual_outcome"])
        difference = -math.log(max(challenger.get(actual, 0.0), 1e-12)) + math.log(
            max(baseline.get(actual, 0.0), 1e-12)
        )
        losses.append(
            {"event_id": str(row["event_id"]), "loss_difference": difference}
        )
    if not losses:
        raise ValueError("paired joint comparison has no jointly covered fights")
    loss_frame = pd.DataFrame(losses)
    cards = [
        group["loss_difference"].to_numpy(float)
        for _, group in loss_frame.groupby("event_id", sort=True)
    ]
    rng = np.random.Generator(np.random.PCG64DXSM(random_seed))
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(cards), size=len(cards))
        estimates[index] = np.concatenate([cards[item] for item in selected]).mean()
    return {
        "n_fights": len(losses),
        "n_events": len(cards),
        "challenger_minus_baseline_log_loss": float(
            loss_frame["loss_difference"].mean()
        ),
        "interval_p025": float(np.quantile(estimates, 0.025)),
        "interval_p975": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": replicates,
    }


def event_card_paired_interval(
    frame: pd.DataFrame,
    challenger_column: str,
    baseline_column: str,
    *,
    replicates: int = 2000,
    random_seed: int = 2903,
) -> dict[str, float | int]:
    """Paired 95% interval for winner log-loss difference by resampled cards."""

    required = {"event_id", "actual_outcome", challenger_column, baseline_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"paired comparison is missing columns: {missing}")
    rows = frame.loc[
        frame["actual_outcome"].astype(str).str.startswith(("red_", "blue_"))
        & frame[challenger_column].notna()
        & frame[baseline_column].notna()
    ].copy()
    if rows.empty:
        raise ValueError("paired comparison has no decisive, jointly covered fights")
    truth = rows["actual_outcome"].astype(str).str.startswith("red_").astype(float)
    challenger = np.clip(pd.to_numeric(rows[challenger_column]).to_numpy(float), 1e-12, 1 - 1e-12)
    baseline = np.clip(pd.to_numeric(rows[baseline_column]).to_numpy(float), 1e-12, 1 - 1e-12)
    rows["_loss_difference"] = -(
        truth.to_numpy() * np.log(challenger)
        + (1 - truth.to_numpy()) * np.log1p(-challenger)
    ) + (
        truth.to_numpy() * np.log(baseline)
        + (1 - truth.to_numpy()) * np.log1p(-baseline)
    )
    rows["_brier_difference"] = (challenger - truth.to_numpy()) ** 2 - (
        baseline - truth.to_numpy()
    ) ** 2
    cards = [
        group[["_loss_difference", "_brier_difference"]].to_numpy(float)
        for _, group in rows.groupby("event_id", sort=True)
    ]
    rng = np.random.Generator(np.random.PCG64DXSM(random_seed))
    estimates = np.empty((replicates, 2), dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(cards), size=len(cards))
        sample = np.concatenate([cards[item] for item in selected])
        estimates[index] = sample.mean(axis=0)
    return {
        "n_fights": int(len(rows)),
        "n_events": len(cards),
        "challenger_minus_baseline_log_loss": float(rows["_loss_difference"].mean()),
        "interval_p025": float(np.quantile(estimates[:, 0], 0.025)),
        "interval_p975": float(np.quantile(estimates[:, 0], 0.975)),
        "challenger_minus_baseline_brier": float(rows["_brier_difference"].mean()),
        "brier_interval_p025": float(np.quantile(estimates[:, 1], 0.025)),
        "brier_interval_p975": float(np.quantile(estimates[:, 1], 0.975)),
        "bootstrap_replicates": replicates,
    }


def _probability_logit(values: Sequence[float], *, clip: float = 1e-6) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("stack probabilities must be one-dimensional")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("stack probabilities must be finite and in [0, 1]")
    bounded = np.clip(probabilities, clip, 1.0 - clip)
    return np.log(bounded) - np.log1p(-bounded)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def fit_nonnegative_logit_stack(
    model_probability: Sequence[float],
    simulation_probability: Sequence[float],
    truth: Sequence[int],
    *,
    l2_penalty: float = 0.01,
) -> dict[str, object]:
    """Fit the predeclared zero-intercept, nonnegative winner stack.

    Regularization is centered on ``(beta_model=1, beta_sim=0)`` so weak or
    redundant simulation evidence shrinks back toward the incumbent rather
    than toward an uncalibrated 50/50 prediction.
    """

    if not math.isfinite(l2_penalty) or l2_penalty < 0:
        raise ValueError("stack l2_penalty must be finite and nonnegative")
    model_logit = _probability_logit(model_probability)
    simulation_logit = _probability_logit(simulation_probability)
    y = np.asarray(truth, dtype=float)
    if y.ndim != 1 or len(y) != len(model_logit) or len(y) != len(simulation_logit):
        raise ValueError("stack inputs must have equal one-dimensional lengths")
    if len(y) == 0 or np.any(~np.isfinite(y)) or np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("stack truth must contain binary outcomes")
    if len(np.unique(y)) != 2:
        raise ValueError("stack fitting requires both winner classes")
    design = np.column_stack((model_logit, simulation_logit))
    center = np.asarray([1.0, 0.0], dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        combined_logit = design @ beta
        probability = _sigmoid(combined_logit)
        penalty_delta = beta - center
        value = float(
            np.mean(np.logaddexp(0.0, combined_logit) - y * combined_logit)
            + 0.5 * l2_penalty * float(penalty_delta @ penalty_delta)
        )
        gradient = (
            design.T @ (probability - y) / len(y)
            + l2_penalty * penalty_delta
        )
        return value, gradient

    optimized = minimize(
        objective,
        center,
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, None), (0.0, None)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
    )
    if not optimized.success or np.any(~np.isfinite(optimized.x)):
        raise RuntimeError(f"winner stack optimization failed: {optimized.message}")
    beta = np.maximum(np.asarray(optimized.x, dtype=float), 0.0)
    return {
        "beta_model": float(beta[0]),
        "beta_sim": float(beta[1]),
        "intercept": 0.0,
        "l2_penalty": float(l2_penalty),
        "training_fights": int(len(y)),
        "objective": float(optimized.fun),
        "optimizer_iterations": int(optimized.nit),
    }


def stacked_win_probability(
    model_probability: Sequence[float],
    simulation_probability: Sequence[float],
    *,
    beta_model: float,
    beta_sim: float,
) -> np.ndarray:
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (beta_model, beta_sim)
    ):
        raise ValueError("stack coefficients must be finite and nonnegative")
    model_logit = _probability_logit(model_probability)
    simulation_logit = _probability_logit(simulation_probability)
    if len(model_logit) != len(simulation_logit):
        raise ValueError("stack probability inputs must have equal lengths")
    return _sigmoid(beta_model * model_logit + beta_sim * simulation_logit)


def evaluate_chronological_winner_stack(
    frame: pd.DataFrame,
    *,
    min_training_fights: int = 100,
    l2_penalty: float = 0.01,
    card_bootstrap_replicates: int = 2000,
    random_seed: int = 2903,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cross-fit stack weights using only earlier out-of-fold predictions."""

    required = {
        "date",
        "event_id",
        "fight_id",
        "actual_outcome",
        "production_red_win_probability",
        "simulation_red_win_probability",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"winner stack ledger is missing columns: {missing}")
    if min_training_fights <= 0:
        raise ValueError("stack min_training_fights must be positive")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise", utc=True)
    result["stack_red_win_probability"] = np.nan
    decisive = result["actual_outcome"].astype(str).str.startswith(("red_", "blue_"))
    jointly_covered = (
        decisive
        & result["production_red_win_probability"].notna()
        & result["simulation_red_win_probability"].notna()
    )
    folds: list[dict[str, object]] = []
    for year in sorted(result.loc[jointly_covered, "date"].dt.year.unique()):
        cutoff = pd.Timestamp(f"{int(year)}-01-01", tz="UTC")
        train = result.loc[jointly_covered & result["date"].lt(cutoff)]
        test_index = result.index[
            jointly_covered & result["date"].dt.year.eq(int(year))
        ]
        fold: dict[str, object] = {
            "test_year": int(year),
            "cutoff_utc": cutoff.isoformat(),
            "training_fights": int(len(train)),
            "test_fights": int(len(test_index)),
        }
        truth = train["actual_outcome"].astype(str).str.startswith("red_").astype(int)
        if len(train) < min_training_fights or truth.nunique() < 2:
            fold["status"] = "warmup_insufficient_prior_oof_fights"
            folds.append(fold)
            continue
        fitted = fit_nonnegative_logit_stack(
            pd.to_numeric(train["production_red_win_probability"], errors="raise"),
            pd.to_numeric(train["simulation_red_win_probability"], errors="raise"),
            truth,
            l2_penalty=l2_penalty,
        )
        result.loc[test_index, "stack_red_win_probability"] = stacked_win_probability(
            pd.to_numeric(
                result.loc[test_index, "production_red_win_probability"],
                errors="raise",
            ),
            pd.to_numeric(
                result.loc[test_index, "simulation_red_win_probability"],
                errors="raise",
            ),
            beta_model=float(fitted["beta_model"]),
            beta_sim=float(fitted["beta_sim"]),
        )
        fold.update({"status": "evaluated", **fitted})
        folds.append(fold)

    covered = jointly_covered & result["stack_red_win_probability"].notna()
    comparison: dict[str, object] = {
        "metric": "winner_log_loss_brier_and_calibration",
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "model": "zero_intercept_nonnegative_logit_stack",
        "config": {
            "min_training_fights": int(min_training_fights),
            "l2_penalty": float(l2_penalty),
            "regularization_center": {"beta_model": 1.0, "beta_sim": 0.0},
            "fit_scope": "strictly_earlier_out_of_fold_calendar_years",
        },
        "n_eligible": int(jointly_covered.sum()),
        "n_covered": int(covered.sum()),
        "coverage": float(covered.sum() / jointly_covered.sum())
        if jointly_covered.any()
        else 0.0,
        "folds": folds,
    }
    if not covered.any():
        comparison.update(
            {
                "status": "insufficient_prior_out_of_fold_history",
                "stack": _binary_metrics([], []),
                "production_same_fights": _binary_metrics([], []),
                "simulation_same_fights": _binary_metrics([], []),
                "candidate_freeze_recommended": False,
            }
        )
        return result, comparison

    rows = result.loc[covered]
    comparison.update(
        {
            "status": "evaluated",
            "stack": _baseline_metrics(rows, "stack_red_win_probability"),
            "production_same_fights": _baseline_metrics(
                rows, "production_red_win_probability"
            ),
            "simulation_same_fights": _baseline_metrics(
                rows, "simulation_red_win_probability"
            ),
            "paired_event_card_interval_vs_production": event_card_paired_interval(
                rows,
                "stack_red_win_probability",
                "production_red_win_probability",
                replicates=card_bootstrap_replicates,
                random_seed=random_seed,
            ),
            "paired_event_card_interval_vs_simulation": event_card_paired_interval(
                rows,
                "stack_red_win_probability",
                "simulation_red_win_probability",
                replicates=card_bootstrap_replicates,
                random_seed=random_seed + 1,
            ),
        }
    )
    paired = dict(comparison["paired_event_card_interval_vs_production"])
    stack_metrics = dict(comparison["stack"])
    production_metrics = dict(comparison["production_same_fights"])
    intercept = stack_metrics.get("calibration_intercept")
    slope = stack_metrics.get("calibration_slope")
    checks = {
        "paired_log_loss_interval_below_zero": float(paired["interval_p975"]) < 0.0,
        "brier_not_worse": (
            stack_metrics.get("brier") is not None
            and production_metrics.get("brier") is not None
            and float(stack_metrics["brier"]) <= float(production_metrics["brier"])
        ),
        "calibration_intercept_within_0_05": (
            intercept is not None and abs(float(intercept)) <= 0.05
        ),
        "calibration_slope_within_0_85_1_15": (
            slope is not None and 0.85 <= float(slope) <= 1.15
        ),
        "at_least_three_evaluated_folds": sum(
            fold.get("status") == "evaluated" for fold in folds
        )
        >= 3,
    }
    comparison["retrospective_checks"] = checks
    comparison["candidate_freeze_recommended"] = all(checks.values())
    return result, comparison


def _market_total_comparison_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Return settleable rows covered by both the market and simulator.

    ``eligible`` is deliberately limited to timestamp-aligned market rows with
    a known duration and valid probability. Pushes and no-contests are omitted
    under the existing full-fight-total settlement contract. ``covered`` then
    requires the simulator to expose the identical market line.
    """

    rows: list[dict[str, object]] = []
    eligible = 0
    omitted_push_or_no_contest = 0
    for row in frame.to_dict("records"):
        line_rounds = _finite_number(row.get("market_total_line_rounds"))
        market_probability = _finite_number(
            row.get("market_total_over_probability")
        )
        duration = _finite_number(row.get("actual_duration_seconds"))
        if line_rounds is None or market_probability is None or duration is None:
            continue
        if not 0.0 <= market_probability <= 1.0:
            raise ValueError("market total probability must be in [0, 1]")
        threshold_seconds = line_rounds * 300.0
        if str(row.get("actual_outcome")) == "no_contest" or math.isclose(
            duration,
            threshold_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            omitted_push_or_no_contest += 1
            continue
        eligible += 1
        simulation_probability = _total_over_probability(
            row["forecast"], line_rounds
        )
        if simulation_probability is None:
            continue
        if not 0.0 <= simulation_probability <= 1.0:
            raise ValueError("simulation total probability must be in [0, 1]")
        rows.append(
            {
                "event_id": str(row["event_id"]),
                "fight_id": str(row["fight_id"]),
                "observed_over": int(duration > threshold_seconds),
                "simulation_over_probability": float(simulation_probability),
                "market_over_probability": float(market_probability),
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=(
                "event_id",
                "fight_id",
                "observed_over",
                "simulation_over_probability",
                "market_over_probability",
            ),
        ),
        eligible,
        omitted_push_or_no_contest,
    )


def _total_event_card_paired_interval(
    frame: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Paired event-card intervals for total log loss and Brier score."""

    if frame.empty:
        raise ValueError("paired total comparison has no jointly covered fights")
    truth = frame["observed_over"].to_numpy(float)
    simulation = np.clip(
        frame["simulation_over_probability"].to_numpy(float),
        1e-12,
        1.0 - 1e-12,
    )
    market = np.clip(
        frame["market_over_probability"].to_numpy(float),
        1e-12,
        1.0 - 1e-12,
    )
    result = frame[["event_id"]].copy()
    result["log_loss_difference"] = -(
        truth * np.log(simulation) + (1.0 - truth) * np.log1p(-simulation)
    ) + (
        truth * np.log(market) + (1.0 - truth) * np.log1p(-market)
    )
    result["brier_difference"] = (simulation - truth) ** 2 - (
        market - truth
    ) ** 2
    cards = [
        group[["log_loss_difference", "brier_difference"]].to_numpy(float)
        for _, group in result.groupby("event_id", sort=True)
    ]
    rng = np.random.Generator(np.random.PCG64DXSM(random_seed))
    estimates = np.empty((replicates, 2), dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(cards), size=len(cards))
        estimates[index] = np.concatenate([cards[item] for item in selected]).mean(
            axis=0
        )
    return {
        "n_fights": int(len(result)),
        "n_events": len(cards),
        "challenger_minus_baseline_log_loss": float(
            result["log_loss_difference"].mean()
        ),
        "log_loss_interval_p025": float(np.quantile(estimates[:, 0], 0.025)),
        "log_loss_interval_p975": float(np.quantile(estimates[:, 0], 0.975)),
        "challenger_minus_baseline_brier": float(
            result["brier_difference"].mean()
        ),
        "brier_interval_p025": float(np.quantile(estimates[:, 1], 0.025)),
        "brier_interval_p975": float(np.quantile(estimates[:, 1], 0.975)),
        "bootstrap_replicates": replicates,
    }


def add_simulation_win_probability_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the decisive red-win view from each coherent simulation forecast."""

    result = frame.copy()
    result["simulation_red_win_probability"] = [
        _red_win_probability(_outcome_probabilities(value)) for value in result["forecast"]
    ]
    return result


def _slice_reports(frame: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    for column in ("division", "sex", "scheduled_rounds", "era", "experience_band"):
        if column not in frame:
            continue
        groups: dict[str, object] = {}
        for value, group in frame.groupby(column, dropna=False, sort=True):
            groups[str(value)] = evaluate_simulation_ledger(group)
        output[column] = groups
    return output


FoldPredictor = Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.DataFrame]


def run_chronological_backtest(
    physical_fights: pd.DataFrame,
    predict_fold: FoldPredictor,
    *,
    config: BacktestConfig | None = None,
    test_filter_column: str | None = None,
) -> tuple[pd.DataFrame, BacktestReport]:
    """Run expanding calendar-year folds with a strict causal callback.

    ``predict_fold(train, test, cutoff)`` must return one row per test fight
    containing at least ``fight_id`` and ``forecast``.  The function should fit
    snapshots, global mechanics and bootstrap members using only ``train``.
    This orchestrator verifies IDs and dates before merging labels.  When
    ``test_filter_column`` is supplied, every earlier row remains available to
    the expanding training window while only truthy rows in that column are
    evaluated.  This makes bounded research runs causal without discarding
    historical training information.
    """

    config = config or BacktestConfig()
    required = {"date", "event_id", "fight_id", "actual_outcome"}
    missing = sorted(required - set(physical_fights.columns))
    if missing:
        raise ValueError(f"physical backtest table is missing columns: {missing}")
    if test_filter_column is not None and test_filter_column not in physical_fights:
        raise ValueError(
            f"backtest test filter column is missing: {test_filter_column}"
        )
    frame = physical_fights.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    if frame["fight_id"].duplicated().any():
        raise ValueError("backtest input must contain one row per physical fight")
    frame = frame.sort_values(["date", "event_id", "fight_id"], kind="stable")
    years = sorted(frame["date"].dt.year.unique())
    first = config.first_test_year if config.first_test_year is not None else years[0] + 1
    last = config.last_test_year if config.last_test_year is not None else years[-1]
    ledgers: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for year in years:
        if year < first or year > last:
            continue
        cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
        train = frame.loc[frame["date"] < cutoff].copy()
        test = frame.loc[frame["date"].dt.year.eq(year)].copy()
        if test_filter_column is not None:
            test = test.loc[test[test_filter_column].fillna(False).astype(bool)].copy()
        if len(train) < config.min_training_fights or test.empty:
            continue
        predictions = predict_fold(train.copy(), test.copy(), cutoff)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predict_fold must return a pandas DataFrame")
        if not {"fight_id", "forecast"} <= set(predictions.columns):
            raise ValueError("fold predictions require fight_id and forecast")
        if predictions["fight_id"].duplicated().any():
            raise ValueError(f"fold {year} returned duplicate fight IDs")
        expected = set(test["fight_id"])
        observed = set(predictions["fight_id"])
        if observed != expected:
            raise ValueError(
                f"fold {year} prediction IDs differ: missing={sorted(expected-observed)[:3]}, "
                f"unexpected={sorted(observed-expected)[:3]}"
            )
        overlap = (set(predictions.columns) & set(test.columns)) - {"fight_id"}
        if overlap:
            raise ValueError(
                f"fold {year} predictions overwrite test columns: {sorted(overlap)}"
            )
        merged = test.merge(
            predictions, on="fight_id", how="left", validate="one_to_one"
        )
        merged["fold_year"] = year
        ledgers.append(merged)
        fold_metrics = evaluate_simulation_ledger(merged)
        folds.append(
            {
                "test_year": int(year),
                "cutoff_utc": cutoff.isoformat(),
                "training_fights": int(len(train)),
                "test_fights": int(len(test)),
                "training_through": train["date"].max().date().isoformat(),
                "metrics": fold_metrics,
            }
        )
    if not ledgers:
        raise ValueError("chronological backtest produced no eligible folds")
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger = add_simulation_win_probability_column(ledger)
    comparisons: dict[str, object] = {}
    stack_warning: str | None = None
    if "production_red_win_probability" in ledger:
        ledger, stack_comparison = evaluate_chronological_winner_stack(
            ledger,
            min_training_fights=config.stack_min_training_fights,
            l2_penalty=config.stack_l2_penalty,
            card_bootstrap_replicates=config.card_bootstrap_replicates,
            random_seed=config.random_seed,
        )
        comparisons["production_simulation_stack"] = stack_comparison
        if stack_comparison.get("status") != "evaluated":
            stack_warning = "winner_stack_insufficient_prior_out_of_fold_history"
    else:
        stack_warning = "winner_stack_unavailable_without_production_baseline"
    for name, column in (
        ("production_winner", "production_red_win_probability"),
        ("competing_risk", "outcome_model_red_win_probability"),
        ("timestamped_market", "market_red_win_probability"),
    ):
        if column not in ledger:
            continue
        eligible = ledger["actual_outcome"].astype(str).str.startswith(
            ("red_", "blue_")
        )
        covered = eligible & ledger[column].notna()
        n_eligible = int(eligible.sum())
        n_covered = int(covered.sum())
        comparisons[name] = {
            "metric": "winner_log_loss",
            "n_eligible": n_eligible,
            "n_covered": n_covered,
            "coverage": n_covered / n_eligible if n_eligible else 0.0,
            "baseline": _baseline_metrics(ledger.loc[covered], column),
            "simulation_same_fights": _baseline_metrics(
                ledger.loc[covered], "simulation_red_win_probability"
            ),
            "paired_event_card_interval": (
                event_card_paired_interval(
                    ledger.loc[covered],
                    "simulation_red_win_probability",
                    column,
                    replicates=config.card_bootstrap_replicates,
                    random_seed=config.random_seed,
                )
                if n_covered
                else None
            ),
        }
    for name, column in (
        ("competing_risk_joint", "outcome_model_forecast"),
        ("population_joint", "population_forecast"),
        ("division_joint", "division_forecast"),
    ):
        if column not in ledger:
            continue
        covered = ledger[column].map(
            lambda value: _optional_outcome_probabilities(value) is not None
        )
        n_eligible = int(len(ledger))
        n_covered = int(covered.sum())
        paired = (
            _joint_event_card_paired_interval(
                ledger.loc[covered],
                "forecast",
                column,
                replicates=config.card_bootstrap_replicates,
                random_seed=config.random_seed,
            )
            if n_covered
            else None
        )
        comparisons[name] = {
            "metric": "joint_side_by_method_log_loss",
            "n_eligible": n_eligible,
            "n_covered": n_covered,
            "coverage": n_covered / n_eligible if n_eligible else 0.0,
            "baseline": _joint_forecast_metrics(ledger.loc[covered], column),
            "simulation_same_fights": _joint_forecast_metrics(
                ledger.loc[covered], "forecast"
            ),
            "paired_event_card_interval": paired,
        }
    if {
        "market_total_line_rounds",
        "market_total_over_probability",
    } <= set(ledger.columns):
        total_rows, total_eligible, total_omitted = _market_total_comparison_rows(
            ledger
        )
        total_covered = len(total_rows)
        total_truth = total_rows["observed_over"].astype(int).tolist()
        comparisons["timestamped_market_totals"] = {
            "metric": "full_fight_total_log_loss_and_brier",
            "n_eligible": int(total_eligible),
            "n_covered": int(total_covered),
            "coverage": (
                total_covered / total_eligible if total_eligible else 0.0
            ),
            "omitted_push_or_no_contest": int(total_omitted),
            "baseline": _binary_metrics(
                total_truth,
                total_rows["market_over_probability"].tolist(),
            ),
            "simulation_same_fights": _binary_metrics(
                total_truth,
                total_rows["simulation_over_probability"].tolist(),
            ),
            "paired_event_card_interval": (
                _total_event_card_paired_interval(
                    total_rows,
                    replicates=config.card_bootstrap_replicates,
                    random_seed=config.random_seed,
                )
                if total_covered
                else None
            ),
        }
    warnings: list[str] = []
    if len(folds) < 3:
        warnings.append("fewer_than_three_eligible_calendar_year_folds")
    if "market_red_win_probability" not in ledger or ledger["market_red_win_probability"].notna().mean() < 0.5:
        warnings.append("timestamped_market_coverage_below_half")
    if stack_warning is not None:
        warnings.append(stack_warning)
    body: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "primary_metric": "joint_side_by_method_log_loss",
        "config": asdict(config),
        "folds": folds,
        "aggregate": evaluate_simulation_ledger(ledger),
        "slices": _slice_reports(ledger),
        "comparisons": comparisons,
        "coverage_warnings": warnings,
        "ledger_sha256": _ledger_fingerprint(ledger),
    }
    report = BacktestReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluation_version=EVALUATION_VERSION,
        candidate_only=True,
        production_enabled=False,
        execution_enabled=False,
        primary_metric="joint_side_by_method_log_loss",
        config=asdict(config),
        folds=tuple(folds),
        aggregate=dict(body["aggregate"]),
        slices=dict(body["slices"]),
        comparisons=comparisons,
        coverage_warnings=tuple(warnings),
        ledger_sha256=str(body["ledger_sha256"]),
        report_sha256=canonical_sha256(body),
    ).validate()
    return ledger, report


def repeated_seed_summary(
    ledgers: Sequence[pd.DataFrame],
) -> dict[str, object]:
    """Quantify end-to-end simulation noise across independently seeded ledgers."""

    if len(ledgers) < 2:
        raise ValueError("at least two independently seeded ledgers are required")
    keys = [set(frame["fight_id"]) for frame in ledgers]
    if any(key != keys[0] for key in keys[1:]):
        raise ValueError("repeated-seed ledgers must cover identical fights")
    joint = [
        float(evaluate_simulation_ledger(frame)["primary_joint_side_method_log_loss"])
        for frame in ledgers
    ]
    winner = [
        float(evaluate_simulation_ledger(frame)["winner"]["log_loss"])
        for frame in ledgers
    ]
    return {
        "seed_repeats": len(ledgers),
        "joint_log_loss_mean": float(np.mean(joint)),
        "joint_log_loss_sd": float(np.std(joint, ddof=1)),
        "winner_log_loss_mean": float(np.mean(winner)),
        "winner_log_loss_sd": float(np.std(winner, ddof=1)),
    }

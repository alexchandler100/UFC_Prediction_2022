"""Candidate discrete-time competing-risk model for UFC method and totals markets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bayesian_total_calibration import fit_total_calibration
from fight_semantics import method_bucket, schedule_from_row


INTERVAL_SECONDS = 150
CONTINUE = "continue"
TERMINAL_OUTCOMES = (
    "fighter_ko_tko",
    "fighter_submission",
    "fighter_decision",
    "fighter_other",
    "opponent_ko_tko",
    "opponent_submission",
    "opponent_decision",
    "opponent_other",
)
MODEL_CLASSES = (CONTINUE, *TERMINAL_OUTCOMES)
TIME_FEATURES = (
    "elapsed_before_seconds",
    "remaining_seconds",
    "fraction_elapsed",
    "scheduled_rounds",
    "is_final_interval",
)
TOTAL_ROUND_THRESHOLDS = {
    "over_0_5_rounds": 150,
    "over_1_5_rounds": 450,
    "over_2_5_rounds": 750,
    "over_3_5_rounds": 1050,
    "over_4_5_rounds": 1350,
}


def _method_bucket(value: object) -> str:
    return method_bucket(value)


def _scheduled_rounds(row: pd.Series) -> int | None:
    return schedule_from_row(row)[0]


def _schedule_basis(row: pd.Series) -> str:
    return schedule_from_row(row)[1]


def _terminal_label(row: pd.Series) -> str:
    try:
        winner = "fighter" if int(row["target"]) == 1 else "opponent"
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("outcome target must be a binary winner label") from error
    return f"{winner}_{_method_bucket(row.get('label_method'))}"


def _binary_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    bins = np.minimum((probability * 10).astype(int), 9)
    calibration_error = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            calibration_error += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(truth[mask].mean())
            )
    return {
        "n": int(len(truth)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - truth) ** 2)),
        "accuracy": float(accuracy_score(truth, probability >= 0.5)),
        "expected_calibration_error": calibration_error,
    }


def _multiclass_log_loss(
    truth: Iterable[str], probability: np.ndarray, labels: Iterable[str]
) -> float:
    """Compute labeled log loss without sklearn's lexicographic-column assumption."""

    ordered_labels = tuple(labels)
    label_index = {label: index for index, label in enumerate(ordered_labels)}
    matrix = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    indices = np.asarray([label_index[str(value)] for value in truth], dtype=int)
    return float(-np.log(matrix[np.arange(len(indices)), indices]).mean())


@dataclass(frozen=True)
class CompetingRiskPrediction:
    terminal_probabilities: dict[str, float]
    survival_after_seconds: dict[int, float]
    scheduled_rounds: int

    @property
    def fighter_win_probability(self) -> float:
        return sum(
            probability
            for outcome, probability in self.terminal_probabilities.items()
            if outcome.startswith("fighter_")
        )

    @property
    def method_probabilities(self) -> dict[str, float]:
        return {
            method: sum(
                probability
                for outcome, probability in self.terminal_probabilities.items()
                if outcome.endswith(f"_{method}")
            )
            for method in ("ko_tko", "submission", "decision", "other")
        }

    def probability_over_seconds(self, threshold_seconds: int) -> float | None:
        horizon = self.scheduled_rounds * 300
        if threshold_seconds <= 0 or threshold_seconds >= horizon:
            return None
        eligible = [
            second
            for second in self.survival_after_seconds
            if second <= threshold_seconds
        ]
        if not eligible:
            return 1.0
        return self.survival_after_seconds[max(eligible)]


class DiscreteTimeOutcomeModel:
    """Multinomial hazards whose probabilities form one coherent fight outcome."""

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        interval_seconds: int = INTERVAL_SECONDS,
        c_value: float = 0.1,
    ):
        if interval_seconds <= 0 or 300 % interval_seconds:
            raise ValueError("interval_seconds must divide a five-minute round")
        if c_value <= 0:
            raise ValueError("c_value must be positive")
        self.feature_columns = tuple(feature_columns)
        if not self.feature_columns:
            raise ValueError("at least one point-in-time feature is required")
        self.interval_seconds = int(interval_seconds)
        self.c_value = float(c_value)
        self.pipeline: Pipeline | None = None
        self.training_fights = 0
        self.training_risk_rows = 0
        self.omitted_unknown_schedule = 0
        self.total_calibration_artifact: dict[str, object] | None = None

    @property
    def model_columns(self) -> tuple[str, ...]:
        return (*self.feature_columns, *TIME_FEATURES)

    def _risk_rows(self, fights: pd.DataFrame) -> pd.DataFrame:
        required = {
            "target",
            "label_method",
            "label_total_fight_seconds",
            *self.feature_columns,
        }
        missing = sorted(required - set(fights.columns))
        if missing:
            raise ValueError(f"outcome training data is missing columns: {missing}")
        output: list[dict[str, object]] = []
        omitted = 0
        for _, fight in fights.iterrows():
            rounds = _scheduled_rounds(fight)
            duration = pd.to_numeric(
                pd.Series([fight.get("label_total_fight_seconds")]), errors="coerce"
            ).iloc[0]
            if rounds is None or pd.isna(duration):
                omitted += 1
                continue
            horizon = rounds * 300
            duration = float(duration)
            if not 0 < duration <= horizon:
                omitted += 1
                continue
            terminal = _terminal_label(fight)
            terminal_interval = min(
                math.ceil(duration / self.interval_seconds),
                math.ceil(horizon / self.interval_seconds),
            )
            for interval_number in range(1, terminal_interval + 1):
                elapsed_before = (interval_number - 1) * self.interval_seconds
                row = {
                    column: fight.get(column) for column in self.feature_columns
                }
                row.update(
                    {
                        "elapsed_before_seconds": elapsed_before,
                        "remaining_seconds": horizon - elapsed_before,
                        "fraction_elapsed": elapsed_before / horizon,
                        "scheduled_rounds": rounds,
                        "is_final_interval": int(
                            interval_number
                            == math.ceil(horizon / self.interval_seconds)
                        ),
                        "risk_target": (
                            terminal
                            if interval_number == terminal_interval
                            else CONTINUE
                        ),
                    }
                )
                output.append(row)
        self.omitted_unknown_schedule = omitted
        return pd.DataFrame(output, columns=[*self.model_columns, "risk_target"])

    def fit(self, fights: pd.DataFrame) -> "DiscreteTimeOutcomeModel":
        risk = self._risk_rows(fights)
        if risk.empty or risk["risk_target"].nunique() < 2:
            raise ValueError("competing-risk training needs at least two outcome classes")
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant", fill_value=0.0, keep_empty_features=True
                    ),
                ),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        penalty="l2",
                        C=self.c_value,
                        fit_intercept=True,
                        max_iter=30_000,
                        random_state=48,
                    ),
                ),
            ]
        )
        self.pipeline.fit(risk[list(self.model_columns)], risk["risk_target"])
        self.training_fights = len(fights) - self.omitted_unknown_schedule
        self.training_risk_rows = len(risk)
        return self

    def predict(self, feature_row: pd.Series | dict[str, object], scheduled_rounds: int) -> CompetingRiskPrediction:
        if self.pipeline is None:
            raise RuntimeError("competing-risk model must be fitted before prediction")
        if scheduled_rounds not in {1, 2, 3, 4, 5}:
            raise ValueError("scheduled_rounds must be between one and five")
        horizon = scheduled_rounds * 300
        rows: list[dict[str, object]] = []
        for elapsed in range(0, horizon, self.interval_seconds):
            row = {column: feature_row.get(column) for column in self.feature_columns}
            row.update(
                {
                    "elapsed_before_seconds": elapsed,
                    "remaining_seconds": horizon - elapsed,
                    "fraction_elapsed": elapsed / horizon,
                    "scheduled_rounds": scheduled_rounds,
                    "is_final_interval": int(
                        elapsed + self.interval_seconds >= horizon
                    ),
                }
            )
            rows.append(row)
        risk = pd.DataFrame(rows, columns=self.model_columns)
        probabilities = self.pipeline.predict_proba(risk)
        classes = tuple(str(value) for value in self.pipeline.named_steps["model"].classes_)
        terminal = {outcome: 0.0 for outcome in TERMINAL_OUTCOMES}
        survival = 1.0
        survival_curve: dict[int, float] = {}
        final_row = {name: 0.0 for name in MODEL_CLASSES}
        for index, values in enumerate(probabilities):
            conditional = dict(zip(classes, values))
            for outcome in TERMINAL_OUTCOMES:
                terminal[outcome] += survival * float(conditional.get(outcome, 0.0))
            survival *= float(conditional.get(CONTINUE, 0.0))
            elapsed_after = min((index + 1) * self.interval_seconds, horizon)
            survival_curve[elapsed_after] = survival
            final_row.update({key: float(value) for key, value in conditional.items()})
        # A finite grid can retain a small learned "continue" mass at the
        # scheduled horizon. Allocate it to decision sides using the model's
        # final conditional decision split so every quoted market is coherent.
        decision_mass = final_row.get("fighter_decision", 0.0) + final_row.get(
            "opponent_decision", 0.0
        )
        fighter_share = (
            final_row.get("fighter_decision", 0.0) / decision_mass
            if decision_mass > 0
            else 0.5
        )
        terminal["fighter_decision"] += survival * fighter_share
        terminal["opponent_decision"] += survival * (1.0 - fighter_share)
        survival_curve[horizon] = 0.0
        total = sum(terminal.values())
        if total <= 0:
            raise RuntimeError("competing-risk prediction has no terminal probability")
        terminal = {key: value / total for key, value in terminal.items()}
        return CompetingRiskPrediction(terminal, survival_curve, scheduled_rounds)


def evaluate_outcome_model(
    training_data: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
) -> tuple[DiscreteTimeOutcomeModel, dict[str, object]]:
    """Select regularization in the past, then score one untouched future holdout."""

    frame = training_data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["date", "event_id", "bout_order", "fight_id"], kind="stable")
    latest = frame["date"].max()
    frame = frame[frame["date"] >= latest - pd.DateOffset(years=10)].reset_index(drop=True)
    if len(frame) < 1000:
        raise ValueError("outcome evaluation requires at least 1,000 recent fights")
    holdout_date = frame.iloc[int(len(frame) * 0.8)]["date"]
    development = frame[frame["date"] < holdout_date].reset_index(drop=True)
    holdout = frame[frame["date"] >= holdout_date].reset_index(drop=True)
    validation_date = development.iloc[int(len(development) * 0.8)]["date"]
    inner_train = development[development["date"] < validation_date].reset_index(drop=True)
    validation = development[development["date"] >= validation_date].reset_index(drop=True)
    if min(len(inner_train), len(validation), len(holdout)) == 0:
        raise ValueError("outcome temporal split produced an empty partition")

    def score(model: DiscreteTimeOutcomeModel, sample: pd.DataFrame) -> float:
        truth: list[str] = []
        matrix: list[list[float]] = []
        for _, row in sample.iterrows():
            rounds = _scheduled_rounds(row)
            if rounds is None:
                continue
            prediction = model.predict(row, rounds)
            truth.append(_terminal_label(row))
            matrix.append(
                [prediction.terminal_probabilities[key] for key in TERMINAL_OUTCOMES]
            )
        if not truth:
            raise ValueError("outcome validation has no fights with known schedule")
        return _multiclass_log_loss(truth, np.asarray(matrix), TERMINAL_OUTCOMES)

    tuning: dict[str, float] = {}
    for candidate in c_grid:
        candidate_model = DiscreteTimeOutcomeModel(
            feature_columns, c_value=float(candidate)
        ).fit(inner_train)
        tuning[str(candidate)] = score(candidate_model, validation)
    selected_c = min((float(value) for value in c_grid), key=lambda value: tuning[str(value)])
    evaluation_model = DiscreteTimeOutcomeModel(
        feature_columns, c_value=selected_c
    ).fit(development)

    terminal_truth: list[str] = []
    terminal_matrix: list[list[float]] = []
    winner_truth: list[int] = []
    winner_probability: list[float] = []
    method_truth: list[str] = []
    method_matrix: list[list[float]] = []
    total_truth: dict[str, list[int]] = {key: [] for key in TOTAL_ROUND_THRESHOLDS}
    total_probability: dict[str, list[float]] = {
        key: [] for key in TOTAL_ROUND_THRESHOLDS
    }
    total_calibration_rows: list[dict[str, object]] = []
    methods = ("ko_tko", "submission", "decision", "other")
    for _, row in holdout.iterrows():
        rounds = _scheduled_rounds(row)
        duration = pd.to_numeric(
            pd.Series([row.get("label_total_fight_seconds")]), errors="coerce"
        ).iloc[0]
        if rounds is None or pd.isna(duration):
            continue
        prediction = evaluation_model.predict(row, rounds)
        terminal_truth.append(_terminal_label(row))
        terminal_matrix.append(
            [prediction.terminal_probabilities[key] for key in TERMINAL_OUTCOMES]
        )
        winner_truth.append(int(row["target"]))
        winner_probability.append(prediction.fighter_win_probability)
        actual_method = _method_bucket(row.get("label_method"))
        method_truth.append(actual_method)
        method_probabilities = prediction.method_probabilities
        method_matrix.append([method_probabilities[key] for key in methods])
        for name, threshold in TOTAL_ROUND_THRESHOLDS.items():
            probability = prediction.probability_over_seconds(threshold)
            if probability is not None:
                target = int(float(duration) > threshold)
                total_truth[name].append(target)
                total_probability[name].append(probability)
                total_calibration_rows.append(
                    {
                        "event_date": row["date"],
                        "event_id": row["event_id"],
                        "fight_id": row["fight_id"],
                        "line": threshold / 300.0,
                        "model_probability": probability,
                        "target": target,
                    }
                )
    if not terminal_truth:
        raise ValueError("outcome holdout has no fights with known schedule")
    terminal_array = np.asarray(terminal_matrix)
    method_array = np.asarray(method_matrix)
    development_terminal: list[str] = []
    development_methods: list[str] = []
    development_totals: dict[str, list[int]] = {
        key: [] for key in TOTAL_ROUND_THRESHOLDS
    }
    for _, row in development.iterrows():
        rounds = _scheduled_rounds(row)
        duration = pd.to_numeric(
            pd.Series([row.get("label_total_fight_seconds")]), errors="coerce"
        ).iloc[0]
        if rounds is None or pd.isna(duration):
            continue
        development_terminal.append(_terminal_label(row))
        development_methods.append(_method_bucket(row.get("label_method")))
        for name, threshold in TOTAL_ROUND_THRESHOLDS.items():
            if rounds * 300 > threshold:
                development_totals[name].append(int(float(duration) > threshold))
    terminal_counts = Counter(development_terminal)
    terminal_prior = np.asarray(
        [
            (terminal_counts[label] + 1.0)
            / (len(development_terminal) + len(TERMINAL_OUTCOMES))
            for label in TERMINAL_OUTCOMES
        ]
    )
    terminal_baseline_loss = _multiclass_log_loss(
        terminal_truth,
        np.tile(terminal_prior, (len(terminal_truth), 1)),
        TERMINAL_OUTCOMES,
    )
    method_counts = Counter(development_methods)
    method_prior = np.asarray(
        [
            (method_counts[label] + 1.0)
            / (len(development_methods) + len(methods))
            for label in methods
        ]
    )
    method_baseline_loss = _multiclass_log_loss(
        method_truth,
        np.tile(method_prior, (len(method_truth), 1)),
        methods,
    )
    joint_loss = _multiclass_log_loss(
        terminal_truth, terminal_array, TERMINAL_OUTCOMES
    )
    method_loss = _multiclass_log_loss(method_truth, method_array, methods)
    winner_metrics = _binary_metrics(
        np.asarray(winner_truth), np.asarray(winner_probability)
    )
    winner_baseline = _binary_metrics(
        np.asarray(winner_truth), np.full(len(winner_truth), 0.5)
    )
    total_metrics: dict[str, object] = {}
    for name in TOTAL_ROUND_THRESHOLDS:
        if not total_truth[name]:
            continue
        model_metrics = _binary_metrics(
            np.asarray(total_truth[name]), np.asarray(total_probability[name])
        )
        development_values = development_totals[name]
        prior = (
            (sum(development_values) + 1.0) / (len(development_values) + 2.0)
            if development_values
            else 0.5
        )
        baseline_metrics = _binary_metrics(
            np.asarray(total_truth[name]),
            np.full(len(total_truth[name]), prior),
        )
        model_metrics["development_base_rate"] = prior
        model_metrics["baseline_log_loss"] = baseline_metrics["log_loss"]
        model_metrics["model_minus_baseline_log_loss"] = (
            float(model_metrics["log_loss"]) - float(baseline_metrics["log_loss"])
        )
        total_metrics[name] = model_metrics
    total_calibration = fit_total_calibration(
        pd.DataFrame(total_calibration_rows)
    )
    report: dict[str, object] = {
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "interval_seconds": INTERVAL_SECONDS,
        "selected_c": selected_c,
        "inner_validation_log_loss_by_c": tuning,
        "development_fights": int(len(development)),
        "holdout_fights": len(terminal_truth),
        "holdout_start": holdout["date"].min().date().isoformat(),
        "holdout_end": holdout["date"].max().date().isoformat(),
        "omitted_holdout_unknown_schedule": int(len(holdout) - len(terminal_truth)),
        "holdout_schedule_basis": dict(
            sorted(Counter(_schedule_basis(row) for _, row in holdout.iterrows()).items())
        ),
        "joint_outcome": {
            "classes": list(TERMINAL_OUTCOMES),
            "log_loss": joint_loss,
            "baseline_log_loss": terminal_baseline_loss,
            "model_minus_baseline_log_loss": joint_loss - terminal_baseline_loss,
            "accuracy": float(
                accuracy_score(
                    terminal_truth,
                    np.asarray(TERMINAL_OUTCOMES)[terminal_array.argmax(axis=1)],
                )
            ),
        },
        "winner": {
            **winner_metrics,
            "baseline_log_loss": winner_baseline["log_loss"],
            "model_minus_baseline_log_loss": (
                float(winner_metrics["log_loss"])
                - float(winner_baseline["log_loss"])
            ),
        },
        "method": {
            "classes": list(methods),
            "log_loss": method_loss,
            "baseline_log_loss": method_baseline_loss,
            "model_minus_baseline_log_loss": method_loss - method_baseline_loss,
            "accuracy": float(
                accuracy_score(
                    method_truth, np.asarray(methods)[method_array.argmax(axis=1)]
                )
            ),
        },
        "total_rounds": total_metrics,
        "bayesian_total_calibration": total_calibration,
    }
    production_model = DiscreteTimeOutcomeModel(
        feature_columns, c_value=selected_c
    ).fit(frame)
    production_model.total_calibration_artifact = total_calibration
    return production_model, report

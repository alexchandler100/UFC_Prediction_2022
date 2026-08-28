"""Chronological, bout-clustered audit for opponent-adjusted observations.

This module is intentionally upstream of the simulator.  It asks whether an
opponent-adjusted observation model predicts the next card's UFCStats counts
better than context-only and marginal-fighter baselines before any candidate
is allowed to consume Monte Carlo time or influence a fight snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .parameters import CausalParameterFitter, canonical_sha256


OPPONENT_AUDIT_SCHEMA_VERSION = 1
_EPSILON = 1e-6
_EFFECT_LIMIT = math.log(2.0)
_EFFECT_ITERATIONS = 12


@dataclass(frozen=True)
class ObservationTarget:
    name: str
    kind: str
    numerator: str
    denominator: str | None
    seed: float


OBSERVATION_TARGETS = (
    ObservationTarget(
        "strike_pace", "rate", "sig_strikes_attempts", None, 7.0
    ),
    ObservationTarget(
        "strike_accuracy",
        "probability",
        "sig_strikes_landed",
        "sig_strikes_attempts",
        0.45,
    ),
    ObservationTarget(
        "takedown_pace", "rate", "takedowns_attempts", None, 0.65
    ),
    ObservationTarget(
        "takedown_accuracy",
        "probability",
        "takedowns_landed",
        "takedowns_attempts",
        0.40,
    ),
    ObservationTarget(
        "submission_pace", "rate", "sub_attempts", None, 0.28
    ),
)


@dataclass(frozen=True)
class OpponentAdjustmentAuditConfig:
    min_prior_ufc_fights: int = 3
    inner_validation_events: int = 8
    minimum_training_fights: int = 500
    ridge_grid: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    context_rate_prior_minutes: float = 300.0
    context_probability_prior_attempts: float = 160.0
    bootstrap_replicates: int = 2000
    random_seed: int = 52237
    max_runtime_seconds: float = 3300.0

    def validate(self) -> None:
        if self.min_prior_ufc_fights < 1:
            raise ValueError("min_prior_ufc_fights must be positive")
        if self.inner_validation_events < 3:
            raise ValueError("inner_validation_events must be at least three")
        if self.minimum_training_fights < 1:
            raise ValueError("minimum_training_fights must be positive")
        if not self.ridge_grid or any(
            not math.isfinite(value) or value <= 0 for value in self.ridge_grid
        ):
            raise ValueError("ridge_grid must contain finite positive values")
        if tuple(sorted(set(self.ridge_grid))) != self.ridge_grid:
            raise ValueError("ridge_grid must be strictly increasing")
        if self.context_rate_prior_minutes <= 0:
            raise ValueError("context_rate_prior_minutes must be positive")
        if self.context_probability_prior_attempts <= 0:
            raise ValueError(
                "context_probability_prior_attempts must be positive"
            )
        if not 100 <= self.bootstrap_replicates <= 10000:
            raise ValueError("bootstrap_replicates must be between 100 and 10000")
        if not 0 < self.max_runtime_seconds <= 3300:
            raise ValueError("max_runtime_seconds must be in (0, 3300]")


class OpponentAdjustmentAuditTimeLimit(RuntimeError):
    """Raised before the observation audit can exceed its declared budget."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise OpponentAdjustmentAuditTimeLimit(
            "opponent-adjustment audit reached its compute budget"
        )


def _logit(value: float) -> float:
    probability = float(np.clip(value, _EPSILON, 1.0 - _EPSILON))
    return math.log(probability / (1.0 - probability))


def _expit(value: float) -> float:
    bounded = float(np.clip(value, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-bounded))


def _era_key(value: object) -> str:
    timestamp = pd.Timestamp(value)
    start = (int(timestamp.year) // 5) * 5
    return f"{start}-{start + 4}"


def _prepare_frame(raw_fights: pd.DataFrame) -> pd.DataFrame:
    frame = CausalParameterFitter(raw_fights).raw_fights.copy()
    frame["era"] = frame["date"].map(_era_key)
    return frame.sort_values(
        ["date", "event_id", "fight_id", "fighter_id"], kind="stable"
    ).reset_index(drop=True)


def _context_key(division: object, era: object) -> tuple[str, str]:
    return str(division).strip() or "Unknown", str(era)


def _target_arrays(
    frame: pd.DataFrame,
    target: ObservationTarget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = pd.to_numeric(frame[target.numerator], errors="coerce").to_numpy(
        dtype=float
    )
    if target.kind == "rate":
        exposure = pd.to_numeric(
            frame["fight_seconds"], errors="coerce"
        ).to_numpy(dtype=float) / 60.0
        valid = (
            np.isfinite(numerator)
            & (numerator >= 0)
            & np.isfinite(exposure)
            & (exposure > 0)
        )
    else:
        if target.denominator is None:  # pragma: no cover - constant contract
            raise ValueError("probability target is missing its denominator")
        exposure = pd.to_numeric(
            frame[target.denominator], errors="coerce"
        ).to_numpy(dtype=float)
        valid = (
            np.isfinite(numerator)
            & (numerator >= 0)
            & np.isfinite(exposure)
            & (exposure > 0)
            & (numerator <= exposure)
        )
    return numerator, exposure, valid


@dataclass(frozen=True)
class _ContextFit:
    global_value: float
    values: dict[tuple[str, str], float]


def _fit_context(
    frame: pd.DataFrame,
    target: ObservationTarget,
    config: OpponentAdjustmentAuditConfig,
) -> _ContextFit:
    numerator, exposure, valid = _target_arrays(frame, target)
    if not valid.any():
        raise ValueError(f"training data have no observations for {target.name}")
    if target.kind == "rate":
        global_value = float(
            (numerator[valid].sum() + 0.5)
            / (exposure[valid].sum() + 0.5 / target.seed)
        )
        prior = config.context_rate_prior_minutes
    else:
        global_value = float(
            (numerator[valid].sum() + 40.0 * target.seed)
            / (exposure[valid].sum() + 40.0)
        )
        prior = config.context_probability_prior_attempts
    observed = frame.loc[valid, ["division", "era"]].copy()
    observed["_numerator"] = numerator[valid]
    observed["_exposure"] = exposure[valid]
    values: dict[tuple[str, str], float] = {}
    grouped = observed.groupby(["division", "era"], sort=True)[
        ["_numerator", "_exposure"]
    ].sum()
    for (division, era), row in grouped.iterrows():
        values[_context_key(division, era)] = float(
            (float(row["_numerator"]) + prior * global_value)
            / (float(row["_exposure"]) + prior)
        )
    return _ContextFit(global_value=global_value, values=values)


def _context_predictions(frame: pd.DataFrame, fit: _ContextFit) -> np.ndarray:
    return np.fromiter(
        (
            fit.values.get(_context_key(division, era), fit.global_value)
            for division, era in zip(frame["division"], frame["era"])
        ),
        dtype=float,
        count=len(frame),
    )


def _group_effects(
    identities: np.ndarray,
    residuals: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    unique, inverse = np.unique(identities.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    totals = np.bincount(inverse, weights=residuals)
    effects = totals / (counts + ridge)
    effects -= float(np.dot(counts, effects) / counts.sum())
    effects = np.clip(effects, -_EFFECT_LIMIT, _EFFECT_LIMIT)
    return {str(identity): float(effects[index]) for index, identity in enumerate(unique)}


def fit_bout_clustered_two_way_effects(
    actor_ids: Iterable[object],
    opponent_ids: Iterable[object],
    residuals: Iterable[float],
    *,
    ridge: float,
    sample_weights: Iterable[float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit equal-bout actor/opponent ridge effects.

    Each fighter-side bout is one unit regardless of its number of actions.
    This prevents a 250-attempt fight from masquerading as 250 independent
    observations of stable fighter quality.
    """

    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    actors = np.asarray([str(value) for value in actor_ids], dtype=object)
    opponents = np.asarray([str(value) for value in opponent_ids], dtype=object)
    values = np.asarray(list(residuals), dtype=float)
    weights = (
        np.ones(len(values), dtype=float)
        if sample_weights is None
        else np.asarray(list(sample_weights), dtype=float)
    )
    if not len(actors) == len(opponents) == len(values) == len(weights):
        raise ValueError("two-way effect inputs must have equal lengths")
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
        & (actors != "")
        & (opponents != "")
    )
    actors = actors[valid]
    opponents = opponents[valid]
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        return {}, {}
    identities = sorted(set(actors) | set(opponents))
    index = {identity: position for position, identity in enumerate(identities)}
    actor_index = np.fromiter(
        (index[value] for value in actors), dtype=np.int64, count=len(actors)
    )
    opponent_index = np.fromiter(
        (index[value] for value in opponents),
        dtype=np.int64,
        count=len(opponents),
    )
    actor_counts = np.bincount(
        actor_index, weights=weights, minlength=len(identities)
    ).astype(float)
    opponent_counts = np.bincount(
        opponent_index, weights=weights, minlength=len(identities)
    ).astype(float)
    actor_effect = np.zeros(len(identities), dtype=float)
    opponent_effect = np.zeros(len(identities), dtype=float)

    def update(group_index: np.ndarray, targets: np.ndarray, counts: np.ndarray) -> np.ndarray:
        totals = np.bincount(
            group_index, weights=weights * targets, minlength=len(identities)
        )
        effects = totals / (counts + ridge)
        active = counts > 0
        effects[active] -= float(
            np.dot(counts[active], effects[active]) / counts[active].sum()
        )
        return np.clip(effects, -_EFFECT_LIMIT, _EFFECT_LIMIT)

    for _ in range(_EFFECT_ITERATIONS):
        actor_effect = update(
            actor_index, values - opponent_effect[opponent_index], actor_counts
        )
        opponent_effect = update(
            opponent_index, values - actor_effect[actor_index], opponent_counts
        )
    return (
        {
            identity: float(actor_effect[position])
            for identity, position in index.items()
            if actor_counts[position] > 0
        },
        {
            identity: float(opponent_effect[position])
            for identity, position in index.items()
            if opponent_counts[position] > 0
        },
    )


@dataclass(frozen=True)
class _TargetFit:
    context: _ContextFit
    marginal_actor: dict[float, dict[str, float]]
    marginal_opponent: dict[float, dict[str, float]]
    adjusted_actor: dict[float, dict[str, float]]
    adjusted_opponent: dict[float, dict[str, float]]


def _fit_target(
    training: pd.DataFrame,
    target: ObservationTarget,
    config: OpponentAdjustmentAuditConfig,
) -> _TargetFit:
    context = _fit_context(training, target, config)
    numerator, exposure, valid = _target_arrays(training, target)
    baseline = _context_predictions(training, context)
    if target.kind == "rate":
        stabilized = (numerator[valid] + 0.5) / (
            exposure[valid] + 0.5 / baseline[valid]
        )
        residuals = np.log(stabilized / baseline[valid])
    else:
        smoothed = np.clip(
            (numerator[valid] + 0.5) / (exposure[valid] + 1.0),
            _EPSILON,
            1.0 - _EPSILON,
        )
        residuals = np.log(smoothed / (1.0 - smoothed)) - np.log(
            baseline[valid] / (1.0 - baseline[valid])
        )
    actors = training.loc[valid, "fighter_id"].astype(str).to_numpy(object)
    opponents = training.loc[valid, "opponent_id"].astype(str).to_numpy(object)
    marginal_actor: dict[float, dict[str, float]] = {}
    marginal_opponent: dict[float, dict[str, float]] = {}
    adjusted_actor: dict[float, dict[str, float]] = {}
    adjusted_opponent: dict[float, dict[str, float]] = {}
    for ridge in config.ridge_grid:
        marginal_actor[ridge] = _group_effects(actors, residuals, ridge)
        marginal_opponent[ridge] = _group_effects(opponents, residuals, ridge)
        actor, opponent = fit_bout_clustered_two_way_effects(
            actors, opponents, residuals, ridge=ridge
        )
        adjusted_actor[ridge] = actor
        adjusted_opponent[ridge] = opponent
    return _TargetFit(
        context=context,
        marginal_actor=marginal_actor,
        marginal_opponent=marginal_opponent,
        adjusted_actor=adjusted_actor,
        adjusted_opponent=adjusted_opponent,
    )


def _linear_prediction(
    baseline: float,
    actor_id: str,
    opponent_id: str,
    target: ObservationTarget,
    fit: _TargetFit,
    ridge: float,
    model: str,
) -> float:
    if model == "marginal":
        actor_effect = fit.marginal_actor[ridge].get(actor_id, 0.0)
        opponent_effect = fit.marginal_opponent[ridge].get(opponent_id, 0.0)
        # Independent marginal histories each contain matchup signal. Average
        # them so the comparator does not double the same deviation.
        effect = (
            actor_effect
            if target.kind == "rate"
            else 0.5 * (actor_effect + opponent_effect)
        )
    elif model == "opponent_adjusted":
        actor_effect = fit.adjusted_actor[ridge].get(actor_id, 0.0)
        opponent_effect = fit.adjusted_opponent[ridge].get(opponent_id, 0.0)
        # Simulator opportunity rates currently consume actor quality only;
        # opponent rows debias that actor estimate. Accuracy has an explicit
        # opponent-defense consumer and therefore uses both terms.
        effect = (
            actor_effect
            if target.kind == "rate"
            else actor_effect + opponent_effect
        )
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown observation model: {model}")
    if target.kind == "rate":
        return float(np.clip(baseline * math.exp(effect), 1e-6, 100.0))
    return float(np.clip(_expit(_logit(baseline) + effect), _EPSILON, 1.0 - _EPSILON))


def _negative_log_likelihood(
    actual: np.ndarray,
    exposure: np.ndarray,
    prediction: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    if kind == "rate":
        mean = np.clip(prediction * exposure, _EPSILON, None)
        return mean - actual * np.log(mean) + gammaln(actual + 1.0)
    probability = np.clip(prediction, _EPSILON, 1.0 - _EPSILON)
    return -(
        gammaln(exposure + 1.0)
        - gammaln(actual + 1.0)
        - gammaln(exposure - actual + 1.0)
        + actual * np.log(probability)
        + (exposure - actual) * np.log(1.0 - probability)
    )


def _predict_target_grid(
    training: pd.DataFrame,
    test: pd.DataFrame,
    target: ObservationTarget,
    config: OpponentAdjustmentAuditConfig,
) -> pd.DataFrame:
    fit = _fit_target(training, target, config)
    actual, exposure, valid = _target_arrays(test, target)
    rows = test.loc[
        valid,
        ["date", "event_id", "fight_id", "fighter_id", "opponent_id"],
    ].copy()
    rows["actual"] = actual[valid]
    rows["exposure"] = exposure[valid]
    baseline = _context_predictions(test, fit.context)[valid]
    rows["context_prediction"] = baseline
    rows["context_loss"] = _negative_log_likelihood(
        actual[valid], exposure[valid], baseline, kind=target.kind
    )
    actor_ids = rows["fighter_id"].astype(str).to_numpy(object)
    opponent_ids = rows["opponent_id"].astype(str).to_numpy(object)
    for model in ("marginal", "opponent_adjusted"):
        for ridge in config.ridge_grid:
            prediction = np.fromiter(
                (
                    _linear_prediction(
                        float(base),
                        str(actor),
                        str(opponent),
                        target,
                        fit,
                        ridge,
                        model,
                    )
                    for base, actor, opponent in zip(
                        baseline, actor_ids, opponent_ids
                    )
                ),
                dtype=float,
                count=len(rows),
            )
            prefix = f"{model}_{ridge:g}"
            rows[f"{prefix}_prediction"] = prediction
            rows[f"{prefix}_loss"] = _negative_log_likelihood(
                actual[valid], exposure[valid], prediction, kind=target.kind
            )
    rows.insert(0, "target", target.name)
    rows.insert(1, "kind", target.kind)
    return rows.reset_index(drop=True)


def _eligible_event_sides(
    frame: pd.DataFrame,
    event_id: str,
    cutoff: pd.Timestamp,
    *,
    minimum_prior: int,
    fight_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = frame.loc[frame["date"].lt(cutoff)].copy()
    test = frame.loc[
        frame["date"].eq(cutoff) & frame["event_id"].astype(str).eq(event_id)
    ].copy()
    if fight_ids is not None:
        test = test.loc[test["fight_id"].astype(str).isin(fight_ids)].copy()
    history = training.groupby("fighter_id", sort=False)["fight_id"].nunique()
    test["_prior"] = test["fighter_id"].map(history).fillna(0).astype(int)
    eligible_fights = set(
        test.groupby("fight_id", sort=False)["_prior"]
        .min()
        .loc[lambda values: values.ge(minimum_prior)]
        .index.astype(str)
    )
    return training, test.loc[test["fight_id"].astype(str).isin(eligible_fights)].copy()


def _grid_scores(predictions: pd.DataFrame, config: OpponentAdjustmentAuditConfig) -> dict[tuple[str, str, float], tuple[float, int]]:
    scores: dict[tuple[str, str, float], tuple[float, int]] = {}
    for target, rows in predictions.groupby("target", sort=False):
        for model in ("marginal", "opponent_adjusted"):
            for ridge in config.ridge_grid:
                values = rows[f"{model}_{ridge:g}_loss"].to_numpy(dtype=float)
                scores[(str(target), model, ridge)] = (
                    float(values.sum()),
                    int(len(values)),
                )
    return scores


def _select_ridges(
    score_sets: Iterable[Mapping[tuple[str, str, float], tuple[float, int]]],
    config: OpponentAdjustmentAuditConfig,
) -> dict[tuple[str, str], float]:
    score_sets = list(score_sets)
    selected: dict[tuple[str, str], float] = {}
    for target in OBSERVATION_TARGETS:
        for model in ("marginal", "opponent_adjusted"):
            candidates: list[tuple[float, float]] = []
            for ridge in config.ridge_grid:
                total_loss = 0.0
                total_rows = 0
                for scores in score_sets:
                    loss, count = scores.get((target.name, model, ridge), (0.0, 0))
                    total_loss += loss
                    total_rows += count
                mean_loss = math.inf if total_rows == 0 else total_loss / total_rows
                candidates.append((mean_loss, ridge))
            # On exact ties prefer stronger regularization.
            selected[(target.name, model)] = min(
                candidates, key=lambda item: (item[0], -item[1])
            )[1]
    return selected


class ChronologicalOpponentRidgeSelector:
    """Select audit-equivalent ridges using only cards before a cutoff.

    The cache belongs to one causal fitter/run. It makes a sequence of outer
    card selections reuse the same strictly historical inner-card scores,
    while future rows remain invisible because every lookup applies an
    explicit timestamp cutoff.
    """

    def __init__(
        self,
        raw_fights: pd.DataFrame,
        config: OpponentAdjustmentAuditConfig | None = None,
    ) -> None:
        self.config = config or OpponentAdjustmentAuditConfig()
        self.config.validate()
        self.frame = _prepare_frame(raw_fights)
        self._inner_cache: dict[
            str, dict[tuple[str, str, float], tuple[float, int]] | None
        ] = {}
        self._selection_cache: dict[
            str, tuple[dict[tuple[str, str], float], tuple[str, ...]]
        ] = {}

    def selected_for_cutoff(
        self, cutoff: object
    ) -> tuple[dict[tuple[str, str], float], tuple[str, ...]]:
        timestamp = pd.to_datetime(cutoff, errors="raise", utc=True)
        cache_key = timestamp.isoformat()
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return cached
        preceding = (
            self.frame.loc[
                self.frame["date"].lt(timestamp), ["date", "event_id"]
            ]
            .drop_duplicates()
            .sort_values(["date", "event_id"], ascending=False, kind="stable")
        )
        inner_scores: list[
            Mapping[tuple[str, str, float], tuple[float, int]]
        ] = []
        inner_ids: list[str] = []
        for internal in preceding.to_dict("records"):
            if len(inner_scores) >= self.config.inner_validation_events:
                break
            event_id = str(internal["event_id"])
            if event_id not in self._inner_cache:
                training, validation = _eligible_event_sides(
                    self.frame,
                    event_id,
                    pd.Timestamp(internal["date"]),
                    minimum_prior=self.config.min_prior_ufc_fights,
                )
                if (
                    training["fight_id"].nunique()
                    < self.config.minimum_training_fights
                    or validation.empty
                ):
                    self._inner_cache[event_id] = None
                else:
                    grids = [
                        _predict_target_grid(
                            training, validation, target, self.config
                        )
                        for target in OBSERVATION_TARGETS
                    ]
                    self._inner_cache[event_id] = _grid_scores(
                        pd.concat(grids, ignore_index=True), self.config
                    )
            scores = self._inner_cache[event_id]
            if scores is not None:
                inner_scores.append(scores)
                inner_ids.append(event_id)
        if len(inner_scores) < self.config.inner_validation_events:
            raise ValueError(
                f"cutoff {timestamp.isoformat()} has only {len(inner_scores)} "
                "eligible inner cards"
            )
        result = (_select_ridges(inner_scores, self.config), tuple(inner_ids))
        self._selection_cache[cache_key] = result
        return result


def _event_interval(
    frame: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    replicates: int,
    random_seed: int,
    deadline: float,
) -> dict[str, float]:
    deltas = (
        frame[candidate_column].to_numpy(float)
        - frame[baseline_column].to_numpy(float)
    )
    event_ids = frame["event_id"].astype(str).to_numpy(object)
    unique = np.unique(event_ids)
    blocks = [deltas[event_ids == event] for event in unique]
    rng = np.random.Generator(np.random.PCG64DXSM(random_seed))
    sampled = np.empty(replicates, dtype=float)
    for index in range(replicates):
        if index % 50 == 0:
            _check_deadline(deadline)
        choices = rng.integers(0, len(blocks), size=len(blocks))
        numerator = math.fsum(float(blocks[item].sum()) for item in choices)
        denominator = sum(len(blocks[item]) for item in choices)
        sampled[index] = numerator / denominator
    return {
        "mean_loss_difference": float(deltas.mean()),
        "event_block_95_interval_low": float(np.quantile(sampled, 0.025)),
        "event_block_95_interval_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(sampled < 0)),
    }


def _overall_relative_interval(
    predictions: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    replicates: int,
    random_seed: int,
    deadline: float,
) -> dict[str, float]:
    grouped = predictions.groupby(["event_id", "target"], sort=True).agg(
        candidate_sum=(candidate_column, "sum"),
        baseline_sum=(baseline_column, "sum"),
        observations=(candidate_column, "size"),
    )
    event_ids = sorted(predictions["event_id"].astype(str).unique())
    targets = [target.name for target in OBSERVATION_TARGETS]

    def estimate(sampled_events: Iterable[str]) -> float:
        candidate = {target: 0.0 for target in targets}
        baseline = {target: 0.0 for target in targets}
        counts = {target: 0 for target in targets}
        for event in sampled_events:
            for target in targets:
                key = (event, target)
                if key not in grouped.index:
                    continue
                row = grouped.loc[key]
                candidate[target] += float(row["candidate_sum"])
                baseline[target] += float(row["baseline_sum"])
                counts[target] += int(row["observations"])
        relative = []
        for target in targets:
            if counts[target] == 0 or baseline[target] <= 0:
                continue
            candidate_mean = candidate[target] / counts[target]
            baseline_mean = baseline[target] / counts[target]
            relative.append(candidate_mean / baseline_mean - 1.0)
        return float(np.mean(relative))

    point = estimate(event_ids)
    rng = np.random.Generator(np.random.PCG64DXSM(random_seed))
    sampled = np.empty(replicates, dtype=float)
    for index in range(replicates):
        if index % 50 == 0:
            _check_deadline(deadline)
        choices = rng.integers(0, len(event_ids), size=len(event_ids))
        sampled[index] = estimate(event_ids[item] for item in choices)
    return {
        "equal_target_mean_relative_loss_difference": point,
        "event_block_95_interval_low": float(np.quantile(sampled, 0.025)),
        "event_block_95_interval_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(sampled < 0)),
    }


def run_opponent_adjustment_audit(
    raw_fights: pd.DataFrame,
    outer_fights: pd.DataFrame,
    *,
    config: OpponentAdjustmentAuditConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run nested chronological selection and score frozen outer cards."""

    settings = config or OpponentAdjustmentAuditConfig()
    settings.validate()
    started = time.monotonic()
    deadline = started + settings.max_runtime_seconds
    frame = _prepare_frame(raw_fights)
    required_outer = {"date", "event_id", "fight_id"}
    missing = sorted(required_outer - set(outer_fights.columns))
    if missing:
        raise ValueError(f"outer_fights are missing columns: {missing}")
    outer = outer_fights.copy()
    outer["date"] = pd.to_datetime(outer["date"], errors="raise", utc=True)
    outer_events = outer[["date", "event_id"]].drop_duplicates().sort_values(
        ["date", "event_id"], kind="stable"
    )
    if outer_events.empty:
        raise ValueError("outer_fights contain no events")
    event_order = (
        frame.loc[frame["date"].lt(outer_events["date"].max()), ["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
    )
    inner_cache: dict[str, dict[tuple[str, str, float], tuple[float, int]] | None] = {}
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for position, outer_event in enumerate(outer_events.to_dict("records"), start=1):
        _check_deadline(deadline)
        cutoff = pd.Timestamp(outer_event["date"])
        event_id = str(outer_event["event_id"])
        preceding = event_order.loc[event_order["date"].lt(cutoff)].sort_values(
            ["date", "event_id"], ascending=False, kind="stable"
        )
        inner_scores: list[Mapping[tuple[str, str, float], tuple[float, int]]] = []
        inner_ids: list[str] = []
        for internal in preceding.to_dict("records"):
            if len(inner_scores) >= settings.inner_validation_events:
                break
            internal_id = str(internal["event_id"])
            if internal_id not in inner_cache:
                training, validation = _eligible_event_sides(
                    frame,
                    internal_id,
                    pd.Timestamp(internal["date"]),
                    minimum_prior=settings.min_prior_ufc_fights,
                )
                if (
                    training["fight_id"].nunique()
                    < settings.minimum_training_fights
                    or validation.empty
                ):
                    inner_cache[internal_id] = None
                else:
                    grids = [
                        _predict_target_grid(training, validation, target, settings)
                        for target in OBSERVATION_TARGETS
                    ]
                    inner_cache[internal_id] = _grid_scores(
                        pd.concat(grids, ignore_index=True), settings
                    )
            cached = inner_cache[internal_id]
            if cached is not None:
                inner_scores.append(cached)
                inner_ids.append(internal_id)
        if len(inner_scores) < settings.inner_validation_events:
            raise ValueError(
                f"event {event_id} has only {len(inner_scores)} eligible inner cards"
            )
        selected = _select_ridges(inner_scores, settings)
        fight_ids = set(
            outer.loc[
                outer["event_id"].astype(str).eq(event_id), "fight_id"
            ].astype(str)
        )
        training, test = _eligible_event_sides(
            frame,
            event_id,
            cutoff,
            minimum_prior=settings.min_prior_ufc_fights,
            fight_ids=fight_ids,
        )
        if set(test["fight_id"].astype(str)) != fight_ids:
            raise ValueError(f"outer event {event_id} lost frozen eligible fights")
        for target in OBSERVATION_TARGETS:
            grid = _predict_target_grid(training, test, target, settings)
            marginal_ridge = selected[(target.name, "marginal")]
            adjusted_ridge = selected[(target.name, "opponent_adjusted")]
            grid["marginal_selected_ridge"] = marginal_ridge
            grid["opponent_adjusted_selected_ridge"] = adjusted_ridge
            grid["marginal_prediction"] = grid[
                f"marginal_{marginal_ridge:g}_prediction"
            ]
            grid["marginal_loss"] = grid[f"marginal_{marginal_ridge:g}_loss"]
            grid["opponent_adjusted_prediction"] = grid[
                f"opponent_adjusted_{adjusted_ridge:g}_prediction"
            ]
            grid["opponent_adjusted_loss"] = grid[
                f"opponent_adjusted_{adjusted_ridge:g}_loss"
            ]
            retained = [
                "target",
                "kind",
                "date",
                "event_id",
                "fight_id",
                "fighter_id",
                "opponent_id",
                "actual",
                "exposure",
                "context_prediction",
                "marginal_prediction",
                "opponent_adjusted_prediction",
                "context_loss",
                "marginal_loss",
                "opponent_adjusted_loss",
                "marginal_selected_ridge",
                "opponent_adjusted_selected_ridge",
            ]
            prediction_frames.append(grid[retained])
            selection_rows.append(
                {
                    "event_id": event_id,
                    "target": target.name,
                    "marginal_ridge": marginal_ridge,
                    "opponent_adjusted_ridge": adjusted_ridge,
                    "inner_event_ids": inner_ids,
                }
            )
        if progress:
            progress(
                f"Scored outer card {position}/{len(outer_events)}: {event_id} "
                f"({len(fight_ids)} fights, {len(inner_ids)} inner cards)."
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    target_results: dict[str, object] = {}
    for target_index, target in enumerate(OBSERVATION_TARGETS):
        rows = predictions.loc[predictions["target"].eq(target.name)].copy()
        context_mean = float(rows["context_loss"].mean())
        marginal_mean = float(rows["marginal_loss"].mean())
        adjusted_mean = float(rows["opponent_adjusted_loss"].mean())
        target_results[target.name] = {
            "kind": target.kind,
            "observations": int(len(rows)),
            "context_mean_negative_log_likelihood": context_mean,
            "marginal_mean_negative_log_likelihood": marginal_mean,
            "opponent_adjusted_mean_negative_log_likelihood": adjusted_mean,
            "marginal_over_context_loss_ratio": marginal_mean / context_mean,
            "opponent_adjusted_over_marginal_loss_ratio": (
                adjusted_mean / marginal_mean
            ),
            "marginal_vs_context": _event_interval(
                rows,
                "marginal_loss",
                "context_loss",
                replicates=settings.bootstrap_replicates,
                random_seed=settings.random_seed + target_index * 101 + 1,
                deadline=deadline,
            ),
            "opponent_adjusted_vs_marginal": _event_interval(
                rows,
                "opponent_adjusted_loss",
                "marginal_loss",
                replicates=settings.bootstrap_replicates,
                random_seed=settings.random_seed + target_index * 101 + 2,
                deadline=deadline,
            ),
        }
    overall_marginal = _overall_relative_interval(
        predictions,
        "marginal_loss",
        "context_loss",
        replicates=settings.bootstrap_replicates,
        random_seed=settings.random_seed + 9001,
        deadline=deadline,
    )
    overall_adjusted = _overall_relative_interval(
        predictions,
        "opponent_adjusted_loss",
        "marginal_loss",
        replicates=settings.bootstrap_replicates,
        random_seed=settings.random_seed + 9002,
        deadline=deadline,
    )
    improved_targets = sum(
        float(dict(value)["opponent_adjusted_over_marginal_loss_ratio"]) < 1.0
        for value in target_results.values()
    )
    no_material_harm = all(
        float(dict(value)["opponent_adjusted_over_marginal_loss_ratio"]) <= 1.01
        for value in target_results.values()
    )
    advances = bool(
        overall_adjusted["event_block_95_interval_high"] < 0.0
        and improved_targets >= 3
        and no_material_harm
    )
    elapsed = time.monotonic() - started
    report: dict[str, object] = {
        "schema_version": OPPONENT_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "candidate_only": True,
        "production_behavior_changed": False,
        "simulation_executed": False,
        "config": asdict(settings),
        "split": {
            "outer_event_cards": int(len(outer_events)),
            "outer_fights": int(outer["fight_id"].nunique()),
            "outer_fight_ids_sha256": canonical_sha256(
                sorted(outer["fight_id"].astype(str).unique())
            ),
            "inner_selection_rule": (
                "target-specific minimum next-card negative log likelihood over "
                f"the {settings.inner_validation_events} strictly preceding "
                "eligible cards; exact ties choose stronger ridge"
            ),
        },
        "evidence_units": {
            "fighter_effect_fit": "one equal-weight observation per fighter-side bout",
            "outer_uncertainty": "physical event-card block bootstrap",
            "conditional_observation_scores": "Poisson count and binomial count likelihoods",
        },
        "targets": target_results,
        "overall": {
            "marginal_vs_context": overall_marginal,
            "opponent_adjusted_vs_marginal": overall_adjusted,
            "opponent_adjusted_point_improved_targets": improved_targets,
            "no_target_loss_worse_by_more_than_one_percent": no_material_harm,
        },
        "candidate_advances_to_simulation_screen": advances,
        "decision": (
            "eligible for one bounded simulator development screen"
            if advances
            else "do not run another simulator screen"
        ),
        "selection_history": selection_rows,
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "max_runtime_seconds": settings.max_runtime_seconds,
            "cached_inner_event_cards": len(inner_cache),
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report, predictions


def _atomic_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _atomic_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def execute_opponent_adjustment_audit(
    *,
    raw_path: str | Path,
    profiles_path: str | Path,
    round_path: str | Path,
    cohort_manifest_path: str | Path,
    cohort_name: str,
    output: str | Path,
    predictions_output: str | Path,
    config: OpponentAdjustmentAuditConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    """Validate a frozen cohort and execute the local observation audit."""

    from .research import (
        _file_sha256,
        _frozen_cohort_selection,
        physical_backtest_frame,
    )

    settings = config or OpponentAdjustmentAuditConfig()
    settings.validate()
    raw = pd.read_csv(raw_path, low_memory=False)
    sources = {
        "raw": _file_sha256(raw_path, required=True),
        "profiles": _file_sha256(profiles_path, required=False),
        "round_stats": _file_sha256(round_path, required=False),
    }
    outer, _events, _counts, cohort_metadata = _frozen_cohort_selection(
        physical_backtest_frame(raw),
        manifest_path=cohort_manifest_path,
        cohort_name=cohort_name,
        min_prior_ufc_fights=settings.min_prior_ufc_fights,
        source_sha256=sources,
    )
    report, predictions = run_opponent_adjustment_audit(
        raw, outer, config=settings, progress=progress
    )
    report["source_sha256"] = sources
    report["cohort"] = cohort_metadata
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    report_path = _atomic_json(output, report)
    predictions_path = _atomic_csv(predictions_output, predictions)
    return report, report_path, predictions_path

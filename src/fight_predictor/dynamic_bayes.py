"""Dynamic fully Bayesian UFC winner model.

Each fighter has a latent ability at every historical appearance. Ability
follows a Gaussian random walk whose variance grows with elapsed time. The
drift variance, initial population variance, matchup coefficients, latent fight
performances, and all historical ability states are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .hierarchical_bayes import (
    HierarchicalBayesPrediction,
    coefficient_prior_scales,
    _numeric_matrix,
    _rms_scale,
    _sample_truncated_latent,
    _validate_frame,
)


@dataclass(frozen=True)
class DynamicBayesConfig:
    burn_in: int = 120
    posterior_draws: int = 120
    thin: int = 1
    chains: int = 2
    coefficient_prior_scale: float = 0.35
    grouped_coefficient_priors: bool = False
    initial_variance_shape: float = 2.0
    initial_variance_scale: float = 0.25
    drift_variance_shape: float = 2.0
    drift_variance_scale: float = 0.04
    minimum_transition_years: float = 1.0 / 365.25
    seed: int = 20260830

    def __post_init__(self) -> None:
        if self.burn_in < 1 or self.posterior_draws < 1:
            raise ValueError("dynamic Bayesian burn-in and draws must be positive")
        if self.thin < 1 or self.chains < 1:
            raise ValueError("dynamic Bayesian thinning and chains must be positive")
        numeric = (
            self.coefficient_prior_scale,
            self.initial_variance_shape,
            self.initial_variance_scale,
            self.drift_variance_shape,
            self.drift_variance_scale,
            self.minimum_transition_years,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("dynamic Bayesian priors must be finite and positive")


def without_elo_features(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Remove ratings replaced by the latent fighter-state process."""

    excluded = {
        "rating_uncertainty_diff",
        "average_opponent_elo_diff",
    }
    return tuple(
        feature
        for feature in feature_columns
        if feature not in excluded
        and not feature.startswith("elo_")
        and not feature.startswith("division_elo_")
    )


@dataclass(frozen=True)
class _StateGraph:
    fight_fighter_state: np.ndarray
    fight_opponent_state: np.ndarray
    previous_state: np.ndarray
    next_state: np.ndarray
    transition_years: np.ndarray
    state_fighter_ids: tuple[str, ...]
    state_dates: tuple[pd.Timestamp, ...]
    initial_states: np.ndarray
    transition_states: np.ndarray
    last_state_by_fighter: dict[str, int]


def _state_graph(
    training: pd.DataFrame, minimum_years: float
) -> tuple[pd.DataFrame, _StateGraph]:
    required = {"date", "event_id", "fight_id", "fighter_id", "opponent_id"}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"dynamic Bayesian training is missing: {sorted(missing)}")
    sort_columns = ["date", "event_id"]
    if "bout_order" in training:
        sort_columns.append("bout_order")
    sort_columns.append("fight_id")
    ordered = training.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    dates = pd.to_datetime(ordered["date"], errors="raise")
    fighter_state = np.empty(len(ordered), dtype=int)
    opponent_state = np.empty(len(ordered), dtype=int)
    state_fighters: list[str] = []
    state_dates: list[pd.Timestamp] = []
    previous: list[int] = []
    transition_years: list[float] = []
    last_by_fighter: dict[str, int] = {}

    for row_index, row in ordered.iterrows():
        date = pd.Timestamp(dates.iloc[row_index])
        for side, destination in (
            (str(row["fighter_id"]), fighter_state),
            (str(row["opponent_id"]), opponent_state),
        ):
            prior = last_by_fighter.get(side, -1)
            node = len(state_fighters)
            destination[row_index] = node
            state_fighters.append(side)
            state_dates.append(date)
            previous.append(prior)
            if prior < 0:
                transition_years.append(0.0)
            else:
                elapsed = max(
                    (date - state_dates[prior]).days / 365.25,
                    minimum_years,
                )
                transition_years.append(float(elapsed))
            last_by_fighter[side] = node

    previous_array = np.asarray(previous, dtype=int)
    next_array = np.full(len(previous_array), -1, dtype=int)
    for node, prior in enumerate(previous_array):
        if prior >= 0:
            next_array[int(prior)] = node
    initial = np.flatnonzero(previous_array < 0)
    transitions = np.flatnonzero(previous_array >= 0)
    return ordered, _StateGraph(
        fight_fighter_state=fighter_state,
        fight_opponent_state=opponent_state,
        previous_state=previous_array,
        next_state=next_array,
        transition_years=np.asarray(transition_years, dtype=float),
        state_fighter_ids=tuple(state_fighters),
        state_dates=tuple(state_dates),
        initial_states=initial,
        transition_states=transitions,
        last_state_by_fighter=last_by_fighter,
    )


def _prediction_state_contract(
    prediction: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, pd.Timestamp], ...]]:
    if "date" not in prediction:
        raise ValueError("dynamic Bayesian predictions require an event date")
    dates = pd.to_datetime(prediction["date"], errors="raise")
    keys = sorted(
        {
            (str(fighter_id), pd.Timestamp(date))
            for fighter_id, date in zip(prediction["fighter_id"], dates)
        }
        | {
            (str(fighter_id), pd.Timestamp(date))
            for fighter_id, date in zip(prediction["opponent_id"], dates)
        },
        key=lambda value: (value[1], value[0]),
    )
    lookup = {key: index for index, key in enumerate(keys)}
    fighter = np.asarray(
        [
            lookup[(str(fighter_id), pd.Timestamp(date))]
            for fighter_id, date in zip(prediction["fighter_id"], dates)
        ],
        dtype=int,
    )
    opponent = np.asarray(
        [
            lookup[(str(fighter_id), pd.Timestamp(date))]
            for fighter_id, date in zip(prediction["opponent_id"], dates)
        ],
        dtype=int,
    )
    return fighter, opponent, tuple(keys)


def _one_dynamic_chain(
    *,
    training_x: np.ndarray,
    prediction_x: np.ndarray,
    target: np.ndarray,
    graph: _StateGraph,
    prediction_fighter: np.ndarray,
    prediction_opponent: np.ndarray,
    prediction_keys: tuple[tuple[str, pd.Timestamp], ...],
    prior_scales: np.ndarray,
    config: DynamicBayesConfig,
    seed: int,
) -> tuple[np.ndarray, list[float], list[float]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    coefficient_count = training_x.shape[1]
    precision = training_x.T @ training_x
    precision.flat[:: coefficient_count + 1] += 1.0 / np.square(prior_scales)
    try:
        coefficient_cholesky = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("dynamic Bayesian coefficient precision is invalid") from error
    coefficients = np.zeros(coefficient_count, dtype=float)
    states = np.zeros(len(graph.previous_state), dtype=float)
    initial_variance = config.initial_variance_scale / max(
        config.initial_variance_shape - 1.0, 0.5
    )
    drift_variance = config.drift_variance_scale / max(
        config.drift_variance_shape - 1.0, 0.5
    )
    total_iterations = config.burn_in + config.posterior_draws * config.thin
    predictions: list[np.ndarray] = []
    initial_scales: list[float] = []
    annual_drift_scales: list[float] = []

    for iteration in range(total_iterations):
        linear = (
            training_x @ coefficients
            + states[graph.fight_fighter_state]
            - states[graph.fight_opponent_state]
        )
        latent = _sample_truncated_latent(rng, linear, target)
        residual = (
            latent
            - states[graph.fight_fighter_state]
            + states[graph.fight_opponent_state]
        )
        rhs = training_x.T @ residual
        coefficient_mean = np.linalg.solve(
            coefficient_cholesky.T,
            np.linalg.solve(coefficient_cholesky, rhs),
        )
        coefficients = coefficient_mean + np.linalg.solve(
            coefficient_cholesky.T, rng.normal(size=coefficient_count)
        )
        feature_location = training_x @ coefficients

        for fight_index in range(len(training_x)):
            for node, other, sign in (
                (
                    int(graph.fight_fighter_state[fight_index]),
                    int(graph.fight_opponent_state[fight_index]),
                    1,
                ),
                (
                    int(graph.fight_opponent_state[fight_index]),
                    int(graph.fight_fighter_state[fight_index]),
                    -1,
                ),
            ):
                if sign == 1:
                    observation = (
                        latent[fight_index]
                        - feature_location[fight_index]
                        + states[other]
                    )
                else:
                    observation = (
                        states[other]
                        - latent[fight_index]
                        + feature_location[fight_index]
                    )
                conditional_precision = 1.0
                weighted_mean = float(observation)
                prior = int(graph.previous_state[node])
                if prior < 0:
                    prior_precision = 1.0 / initial_variance
                    conditional_precision += prior_precision
                else:
                    prior_precision = 1.0 / (
                        drift_variance * graph.transition_years[node]
                    )
                    conditional_precision += prior_precision
                    weighted_mean += prior_precision * states[prior]
                following = int(graph.next_state[node])
                if following >= 0:
                    next_precision = 1.0 / (
                        drift_variance * graph.transition_years[following]
                    )
                    conditional_precision += next_precision
                    weighted_mean += next_precision * states[following]
                states[node] = rng.normal(
                    weighted_mean / conditional_precision,
                    1.0 / math.sqrt(conditional_precision),
                )

        initial_shape = config.initial_variance_shape + len(graph.initial_states) / 2.0
        initial_scale = config.initial_variance_scale + float(
            states[graph.initial_states] @ states[graph.initial_states]
        ) / 2.0
        initial_variance = 1.0 / rng.gamma(
            shape=initial_shape, scale=1.0 / initial_scale
        )
        increments = (
            states[graph.transition_states]
            - states[graph.previous_state[graph.transition_states]]
        )
        drift_shape = config.drift_variance_shape + len(increments) / 2.0
        drift_scale = config.drift_variance_scale + float(
            np.sum(
                np.square(increments)
                / graph.transition_years[graph.transition_states]
            )
        ) / 2.0
        drift_variance = 1.0 / rng.gamma(
            shape=drift_shape, scale=1.0 / drift_scale
        )

        if iteration < config.burn_in or (iteration - config.burn_in) % config.thin:
            continue
        predictive_mean = np.empty(len(prediction_keys), dtype=float)
        predictive_variance = np.empty(len(prediction_keys), dtype=float)
        for index, (fighter_id, date) in enumerate(prediction_keys):
            last = graph.last_state_by_fighter.get(fighter_id)
            if last is None:
                predictive_mean[index] = 0.0
                predictive_variance[index] = initial_variance
            else:
                if date < graph.state_dates[last]:
                    raise ValueError(
                        "prediction date precedes a fighter's training state"
                    )
                elapsed = max(
                    (date - graph.state_dates[last]).days / 365.25,
                    config.minimum_transition_years,
                )
                predictive_mean[index] = states[last]
                predictive_variance[index] = drift_variance * elapsed
        matchup_mean = (
            prediction_x @ coefficients
            + predictive_mean[prediction_fighter]
            - predictive_mean[prediction_opponent]
        )
        matchup_variance = (
            predictive_variance[prediction_fighter]
            + predictive_variance[prediction_opponent]
        )
        predictions.append(ndtr(matchup_mean / np.sqrt(1.0 + matchup_variance)))
        initial_scales.append(math.sqrt(initial_variance))
        annual_drift_scales.append(math.sqrt(drift_variance))

    draws = np.asarray(predictions, dtype=float)
    if draws.shape != (config.posterior_draws, len(prediction_x)):
        raise RuntimeError("dynamic Bayesian sampler retained the wrong draw count")
    return draws, initial_scales, annual_drift_scales


def dynamic_bayes_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: DynamicBayesConfig | None = None,
) -> HierarchicalBayesPrediction:
    """Fit time-evolving fighter states and return posterior win probabilities."""

    selected = config or DynamicBayesConfig()
    features = tuple(feature_columns)
    if not features:
        raise ValueError("dynamic Bayesian model needs at least one feature")
    _validate_frame(training, features, labels=True)
    _validate_frame(prediction, features, labels=False)
    ordered, graph = _state_graph(training, selected.minimum_transition_years)
    raw_training = _numeric_matrix(ordered, features)
    raw_prediction = _numeric_matrix(prediction, features)
    scale = _rms_scale(raw_training)
    training_x = raw_training / scale
    prediction_x = raw_prediction / scale
    target = ordered["target"].to_numpy(dtype=int)
    prediction_fighter, prediction_opponent, prediction_keys = (
        _prediction_state_contract(prediction)
    )
    prior_scales = coefficient_prior_scales(
        features,
        base_scale=selected.coefficient_prior_scale,
        grouped=selected.grouped_coefficient_priors,
    )
    chain_draws: list[np.ndarray] = []
    chain_means: list[np.ndarray] = []
    initial_scales: list[float] = []
    drift_scales: list[float] = []
    for chain in range(selected.chains):
        draws, chain_initial, chain_drift = _one_dynamic_chain(
            training_x=training_x,
            prediction_x=prediction_x,
            target=target,
            graph=graph,
            prediction_fighter=prediction_fighter,
            prediction_opponent=prediction_opponent,
            prediction_keys=prediction_keys,
            prior_scales=prior_scales,
            config=selected,
            seed=selected.seed + chain * 1_000_003,
        )
        chain_draws.append(draws)
        chain_means.append(np.mean(draws, axis=0))
        initial_scales.extend(chain_initial)
        drift_scales.extend(chain_drift)
    all_draws = np.concatenate(chain_draws, axis=0)
    chain_differences = (
        np.ptp(np.asarray(chain_means), axis=0)
        if selected.chains > 1
        else np.asarray([], dtype=float)
    )
    probability = np.mean(all_draws, axis=0)
    return HierarchicalBayesPrediction(
        probability=probability,
        lower_probability=np.quantile(all_draws, 0.05, axis=0),
        upper_probability=np.quantile(all_draws, 0.95, axis=0),
        diagnostics={
            "model": "dynamic_hierarchical_probit",
            "training_fights": len(ordered),
            "fighter_appearance_states": len(graph.previous_state),
            "fighters_in_posterior": len(graph.initial_states),
            "features": len(features),
            "grouped_coefficient_priors": selected.grouped_coefficient_priors,
            "chains": selected.chains,
            "burn_in_per_chain": selected.burn_in,
            "posterior_draws_per_chain": selected.posterior_draws,
            "total_retained_draws": len(all_draws),
            "mean_initial_fighter_scale": float(np.mean(initial_scales)),
            "mean_annual_ability_drift_scale": float(np.mean(drift_scales)),
            "mean_absolute_chain_mean_probability_difference": (
                float(np.mean(chain_differences)) if len(chain_differences) else None
            ),
            "p95_absolute_chain_mean_probability_difference": (
                float(np.quantile(chain_differences, 0.95))
                if len(chain_differences)
                else None
            ),
            "maximum_absolute_chain_mean_probability_difference": (
                float(np.max(chain_differences)) if len(chain_differences) else None
            ),
            "future_state_uncertainty": (
                "known fighter variance grows with time since the last training "
                "fight; unseen fighters integrate over the initial population prior"
            ),
        },
    )

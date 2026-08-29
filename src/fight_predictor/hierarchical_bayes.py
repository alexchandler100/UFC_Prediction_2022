"""Fully Bayesian hierarchical probit model for UFC fight winners.

The model combines the existing point-in-time matchup differences with a
partially pooled latent ability for every fighter::

    P(fighter wins) = Phi(x @ beta + ability[fighter] - ability[opponent])

Coefficient, fighter-ability, and population-variance priors are all sampled.
Albert-Chib latent-normal augmentation makes every conditional update exact;
the implementation therefore needs only NumPy and SciPy rather than a large
probabilistic-programming dependency.  It is research-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


@dataclass(frozen=True)
class HierarchicalBayesConfig:
    burn_in: int = 120
    posterior_draws: int = 120
    thin: int = 1
    chains: int = 2
    coefficient_prior_scale: float = 0.35
    ability_variance_shape: float = 2.0
    ability_variance_scale: float = 0.25
    seed: int = 20260829

    def __post_init__(self) -> None:
        if self.burn_in < 1 or self.posterior_draws < 1:
            raise ValueError("Bayesian burn-in and posterior draws must be positive")
        if self.thin < 1 or self.chains < 1:
            raise ValueError("Bayesian thinning and chains must be positive")
        numeric = (
            self.coefficient_prior_scale,
            self.ability_variance_shape,
            self.ability_variance_scale,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("Bayesian prior parameters must be finite and positive")


@dataclass(frozen=True)
class HierarchicalBayesPrediction:
    probability: np.ndarray
    lower_probability: np.ndarray
    upper_probability: np.ndarray
    diagnostics: dict[str, object]


def _rms_scale(values: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(values), axis=0))
    return np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)


def _validate_frame(frame: pd.DataFrame, features: Sequence[str], *, labels: bool) -> None:
    required = {"fighter_id", "opponent_id", *features}
    if labels:
        required.add("target")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bayesian fight frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Bayesian fight frame cannot be empty")
    if labels and set(frame["target"].astype(int)) - {0, 1}:
        raise ValueError("Bayesian fight targets must be binary")
    if (
        frame["fighter_id"].astype(str).eq("").any()
        or frame["opponent_id"].astype(str).eq("").any()
    ):
        raise ValueError("Bayesian fight identities cannot be empty")


def _numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    values = (
        frame[list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError("Bayesian fight features must be finite after imputation")
    return values


def _fighter_indices(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    identities = sorted(
        set(training["fighter_id"].astype(str))
        | set(training["opponent_id"].astype(str))
    )
    lookup = {fighter_id: index for index, fighter_id in enumerate(identities)}
    prediction_identities = (
        set(prediction["fighter_id"].astype(str))
        | set(prediction["opponent_id"].astype(str))
    )
    unseen = sorted(prediction_identities - set(identities))
    prediction_lookup = {
        **lookup,
        **{
            fighter_id: len(identities) + index
            for index, fighter_id in enumerate(unseen)
        },
    }
    train_fighter = training["fighter_id"].astype(str).map(lookup).to_numpy(dtype=int)
    train_opponent = training["opponent_id"].astype(str).map(lookup).to_numpy(dtype=int)
    predict_fighter = prediction["fighter_id"].astype(str).map(
        prediction_lookup
    ).to_numpy(dtype=int)
    predict_opponent = prediction["opponent_id"].astype(str).map(
        prediction_lookup
    ).to_numpy(dtype=int)
    return (
        train_fighter,
        train_opponent,
        predict_fighter,
        predict_opponent,
        identities,
        unseen,
    )


def _incidence_lists(
    fighter_index: np.ndarray,
    opponent_index: np.ndarray,
    fighter_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    as_fighter: list[list[int]] = [[] for _ in range(fighter_count)]
    as_opponent: list[list[int]] = [[] for _ in range(fighter_count)]
    for row, (fighter, opponent) in enumerate(zip(fighter_index, opponent_index)):
        as_fighter[int(fighter)].append(row)
        as_opponent[int(opponent)].append(row)
    return (
        [np.asarray(rows, dtype=int) for rows in as_fighter],
        [np.asarray(rows, dtype=int) for rows in as_opponent],
    )


def _sample_truncated_latent(
    rng: np.random.Generator,
    location: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    boundary = ndtr(-location)
    uniform = rng.random(len(location))
    probability = np.where(
        target == 1,
        boundary + (1.0 - boundary) * uniform,
        boundary * uniform,
    )
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    latent = location + ndtri(probability)
    if not np.isfinite(latent).all():
        raise RuntimeError("Bayesian latent-variable draw became non-finite")
    return latent


def _one_chain(
    *,
    training_x: np.ndarray,
    prediction_x: np.ndarray,
    target: np.ndarray,
    train_fighter: np.ndarray,
    train_opponent: np.ndarray,
    predict_fighter: np.ndarray,
    predict_opponent: np.ndarray,
    fighter_count: int,
    unseen_fighter_count: int,
    config: HierarchicalBayesConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    coefficient_count = training_x.shape[1]
    prior_precision = 1.0 / config.coefficient_prior_scale**2
    precision = training_x.T @ training_x
    precision.flat[:: coefficient_count + 1] += prior_precision
    try:
        precision_cholesky = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("Bayesian coefficient precision is not positive definite") from error

    as_fighter, as_opponent = _incidence_lists(
        train_fighter, train_opponent, fighter_count
    )
    coefficients = np.zeros(coefficient_count, dtype=float)
    abilities = np.zeros(fighter_count, dtype=float)
    ability_variance = config.ability_variance_scale / max(
        config.ability_variance_shape - 1.0, 0.5
    )
    total_iterations = config.burn_in + config.posterior_draws * config.thin
    predictions: list[np.ndarray] = []
    ability_scales: list[float] = []

    for iteration in range(total_iterations):
        linear = (
            training_x @ coefficients
            + abilities[train_fighter]
            - abilities[train_opponent]
        )
        latent = _sample_truncated_latent(rng, linear, target)

        residual = latent - abilities[train_fighter] + abilities[train_opponent]
        mean_rhs = training_x.T @ residual
        mean = np.linalg.solve(
            precision_cholesky.T,
            np.linalg.solve(precision_cholesky, mean_rhs),
        )
        noise = np.linalg.solve(
            precision_cholesky.T, rng.normal(size=coefficient_count)
        )
        coefficients = mean + noise

        feature_location = training_x @ coefficients
        ability_precision = 1.0 / ability_variance
        for fighter in range(fighter_count):
            fighter_rows = as_fighter[fighter]
            opponent_rows = as_opponent[fighter]
            observation_sum = 0.0
            if len(fighter_rows):
                observation_sum += float(
                    np.sum(
                        latent[fighter_rows]
                        - feature_location[fighter_rows]
                        + abilities[train_opponent[fighter_rows]]
                    )
                )
            if len(opponent_rows):
                observation_sum += float(
                    np.sum(
                        abilities[train_fighter[opponent_rows]]
                        - latent[opponent_rows]
                        + feature_location[opponent_rows]
                    )
                )
            conditional_precision = (
                len(fighter_rows) + len(opponent_rows) + ability_precision
            )
            conditional_mean = observation_sum / conditional_precision
            abilities[fighter] = rng.normal(
                conditional_mean, 1.0 / math.sqrt(conditional_precision)
            )

        posterior_shape = config.ability_variance_shape + fighter_count / 2.0
        posterior_scale = (
            config.ability_variance_scale + float(abilities @ abilities) / 2.0
        )
        ability_variance = 1.0 / rng.gamma(
            shape=posterior_shape, scale=1.0 / posterior_scale
        )

        if iteration < config.burn_in or (iteration - config.burn_in) % config.thin:
            continue
        predictive_abilities = np.concatenate(
            [
                abilities,
                rng.normal(
                    0.0,
                    math.sqrt(ability_variance),
                    size=unseen_fighter_count,
                ),
            ]
        )
        prediction_ability = (
            predictive_abilities[predict_fighter]
            - predictive_abilities[predict_opponent]
        )
        predictions.append(ndtr(prediction_x @ coefficients + prediction_ability))
        ability_scales.append(math.sqrt(ability_variance))

    draws = np.asarray(predictions, dtype=float)
    if draws.shape != (config.posterior_draws, len(prediction_x)):
        raise RuntimeError("Bayesian sampler retained an unexpected number of draws")
    return draws, np.mean(draws, axis=0), ability_scales


def hierarchical_bayes_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: HierarchicalBayesConfig | None = None,
) -> HierarchicalBayesPrediction:
    """Fit the posterior using training rows and predict unlabeled matchups."""

    selected = config or HierarchicalBayesConfig()
    features = tuple(feature_columns)
    if not features:
        raise ValueError("Bayesian fight model needs at least one feature")
    _validate_frame(training, features, labels=True)
    _validate_frame(prediction, features, labels=False)
    raw_training = _numeric_matrix(training, features)
    raw_prediction = _numeric_matrix(prediction, features)
    scale = _rms_scale(raw_training)
    training_x = raw_training / scale
    prediction_x = raw_prediction / scale
    (
        train_fighter,
        train_opponent,
        predict_fighter,
        predict_opponent,
        identities,
        unseen_identities,
    ) = _fighter_indices(training, prediction)
    target = training["target"].to_numpy(dtype=int)

    chain_draws: list[np.ndarray] = []
    chain_means: list[np.ndarray] = []
    ability_scales: list[float] = []
    for chain in range(selected.chains):
        draws, means, scales = _one_chain(
            training_x=training_x,
            prediction_x=prediction_x,
            target=target,
            train_fighter=train_fighter,
            train_opponent=train_opponent,
            predict_fighter=predict_fighter,
            predict_opponent=predict_opponent,
            fighter_count=len(identities),
            unseen_fighter_count=len(unseen_identities),
            config=selected,
            seed=selected.seed + chain * 1_000_003,
        )
        chain_draws.append(draws)
        chain_means.append(means)
        ability_scales.extend(scales)
    all_draws = np.concatenate(chain_draws, axis=0)
    probability = np.mean(all_draws, axis=0)
    lower = np.quantile(all_draws, 0.05, axis=0)
    upper = np.quantile(all_draws, 0.95, axis=0)
    chain_matrix = np.asarray(chain_means)
    chain_differences = (
        np.ptp(chain_matrix, axis=0)
        if selected.chains > 1
        else np.asarray([], dtype=float)
    )
    if not (
        np.isfinite(probability).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise RuntimeError("Bayesian posterior prediction is non-finite")
    return HierarchicalBayesPrediction(
        probability=probability,
        lower_probability=lower,
        upper_probability=upper,
        diagnostics={
            "model": "hierarchical_probit_with_fighter_random_effects",
            "training_fights": len(training),
            "fighters_in_posterior": len(identities),
            "unseen_prediction_fighters_integrated_over_population_prior": len(
                unseen_identities
            ),
            "features": len(features),
            "chains": selected.chains,
            "burn_in_per_chain": selected.burn_in,
            "posterior_draws_per_chain": selected.posterior_draws,
            "thin": selected.thin,
            "total_retained_draws": int(len(all_draws)),
            "coefficient_prior": (
                f"independent Normal(0, {selected.coefficient_prior_scale:g}^2)"
            ),
            "fighter_ability_prior": (
                "Normal(0, population_variance); population variance has an "
                "inverse-gamma prior and is sampled"
            ),
            "mean_posterior_fighter_ability_scale": float(np.mean(ability_scales)),
            "maximum_absolute_chain_mean_probability_difference": (
                float(np.max(chain_differences)) if len(chain_differences) else None
            ),
            "mean_absolute_chain_mean_probability_difference": (
                float(np.mean(chain_differences)) if len(chain_differences) else None
            ),
            "p95_absolute_chain_mean_probability_difference": (
                float(np.quantile(chain_differences, 0.95))
                if len(chain_differences)
                else None
            ),
            "unseen_fighter_ability": (
                "sampled from Normal(0, population_variance) in every posterior draw"
            ),
        },
    )

"""Fully Bayesian logistic winner model with learned group shrinkage.

All matchup coefficients are sampled with Hamiltonian Monte Carlo whose mass
matrix is shaped by a Laplace approximation. Feature-group variances use exact
inverse-gamma conditional draws. The approximation only improves movement; a
Metropolis correction preserves the exact target posterior. The model is
research-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .hierarchical_bayes import (
    HierarchicalBayesPrediction,
    _numeric_matrix,
    _rms_scale,
    _validate_frame,
)


@dataclass(frozen=True)
class BayesianLogisticConfig:
    burn_in: int = 300
    posterior_draws: int = 300
    thin: int = 1
    chains: int = 2
    grouped_shrinkage: bool = True
    variance_prior_shape: float = 3.0
    variance_prior_scale: float = 0.08
    hmc_step_size: float = 0.18
    hmc_leapfrog_steps: int = 10
    hmc_target_acceptance: float = 0.75
    maximum_optimization_iterations: int = 500
    seed: int = 20260901

    def __post_init__(self) -> None:
        if self.burn_in < 1 or self.posterior_draws < 1:
            raise ValueError("Bayesian logistic burn-in and draws must be positive")
        if self.thin < 1 or self.chains < 1:
            raise ValueError("Bayesian logistic thinning and chains must be positive")
        if not math.isfinite(self.hmc_step_size) or self.hmc_step_size <= 0.0:
            raise ValueError("Bayesian logistic HMC step size must be positive")
        if self.hmc_leapfrog_steps < 2:
            raise ValueError("Bayesian logistic needs at least two leapfrog steps")
        if not 0.5 <= self.hmc_target_acceptance < 1.0:
            raise ValueError("Bayesian logistic HMC acceptance target is invalid")
        if self.maximum_optimization_iterations < 50:
            raise ValueError("Bayesian logistic optimization limit is too small")
        prior = (self.variance_prior_shape, self.variance_prior_scale)
        if not all(math.isfinite(value) and value > 0.0 for value in prior):
            raise ValueError("Bayesian logistic variance prior must be positive")
        if self.variance_prior_shape <= 1.0:
            raise ValueError("Bayesian logistic variance prior must have a finite mean")


def feature_group(feature: str) -> str:
    """Assign one stable, human-readable shrinkage group to a feature."""

    value = feature.casefold()
    if "elo" in value or "rating" in value:
        return "rating"
    if any(token in value for token in ("age", "height", "reach")):
        return "physical"
    if any(
        token in value
        for token in ("td_", "control_", "sub_attempt", "reversal")
    ):
        return "grappling"
    if any(token in value for token in ("sig_", "total_", "knockdown")):
        return "striking"
    if any(token in value for token in ("days_since", "has_history", "fights_log")):
        return "activity_experience"
    if any(
        token in value
        for token in (
            "wins_log",
            "losses_log",
            "win_rate",
            "finish_win_rate",
            "finish_loss_rate",
        )
    ):
        return "record_results"
    return "other"


def feature_groups(
    feature_columns: Sequence[str], *, grouped: bool
) -> tuple[str, ...]:
    if not grouped:
        return tuple("all_features" for _ in feature_columns)
    return tuple(feature_group(feature) for feature in feature_columns)


def _joint_log_posterior_gradient(
    parameters: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    group_index: np.ndarray,
    group_count: int,
    prior_shape: float,
    prior_scale: float,
) -> tuple[float, np.ndarray]:
    coefficient_count = design.shape[1]
    coefficients = parameters[:coefficient_count]
    log_variances = parameters[coefficient_count:]
    if len(log_variances) != group_count:
        raise ValueError("Bayesian logistic parameter dimension is inconsistent")
    inverse_variances = np.exp(-np.clip(log_variances, -40.0, 40.0))
    linear = design @ coefficients
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    log_posterior = float(
        np.sum(target * linear - np.logaddexp(0.0, linear))
    )
    coefficient_gradient = design.T @ (target - probability)
    log_variance_gradient = np.empty(group_count, dtype=float)
    for group in range(group_count):
        members = group_index == group
        squared_sum = float(coefficients[members] @ coefficients[members])
        inverse_variance = inverse_variances[group]
        log_posterior -= (
            (np.sum(members) / 2.0 + prior_shape) * log_variances[group]
            + (squared_sum / 2.0 + prior_scale) * inverse_variance
        )
        coefficient_gradient[members] -= (
            coefficients[members] * inverse_variance
        )
        log_variance_gradient[group] = -(
            np.sum(members) / 2.0 + prior_shape
        ) + (squared_sum / 2.0 + prior_scale) * inverse_variance
    gradient = np.concatenate([coefficient_gradient, log_variance_gradient])
    if not math.isfinite(log_posterior) or not np.isfinite(gradient).all():
        raise RuntimeError("Bayesian logistic posterior became non-finite")
    return log_posterior, gradient


def _laplace_proposal(
    design: np.ndarray,
    target: np.ndarray,
    group_index: np.ndarray,
    group_count: int,
    config: BayesianLogisticConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    coefficient_count = design.shape[1]
    prior_mean_variance = (
        config.variance_prior_scale / (config.variance_prior_shape - 1.0)
    )
    initial = np.concatenate(
        [
            np.zeros(coefficient_count, dtype=float),
            np.full(group_count, math.log(prior_mean_variance), dtype=float),
        ]
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = _joint_log_posterior_gradient(
            parameters,
            design,
            target,
            group_index,
            group_count,
            config.variance_prior_shape,
            config.variance_prior_scale,
        )
        return -value, -gradient

    bounds = [(None, None)] * coefficient_count + [
        (math.log(1e-8), math.log(100.0))
    ] * group_count
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": config.maximum_optimization_iterations,
            "ftol": 1e-12,
            "gtol": 1e-7,
        },
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Bayesian logistic posterior mode failed: {result.message}")
    mode = np.asarray(result.x, dtype=float)
    coefficients = mode[:coefficient_count]
    log_variances = mode[coefficient_count:]
    inverse_variances = np.exp(-log_variances)
    linear = design @ coefficients
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    weights = probability * (1.0 - probability)
    precision = design.T @ (weights[:, None] * design)
    coefficient_diagonal = np.arange(coefficient_count)
    precision[coefficient_diagonal, coefficient_diagonal] += inverse_variances[
        group_index
    ]
    jitter = 1e-10
    for _ in range(8):
        try:
            cholesky = np.linalg.cholesky(
                precision + np.eye(coefficient_count, dtype=float) * jitter
            )
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise RuntimeError("Bayesian logistic proposal precision is not positive")
    return mode, cholesky, {
        "optimizer_iterations": int(result.nit),
        "optimizer_gradient_maximum": float(np.max(np.abs(result.jac))),
        "precision_jitter": jitter,
    }


def _coefficient_log_posterior_gradient(
    coefficients: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    inverse_variance_by_coefficient: np.ndarray,
) -> tuple[float, np.ndarray]:
    linear = design @ coefficients
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    log_posterior = float(
        np.sum(target * linear - np.logaddexp(0.0, linear))
        - 0.5
        * np.sum(np.square(coefficients) * inverse_variance_by_coefficient)
    )
    gradient = (
        design.T @ (target - probability)
        - coefficients * inverse_variance_by_coefficient
    )
    return log_posterior, gradient


def _one_chain(
    *,
    training_x: np.ndarray,
    prediction_x: np.ndarray,
    target: np.ndarray,
    group_index: np.ndarray,
    group_names: tuple[str, ...],
    posterior_mode: np.ndarray,
    mass_cholesky: np.ndarray,
    config: BayesianLogisticConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, list[float]], dict[str, object]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    coefficient_count = training_x.shape[1]
    coefficients = posterior_mode[:coefficient_count].copy()
    group_variances = np.exp(posterior_mode[coefficient_count:]).copy()
    mass_inverse = np.linalg.solve(
        mass_cholesky.T,
        np.linalg.solve(mass_cholesky, np.eye(coefficient_count, dtype=float)),
    )
    total_iterations = config.burn_in + config.posterior_draws * config.thin
    predictions: list[np.ndarray] = []
    retained_scales = {name: [] for name in group_names}
    retained_acceptance_probabilities: list[float] = []
    retained_accepted = 0
    log_step_size = math.log(config.hmc_step_size)

    for iteration in range(total_iterations):
        inverse_variance_by_coefficient = (
            1.0 / group_variances[group_index]
        )
        current_log_posterior, current_gradient = (
            _coefficient_log_posterior_gradient(
                coefficients,
                training_x,
                target,
                inverse_variance_by_coefficient,
            )
        )
        initial_momentum = mass_cholesky @ rng.normal(size=coefficient_count)
        proposal_coefficients = coefficients.copy()
        proposal_momentum = (
            initial_momentum + math.exp(log_step_size) * current_gradient / 2.0
        )
        leapfrog_steps = max(
            2,
            config.hmc_leapfrog_steps + int(rng.integers(-2, 3)),
        )
        proposal_log_posterior = current_log_posterior
        proposal_gradient = current_gradient
        for step in range(leapfrog_steps):
            proposal_coefficients += (
                math.exp(log_step_size) * (mass_inverse @ proposal_momentum)
            )
            proposal_log_posterior, proposal_gradient = (
                _coefficient_log_posterior_gradient(
                    proposal_coefficients,
                    training_x,
                    target,
                    inverse_variance_by_coefficient,
                )
            )
            if step + 1 < leapfrog_steps:
                proposal_momentum += math.exp(log_step_size) * proposal_gradient
        proposal_momentum += math.exp(log_step_size) * proposal_gradient / 2.0
        current_kinetic = 0.5 * float(
            initial_momentum @ (mass_inverse @ initial_momentum)
        )
        proposal_kinetic = 0.5 * float(
            proposal_momentum @ (mass_inverse @ proposal_momentum)
        )
        log_acceptance = min(
            0.0,
            proposal_log_posterior
            - proposal_kinetic
            - current_log_posterior
            + current_kinetic,
        )
        acceptance_probability = math.exp(log_acceptance)
        accepted = float(rng.random()) < acceptance_probability
        if accepted:
            coefficients = proposal_coefficients

        for index in range(len(group_names)):
            members = coefficients[group_index == index]
            posterior_shape = config.variance_prior_shape + len(members) / 2.0
            posterior_scale = (
                config.variance_prior_scale + float(members @ members) / 2.0
            )
            group_variances[index] = 1.0 / rng.gamma(
                shape=posterior_shape,
                scale=1.0 / posterior_scale,
            )

        if iteration < config.burn_in:
            adaptation_rate = 0.08 / math.sqrt(iteration + 1.0)
            log_step_size += adaptation_rate * (
                acceptance_probability - config.hmc_target_acceptance
            )
            log_step_size = float(
                np.clip(log_step_size, math.log(1e-4), math.log(2.0))
            )
            continue
        if (iteration - config.burn_in) % config.thin:
            continue
        predictions.append(
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(prediction_x @ coefficients, -40.0, 40.0)
                )
            )
        )
        for index, name in enumerate(group_names):
            retained_scales[name].append(math.sqrt(group_variances[index]))
        retained_acceptance_probabilities.append(acceptance_probability)
        retained_accepted += int(accepted)

    draws = np.asarray(predictions, dtype=float)
    if draws.shape != (config.posterior_draws, len(prediction_x)):
        raise RuntimeError("Bayesian logistic sampler retained the wrong draw count")
    return draws, retained_scales, {
        "final_step_size": math.exp(log_step_size),
        "retained_acceptance_rate": retained_accepted / config.posterior_draws,
        "mean_retained_acceptance_probability": float(
            np.mean(retained_acceptance_probabilities)
        ),
    }


def bayesian_logistic_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: BayesianLogisticConfig | None = None,
) -> HierarchicalBayesPrediction:
    """Fit learned shrinkage scales and return posterior win probabilities."""

    selected = config or BayesianLogisticConfig()
    features = tuple(feature_columns)
    if not features:
        raise ValueError("Bayesian logistic model needs at least one feature")
    _validate_frame(training, features, labels=True)
    _validate_frame(prediction, features, labels=False)
    raw_training = _numeric_matrix(training, features)
    raw_prediction = _numeric_matrix(prediction, features)
    scale = _rms_scale(raw_training)
    training_x = raw_training / scale
    prediction_x = raw_prediction / scale
    target = training["target"].to_numpy(dtype=int)
    assigned_groups = feature_groups(features, grouped=selected.grouped_shrinkage)
    group_names = tuple(sorted(set(assigned_groups)))
    group_lookup = {name: index for index, name in enumerate(group_names)}
    group_index = np.asarray(
        [group_lookup[name] for name in assigned_groups], dtype=int
    )
    posterior_mode, mass_cholesky, proposal_diagnostics = (
        _laplace_proposal(
            training_x,
            target,
            group_index,
            len(group_names),
            selected,
        )
    )

    chain_draws: list[np.ndarray] = []
    group_scales = {name: [] for name in group_names}
    sampler_chains: list[dict[str, object]] = []
    chain_means: list[np.ndarray] = []
    for chain in range(selected.chains):
        draws, scales, chain_diagnostics = _one_chain(
            training_x=training_x,
            prediction_x=prediction_x,
            target=target,
            group_index=group_index,
            group_names=group_names,
            posterior_mode=posterior_mode,
            mass_cholesky=mass_cholesky,
            config=selected,
            seed=selected.seed + chain * 1_000_003,
        )
        chain_draws.append(draws)
        chain_means.append(np.mean(draws, axis=0))
        sampler_chains.append(chain_diagnostics)
        for name in group_names:
            group_scales[name].extend(scales[name])

    all_draws = np.concatenate(chain_draws, axis=0)
    lag_one_correlations: list[np.ndarray] = []
    for draws in chain_draws:
        centered = draws - np.mean(draws, axis=0)
        denominator = np.sum(np.square(centered), axis=0)
        correlation = np.divide(
            np.sum(centered[:-1] * centered[1:], axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-15,
        )
        lag_one_correlations.append(correlation)
    all_lag_one = np.concatenate(lag_one_correlations)
    chain_differences = (
        np.ptp(np.asarray(chain_means), axis=0)
        if selected.chains > 1
        else np.asarray([], dtype=float)
    )
    probability = np.mean(all_draws, axis=0)
    lower = np.quantile(all_draws, 0.05, axis=0)
    upper = np.quantile(all_draws, 0.95, axis=0)
    posterior_group_scales = {
        name: {
            "mean": float(np.mean(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "feature_count": int(np.sum(group_index == group_lookup[name])),
        }
        for name, values in group_scales.items()
    }
    return HierarchicalBayesPrediction(
        probability=probability,
        lower_probability=lower,
        upper_probability=upper,
        diagnostics={
            "model": "fully_bayesian_logistic_learned_group_shrinkage",
            "training_fights": len(training),
            "features": len(features),
            "feature_groups": list(group_names),
            "grouped_shrinkage": selected.grouped_shrinkage,
            "variance_prior_shape": selected.variance_prior_shape,
            "variance_prior_scale": selected.variance_prior_scale,
            "posterior_group_coefficient_scales": posterior_group_scales,
            "sampler": "Laplace-preconditioned Hamiltonian Monte Carlo",
            "hmc_initial_step_size": selected.hmc_step_size,
            "hmc_leapfrog_steps_center": selected.hmc_leapfrog_steps,
            "hmc_target_acceptance": selected.hmc_target_acceptance,
            "proposal_diagnostics": proposal_diagnostics,
            "chains": selected.chains,
            "burn_in_per_chain": selected.burn_in,
            "posterior_draws_per_chain": selected.posterior_draws,
            "thin": selected.thin,
            "total_retained_draws": len(all_draws),
            "sampler_chains": sampler_chains,
            "mean_lag_one_probability_autocorrelation": float(
                np.mean(all_lag_one)
            ),
            "p95_lag_one_probability_autocorrelation": float(
                np.quantile(all_lag_one, 0.95)
            ),
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
        },
    )

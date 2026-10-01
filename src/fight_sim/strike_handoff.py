"""Research-only preservation of the strike observation model.

No default snapshot or engine mode imports this module. The mapping operates on
a single pre-fight matchup: observed outcomes, duration and action counts are
not arguments. All inverse calculations enforce units, not outcome fit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from .domain import Side, SimulationRunSpec
from .opponent_audit import (
    OBSERVATION_TARGETS, OpponentAdjustmentAuditConfig, _ContextFit,
    _context_key, _target_arrays, fit_bout_clustered_two_way_effects,
)
from .strike_bridge_audit import integrate, neutral_phase_rates

STRIKE_TARGETS = OBSERVATION_TARGETS[:2]


def weighted_observation_context(frame, target, weights, config=None):
    """Original observation context formula, with card multiplicities."""
    settings = config or OpponentAdjustmentAuditConfig()
    numerator, exposure, valid = _target_arrays(frame, target)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(frame) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid observation weights")
    valid &= weights > 0
    if not valid.any():
        raise ValueError("no weighted observations")
    total = float(np.sum(numerator[valid] * weights[valid]))
    denominator = float(np.sum(exposure[valid] * weights[valid]))
    if target.kind == "rate":
        global_value = (total + .5) / (denominator + .5 / target.seed)
        prior = settings.context_rate_prior_minutes
    else:
        global_value = (total + 40 * target.seed) / (denominator + 40)
        prior = settings.context_probability_prior_attempts
    observed = frame.loc[valid, ["division", "era"]].copy()
    observed["numerator"] = numerator[valid] * weights[valid]
    observed["exposure"] = exposure[valid] * weights[valid]
    grouped = observed.groupby(["division", "era"], sort=True)[["numerator", "exposure"]].sum()
    return _ContextFit(global_value, {
        _context_key(division, era): (row.numerator + prior * global_value) / (row.exposure + prior)
        for (division, era), row in grouped.iterrows()
    })


@dataclass(frozen=True)
class StrikeEffects:
    target_name: str
    actor: dict[str, float]
    opponent: dict[str, float]

    def predict(self, baseline, actors, opponents):
        effects = np.array([self.actor.get(str(actor), 0) for actor in actors])
        if self.target_name == "strike_pace":
            return np.clip(np.asarray(baseline) * np.exp(effects), 1e-6, 100)
        effects += np.array([self.opponent.get(str(other), 0) for other in opponents])
        return np.clip(expit(logit(baseline) + effects), 1e-6, 1 - 1e-6)


def fit_strike_effects(training, target, baseline, weights, ridge, *, cutoff):
    """Fit only strictly earlier physical fighter-side bouts."""
    cutoff = pd.to_datetime(cutoff, utc=True)
    dates = pd.to_datetime(training.date, utc=True)
    if pd.isna(cutoff) or training.empty or dates.isna().any() or dates.ge(cutoff).any():
        raise ValueError("strike training must be strictly before the cutoff")
    numerator, exposure, valid = _target_arrays(training, target)
    baseline, weights = np.asarray(baseline, float), np.asarray(weights, float)
    if len(baseline) != len(training) or len(weights) != len(training):
        raise ValueError("training/baseline/weight length mismatch")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid effect weights")
    valid &= np.isfinite(baseline) & (baseline > 0) & (weights > 0)
    if target.kind != "rate":
        valid &= baseline < 1
    if not valid.any():
        raise ValueError("no weighted effect observations")
    if target.kind == "rate":
        stabilized = (numerator[valid] + .5) / (exposure[valid] + .5 / baseline[valid])
        residual = np.log(stabilized / baseline[valid])
    else:
        smoothed = np.clip((numerator[valid] + .5) / (exposure[valid] + 1), 1e-6, 1 - 1e-6)
        residual = logit(smoothed) - logit(baseline[valid])
    actor, opponent = fit_bout_clustered_two_way_effects(
        training.loc[valid, "fighter_id"], training.loc[valid, "opponent_id"], residual,
        ridge=ridge, sample_weights=weights[valid],
    )
    return StrikeEffects(target.name, actor, opponent)


def covariate_multiplier(frame, effects):
    """The existing v2 covariate formula; used only in diagnostic arms."""
    age = np.asarray(frame["age_years"], float)
    experience = np.asarray(frame["experience_fights"], float)
    layoff = np.asarray(frame["layoff_days"], float)
    age = np.where(np.isfinite(age), (age - 30) / 10, effects.get("age_center", 0))
    experience = np.log1p(np.maximum(np.nan_to_num(experience, nan=0), 0))
    layoff = np.where(np.isfinite(layoff), np.log1p(np.maximum(layoff, 0) / 365.25), effects.get("log_layoff_years_center", 0))
    adjustment = (
        effects.get("age_per_decade", 0) * (age - effects.get("age_center", 0))
        + effects.get("log_experience", 0) * (experience - effects.get("log_experience_center", 0))
        + effects.get("log_layoff_years", 0) * (layoff - effects.get("log_layoff_years_center", 0))
    )
    return np.clip(np.exp(adjustment), .6, 1.4)


def context_occupancy(context):
    return {
        "distance": context["distance_phase_share"],
        "clinch": context["clinch_phase_share"],
        "ground_top": context["ground_phase_share"] / 2,
        "ground_bottom": context["ground_phase_share"] / 2,
        "scramble": 0.0,
    }


def map_matchup_strikes(spec: SimulationRunSpec, targets, occupancy):
    """Preserve each side's predicted neutral pace and combined hit probability.

    `targets` is {Side: (attempts_per_minute, landing_probability)}. Ground top
    and bottom are separate occupancy states. The entire original non-strike
    parameter vector, mechanics configuration and seeds are retained.
    """
    mapped = replace(spec, **{
        side.value: replace(getattr(spec, side.value), parameters=replace(
            getattr(spec, side.value).parameters, strike_defense=.5, strike_accuracy=.5
        )) for side in Side
    })
    for side in Side:
        rate, probability = map(float, targets[side])
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("invalid strike mapping target")
        snapshot = getattr(mapped, side.value)
        original = snapshot.parameters

        def scale_spec(scale):
            fields = {f"strike_rate_{phase}": getattr(original, f"strike_rate_{phase}") * scale
                      for phase in ("distance", "clinch", "ground")}
            return replace(mapped, **{side.value: replace(snapshot, parameters=replace(original, **fields))})

        base_rate = integrate(neutral_phase_rates(mapped, side), occupancy[side])[0]
        if base_rate <= 0:
            raise ValueError("cannot normalize zero strike intensity")
        scale = rate / base_rate
        proposed = scale_spec(scale)
        actual = integrate(neutral_phase_rates(proposed, side), occupancy[side])[0]
        if not math.isclose(actual, rate, rel_tol=1e-10, abs_tol=1e-10):
            def residual(value):
                return integrate(neutral_phase_rates(scale_spec(value), side), occupancy[side])[0] - rate
            upper = max(1.0, scale)
            for _ in range(50):
                if residual(upper) >= 0:
                    break
                upper *= 2
            else:
                raise ValueError("target pace exceeds engine hazard capacity")
            if residual(0) > 0:
                raise ValueError("target pace is below engine hazard floor")
            proposed = scale_spec(brentq(residual, 0, upper, xtol=1e-12))
        mapped = proposed
        snapshot = getattr(mapped, side.value)

        def accuracy_spec(value):
            return replace(mapped, **{side.value: replace(snapshot, parameters=replace(snapshot.parameters, strike_accuracy=value))})

        def effective_accuracy(value):
            attempts, landed = integrate(neutral_phase_rates(accuracy_spec(value), side), occupancy[side])
            return landed / attempts

        guess = probability * .5 / effective_accuracy(.5)
        if not 0 <= guess <= 1 or not math.isclose(effective_accuracy(guess), probability, rel_tol=1e-10, abs_tol=1e-10):
            if not effective_accuracy(0) <= probability <= effective_accuracy(1):
                raise ValueError("target accuracy outside attainable engine clipping bounds")
            guess = brentq(lambda value: effective_accuracy(value) - probability, 0, 1, xtol=1e-12)
        mapped = accuracy_spec(guess)
    for side in Side:
        rate, landed = integrate(neutral_phase_rates(mapped, side), occupancy[side])
        if not np.allclose([rate, landed / rate], targets[side], rtol=1e-8, atol=1e-8):
            raise ValueError("mapped engine failed its neutral strike contract")
    return mapped

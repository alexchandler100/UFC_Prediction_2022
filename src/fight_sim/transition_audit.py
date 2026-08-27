"""Causal audits for sparse fighter-specific transition candidates.

UFCStats publishes round totals, not action timestamps.  Consequently these
targets are deliberately named ``same_round_association``: a knockdown and a
KO/TKO in the same round do not prove that the recorded knockdown caused the
finish, and credited control in a takedown round is not observed top-position
time.  The audit tests whether strongly pooled fighter/opponent histories add
held-out predictive information before any such candidate can enter the
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

import numpy as np
import pandas as pd

from fight_semantics import method_bucket
from ufc_round_data import ROUND_DATA_COLUMNS, validate_normalized_round_stats


TRANSITION_AUDIT_SCHEMA_VERSION = 1
_EPSILON = 1e-6


@dataclass(frozen=True)
class TransitionAuditConfig:
    holdout_latest_events: int = 5
    context_prior_opportunities: float = 25.0
    fighter_prior_opportunities: float = 12.0
    bootstrap_replicates: int = 2000
    random_seed: int = 41041
    max_runtime_seconds: float = 3000.0
    as_of: object | None = None

    def validate(self) -> None:
        if self.holdout_latest_events < 1:
            raise ValueError("holdout_latest_events must be positive")
        if self.context_prior_opportunities <= 0:
            raise ValueError("context_prior_opportunities must be positive")
        if self.fighter_prior_opportunities <= 0:
            raise ValueError("fighter_prior_opportunities must be positive")
        if not 100 <= self.bootstrap_replicates <= 10000:
            raise ValueError("bootstrap_replicates must be between 100 and 10000")
        if not 0 < self.max_runtime_seconds <= 3300:
            raise ValueError("max_runtime_seconds must be in (0, 3300]")


class TransitionAuditTimeLimit(RuntimeError):
    """Raised before an audit can overrun its caller-declared compute budget."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TransitionAuditTimeLimit("transition audit reached its compute budget")


def _clip_probability(value: float) -> float:
    return float(np.clip(value, _EPSILON, 1.0 - _EPSILON))


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _expit(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _round_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, ROUND_DATA_COLUMNS].copy()
    normalized = normalized.sort_values("round_stat_id", kind="stable")
    payload = normalized.to_csv(
        index=False, lineterminator="\n", na_rep="<NA>"
    ).encode("utf-8")
    return sha256(payload).hexdigest()


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


def build_transition_opportunities(
    round_stats: pd.DataFrame,
    *,
    as_of: object | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build source-honest conditional targets from reconciled round rows."""

    if round_stats.empty:
        raise ValueError("round statistics are empty; run the bounded round backfill first")
    validate_normalized_round_stats(round_stats)
    frame = round_stats.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, errors="raise", utc=True)
        if isinstance(cutoff, pd.DatetimeIndex):
            raise TypeError("as_of must be a scalar timestamp")
        frame = frame.loc[frame["date"].lt(cutoff)].copy()
    if frame.empty:
        raise ValueError("no round rows exist strictly before as_of")

    reconciliation = frame["reconciliation_status"].astype("string").str.casefold()
    eligible = frame.loc[reconciliation.eq("matched")].copy()
    excluded_rows = int(len(frame) - len(eligible))
    if eligible.empty:
        raise ValueError("no fully reconciled round rows are available")

    numeric = (
        "round",
        "finish_round",
        "round_seconds",
        "knockdowns",
        "takedowns_landed",
        "sub_attempts",
        "control",
    )
    for column in numeric:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    eligible["method_bucket"] = eligible["method"].map(method_bucket)
    eligible["won"] = eligible["result"].astype("string").str.upper().eq("W")
    eligible["finish_same_round"] = eligible["round"].eq(eligible["finish_round"])
    identity = [
        "date",
        "event_id",
        "fight_id",
        "fighter_id",
        "fighter",
        "opponent_id",
        "opponent",
        "division",
        "round",
    ]

    def binary_target(
        opportunity: pd.Series,
        outcome: pd.Series,
        required: tuple[str, ...],
    ) -> pd.DataFrame:
        known = eligible.loc[:, required].notna().all(axis=1)
        result = eligible.loc[known & opportunity, identity].copy()
        result["actual"] = outcome.loc[result.index].astype(int)
        return result.reset_index(drop=True)

    knockdown = eligible["knockdowns"].gt(0)
    takedown = eligible["takedowns_landed"].gt(0)
    same_round_ko = (
        eligible["won"]
        & eligible["finish_same_round"]
        & eligible["method_bucket"].eq("ko_tko")
    )
    same_round_submission = (
        eligible["won"]
        & eligible["finish_same_round"]
        & eligible["method_bucket"].eq("submission")
    )
    targets = {
        "knockdown_round_to_ko_tko_same_round_association": binary_target(
            knockdown,
            same_round_ko,
            ("knockdowns", "round", "finish_round"),
        ),
        "takedown_round_to_submission_attempt_same_round_association": binary_target(
            takedown,
            eligible["sub_attempts"].gt(0),
            ("takedowns_landed", "sub_attempts"),
        ),
        "takedown_round_to_submission_win_same_round_association": binary_target(
            takedown,
            same_round_submission,
            ("takedowns_landed", "round", "finish_round"),
        ),
    }

    control_known = eligible[["takedowns_landed", "control", "round_seconds"]].notna().all(axis=1)
    control = eligible.loc[
        control_known & takedown & eligible["round_seconds"].gt(0), identity
    ].copy()
    control["actual"] = (
        eligible.loc[control.index, "control"]
        / eligible.loc[control.index, "round_seconds"]
    ).clip(0.0, 1.0)
    targets["takedown_round_to_credited_control_share_same_round_association"] = (
        control.reset_index(drop=True)
    )

    metadata = {
        "round_data_sha256": _round_frame_sha256(frame),
        "source_round_rows": int(len(frame)),
        "fully_reconciled_round_rows": int(len(eligible)),
        "excluded_unreconciled_round_rows": excluded_rows,
        "physical_fights": int(eligible["fight_id"].nunique()),
        "event_cards": int(eligible["event_id"].nunique()),
        "source_limitation": (
            "UFCStats round totals are interval-censored: same-round association "
            "does not establish action order or causal conversion. Credited CTRL "
            "does not identify top versus bottom position."
        ),
    }
    return targets, metadata


def _source_event_order(
    round_stats: pd.DataFrame, *, as_of: object | None
) -> pd.DataFrame:
    """Return every reconciled source card, including cards with no target event."""

    frame = round_stats.loc[:, ["date", "event_id", "reconciliation_status"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, errors="raise", utc=True)
        frame = frame.loc[frame["date"].lt(cutoff)]
    frame = frame.loc[
        frame["reconciliation_status"].astype("string").str.casefold().eq("matched")
    ]
    return (
        frame[["date", "event_id"]]
        .drop_duplicates()
        .sort_values(["date", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def _counts(frame: pd.DataFrame, keys: list[str]) -> dict[object, tuple[float, int]]:
    grouped = frame.groupby(keys, sort=False)["actual"].agg(["sum", "count"])
    if len(keys) == 1:
        return {
            key: (float(row["sum"]), int(row["count"]))
            for key, row in grouped.iterrows()
        }
    return {
        (key if isinstance(key, tuple) else (key,)): (
            float(row["sum"]),
            int(row["count"]),
        )
        for key, row in grouped.iterrows()
    }


def _context_key(row: Mapping[str, object]) -> tuple[str, int]:
    return str(row["division"]), int(row["round"])


def _binary_predictions(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TransitionAuditConfig,
) -> pd.DataFrame:
    successes = float(development["actual"].sum())
    opportunities = int(len(development))
    global_mean = (successes + 0.5) / (opportunities + 1.0)
    contexts = _counts(development, ["division", "round"])
    actors = _counts(development, ["fighter_id"])
    defenders = _counts(development, ["opponent_id"])
    predictions = holdout.copy()
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    actor_history: list[int] = []
    defender_history: list[int] = []
    for row in holdout.to_dict("records"):
        context_successes, context_n = contexts.get(_context_key(row), (0.0, 0))
        context_mean = (
            context_successes
            + config.context_prior_opportunities * global_mean
        ) / (context_n + config.context_prior_opportunities)
        actor_successes, actor_n = actors.get(str(row["fighter_id"]), (0.0, 0))
        defender_successes, defender_n = defenders.get(
            str(row["opponent_id"]), (0.0, 0)
        )
        actor_mean = (
            actor_successes
            + config.fighter_prior_opportunities * context_mean
        ) / (actor_n + config.fighter_prior_opportunities)
        defender_mean = (
            defender_successes
            + config.fighter_prior_opportunities * context_mean
        ) / (defender_n + config.fighter_prior_opportunities)
        # Averaging the two shrunken log-odds is intentionally conservative:
        # equal actor and defender evidence reproduces that shared propensity,
        # while a single sparse side cannot create an extreme prediction.
        candidate = _expit((_logit(actor_mean) + _logit(defender_mean)) / 2.0)
        baseline_values.append(_clip_probability(context_mean))
        candidate_values.append(_clip_probability(candidate))
        actor_history.append(actor_n)
        defender_history.append(defender_n)
    predictions["context_probability"] = baseline_values
    predictions["fighter_opponent_probability"] = candidate_values
    predictions["actor_prior_opportunities"] = actor_history
    predictions["opponent_prior_opportunities"] = defender_history
    return predictions


def _continuous_predictions(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TransitionAuditConfig,
) -> pd.DataFrame:
    global_mean = float(development["actual"].mean())
    contexts = _counts(development, ["division", "round"])
    actors = _counts(development, ["fighter_id"])
    defenders = _counts(development, ["opponent_id"])
    predictions = holdout.copy()
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    actor_history: list[int] = []
    defender_history: list[int] = []
    for row in holdout.to_dict("records"):
        context_total, context_n = contexts.get(_context_key(row), (0.0, 0))
        context_mean = (
            context_total + config.context_prior_opportunities * global_mean
        ) / (context_n + config.context_prior_opportunities)
        actor_total, actor_n = actors.get(str(row["fighter_id"]), (0.0, 0))
        defender_total, defender_n = defenders.get(str(row["opponent_id"]), (0.0, 0))
        actor_mean = (
            actor_total + config.fighter_prior_opportunities * context_mean
        ) / (actor_n + config.fighter_prior_opportunities)
        defender_mean = (
            defender_total + config.fighter_prior_opportunities * context_mean
        ) / (defender_n + config.fighter_prior_opportunities)
        baseline_values.append(float(np.clip(context_mean, 0.0, 1.0)))
        candidate_values.append(float(np.clip((actor_mean + defender_mean) / 2.0, 0.0, 1.0)))
        actor_history.append(actor_n)
        defender_history.append(defender_n)
    predictions["context_mean"] = baseline_values
    predictions["fighter_opponent_mean"] = candidate_values
    predictions["actor_prior_opportunities"] = actor_history
    predictions["opponent_prior_opportunities"] = defender_history
    return predictions


def _event_block_interval(
    deltas: np.ndarray,
    event_ids: np.ndarray,
    *,
    replicates: int,
    random_seed: int,
    deadline: float,
) -> dict[str, float]:
    unique = np.unique(event_ids)
    blocks = [deltas[event_ids == event] for event in unique]
    rng = np.random.default_rng(random_seed)
    sampled = np.empty(replicates, dtype=float)
    for index in range(replicates):
        if index % 50 == 0:
            _check_deadline(deadline)
        choices = rng.integers(0, len(blocks), size=len(blocks))
        numerator = sum(float(blocks[item].sum()) for item in choices)
        denominator = sum(int(len(blocks[item])) for item in choices)
        sampled[index] = numerator / denominator
    return {
        "mean_delta": float(deltas.mean()),
        "event_block_95_interval_low": float(np.quantile(sampled, 0.025)),
        "event_block_95_interval_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(sampled < 0.0)),
    }


def _binary_result(
    target: str,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TransitionAuditConfig,
    deadline: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    predictions = _binary_predictions(development, holdout, config)
    actual = predictions["actual"].to_numpy(dtype=float)
    baseline = predictions["context_probability"].to_numpy(dtype=float)
    candidate = predictions["fighter_opponent_probability"].to_numpy(dtype=float)
    baseline_loss = -(actual * np.log(baseline) + (1.0 - actual) * np.log(1.0 - baseline))
    candidate_loss = -(actual * np.log(candidate) + (1.0 - actual) * np.log(1.0 - candidate))
    baseline_brier = np.square(actual - baseline)
    candidate_brier = np.square(actual - candidate)
    interval = _event_block_interval(
        candidate_loss - baseline_loss,
        predictions["event_id"].astype(str).to_numpy(),
        replicates=config.bootstrap_replicates,
        random_seed=config.random_seed + int(sha256(target.encode()).hexdigest()[:6], 16),
        deadline=deadline,
    )
    adequate = (
        len(development) >= 100
        and len(holdout) >= 30
        and 5 <= int(holdout["actual"].sum()) <= len(holdout) - 5
    )
    retained = adequate and interval["event_block_95_interval_high"] < 0.0
    report = {
        "kind": "binary",
        "development_opportunities": int(len(development)),
        "holdout_opportunities": int(len(holdout)),
        "development_successes": int(development["actual"].sum()),
        "holdout_successes": int(holdout["actual"].sum()),
        "context_log_loss": float(baseline_loss.mean()),
        "fighter_opponent_log_loss": float(candidate_loss.mean()),
        "context_brier": float(baseline_brier.mean()),
        "fighter_opponent_brier": float(candidate_brier.mean()),
        "paired_log_loss": interval,
        "evidence_adequate_for_mechanic": adequate,
        "candidate_retained": retained,
        "decision": (
            "retain_for_separate_simulator-mechanic validation"
            if retained
            else "do not alter simulator"
        ),
    }
    predictions.insert(0, "target", target)
    predictions.insert(1, "kind", "binary")
    return report, predictions


def _continuous_result(
    target: str,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TransitionAuditConfig,
    deadline: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    predictions = _continuous_predictions(development, holdout, config)
    actual = predictions["actual"].to_numpy(dtype=float)
    baseline = predictions["context_mean"].to_numpy(dtype=float)
    candidate = predictions["fighter_opponent_mean"].to_numpy(dtype=float)
    baseline_absolute = np.abs(actual - baseline)
    candidate_absolute = np.abs(actual - candidate)
    interval = _event_block_interval(
        candidate_absolute - baseline_absolute,
        predictions["event_id"].astype(str).to_numpy(),
        replicates=config.bootstrap_replicates,
        random_seed=config.random_seed + int(sha256(target.encode()).hexdigest()[:6], 16),
        deadline=deadline,
    )
    adequate = len(development) >= 100 and len(holdout) >= 50
    retained = adequate and interval["event_block_95_interval_high"] < 0.0
    report = {
        "kind": "continuous",
        "development_opportunities": int(len(development)),
        "holdout_opportunities": int(len(holdout)),
        "context_mae": float(baseline_absolute.mean()),
        "fighter_opponent_mae": float(candidate_absolute.mean()),
        "context_rmse": float(np.sqrt(np.square(actual - baseline).mean())),
        "fighter_opponent_rmse": float(np.sqrt(np.square(actual - candidate).mean())),
        "paired_absolute_error": interval,
        "evidence_adequate_for_mechanic": adequate,
        "candidate_retained": retained,
        "decision": (
            "retain for separate simulator-mechanic validation"
            if retained
            else "do not alter simulator"
        ),
    }
    predictions.insert(0, "target", target)
    predictions.insert(1, "kind", "continuous")
    return report, predictions


def run_transition_audit(
    round_stats: pd.DataFrame,
    *,
    config: TransitionAuditConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate pooled fighter/opponent effects on a locked latest-event holdout."""

    settings = config or TransitionAuditConfig()
    settings.validate()
    started_at = time.monotonic()
    deadline = started_at + settings.max_runtime_seconds
    targets, source = build_transition_opportunities(round_stats, as_of=settings.as_of)
    _check_deadline(deadline)

    event_order = _source_event_order(round_stats, as_of=settings.as_of)
    if len(event_order) <= settings.holdout_latest_events:
        raise ValueError(
            "round data do not contain enough event cards for the requested holdout"
        )
    holdout_ids = set(event_order.tail(settings.holdout_latest_events)["event_id"].astype(str))
    development_ids = set(event_order.iloc[:-settings.holdout_latest_events]["event_id"].astype(str))
    results: dict[str, object] = {}
    prediction_frames: list[pd.DataFrame] = []
    warnings = [source["source_limitation"]]
    for target, opportunities in targets.items():
        _check_deadline(deadline)
        development = opportunities[
            opportunities["event_id"].astype(str).isin(development_ids)
        ].copy()
        holdout = opportunities[
            opportunities["event_id"].astype(str).isin(holdout_ids)
        ].copy()
        if development.empty or holdout.empty:
            results[target] = {
                "status": "insufficient_opportunities",
                "development_opportunities": int(len(development)),
                "holdout_opportunities": int(len(holdout)),
                "candidate_retained": False,
                "decision": "do not alter simulator",
            }
            warnings.append(f"{target} lacks development or holdout opportunities")
            continue
        if "control_share" in target:
            result, predictions = _continuous_result(
                target, development, holdout, settings, deadline
            )
        else:
            result, predictions = _binary_result(
                target, development, holdout, settings, deadline
            )
        results[target] = result
        prediction_frames.append(predictions)
        if not result["evidence_adequate_for_mechanic"]:
            warnings.append(f"{target} is below its predeclared minimum evidence threshold")

    elapsed = time.monotonic() - started_at
    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False)
        if prediction_frames
        else pd.DataFrame()
    )
    report: dict[str, object] = {
        "schema_version": TRANSITION_AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "candidate_only": True,
        "production_behavior_changed": False,
        "source": source,
        "split": {
            "development_event_cards": len(development_ids),
            "holdout_event_cards": len(holdout_ids),
            "holdout_event_ids": sorted(holdout_ids),
            "strict_as_of": None if settings.as_of is None else str(settings.as_of),
        },
        "pooling": {
            "context_prior_opportunities": settings.context_prior_opportunities,
            "fighter_prior_opportunities": settings.fighter_prior_opportunities,
            "actor_opponent_combination": "mean of strongly pooled actor/opponent logits or means",
        },
        "targets": results,
        "warnings": warnings,
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "max_runtime_seconds": settings.max_runtime_seconds,
            "bootstrap_replicates": settings.bootstrap_replicates,
            "random_seed": settings.random_seed,
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report, predictions


def execute_transition_audit(
    *,
    round_path: str | Path,
    output: str | Path,
    predictions_output: str | Path,
    config: TransitionAuditConfig | None = None,
) -> tuple[dict[str, object], Path, Path]:
    source = Path(round_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"round statistics do not exist: {source}; run python -m fight_sim backfill"
        )
    report, predictions = run_transition_audit(
        pd.read_csv(source, low_memory=False), config=config
    )
    report_path = _atomic_json(output, report)
    predictions_path = _atomic_csv(predictions_output, predictions)
    return report, report_path, predictions_path
